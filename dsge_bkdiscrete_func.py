#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np

from dsgesetup_func import getmodel_inputdict

sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/statespace')))
from statespace_func import irgraphs_multiplelines

# Compute policy function:{{{1
def gxhx(fxe, fxep, fy, fyp, Nvar = None, NK = None, raiseerror = True):
    """
    Here I just translate Schmitt-Grohe and Uribe's gx_hx matlab code into Python.
    Schmitt-Grohe Uribe codes do 2 step Schur stuff:
    1. find number of eigenvalues by non-ordered eigenvalues to get number of states
    2. reorder eigenvalues

    """
    import numpy as np
    from scipy.linalg import ordqz
    # from np.lin_alg import matrix_rank

    # STEP ZERO: STACK APPROPRIATELY

    # get NK and Nvar if not specified
    if NK is None or Nvar is None:
        dim = fxe.shape
        Nvar = dim[0]
        NK = dim[1]

    # use np matrices rather than sympy matrices
    # A = fxep.row_join(fyp)
    # A = -A
    A = np.concatenate((fxep, fyp), axis = 1)

    # B = fxe.row_join(fy)
    B = -np.concatenate((fxe, fy), axis = 1)

    # old - can delete:{{{
    # 20250529
    # used to get number of states without upper diagonal Schur and then apply upper diagonal Schur
    # I'm not sure why I did this as it seems inefficient
    # it also could yield errors for the non-ouc Schur but not the ouc Schur
    # so now I switched to do the Schur decomp only for ouc and get states from that
    # I can probably delete this but I'm leaving it for the moment until I'm sure the new method works


    # # get number of states
    # ret_temp = ordqz(A, B, sort = 'ouc')
    # try:
    #     # reorder states in desired order
    #     ret_temp = ordqz(A, B)
    # except:
    #     print('ordqz failed.')
    #     print('fxe:')
    #     print(fxe)
    #     print('fxep:')
    #     print(fxep)
    #     print('fy:')
    #     print(fy)
    #     print('fyp:')
    #     print(fyp)
    #
    #     ret_temp = ordqz(A, B)
    # AA = ret_temp[0]
    # BB = ret_temp[1]
    # alpha = ret_temp[2]
    # beta = ret_temp[3]
    # nk = 0
    # for i in range(Nvar):
    #     if np.absolute(beta[i]) < np.absolute(alpha[i]):
    #         nk = nk + 1
    # old - can delete:}}}

    # STEP ONE: UPPER_DIAGONAL SCHUR

    # reorder
    try:
        # reorder states in desired order
        ret = ordqz(A, B, sort = 'ouc')
    except:
        print('ordqz failed.')
        print('fxe:')
        print(fxe)
        print('fxep:')
        print(fxep)
        print('fy:')
        print(fy)
        print('fyp:')
        print(fyp)

        ret = ordqz(A, B, sort = 'ouc')
    s = ret[0]
    t = ret[1]
    alpha = ret[2]
    beta = ret[3]
    q = ret[4]
    z = ret[5]

    # STEP TWO: NUMBER OF STATES

    nk = 0
    for i in range(Nvar):
        if np.absolute(beta[i]) < np.absolute(alpha[i]):
            nk = nk + 1

    if nk < NK:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(NK)
        print('No local equilibrium exists.')
        if raiseerror is True:
            raise ValueError('')
    if nk > NK:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(NK)
        print('The equilibrium is locally indeterminate.')
        if raiseerror is True:
            raise ValueError('')

    # STEP THREE: BACK OUT SOLVED POLICY FUNCTIONS

    # Here I go ahead and use dimensions of states I specified rather than dimensions suggested by eigenvalues.

    z21 = z[(NK):, 0: NK]
    z11 = z[0: NK, 0: NK]

    # return error if can't invert
    if np.linalg.matrix_rank(z11) < NK:
        raise ValueError('Invertibility condition violated.')


    s11 = s[0: NK, 0: NK]
    t11 = t[0: NK, 0: NK]

    z11i = np.linalg.inv(z11)

    gx = np.dot(z21, z11i)
    hx = np.dot(np.dot(np.dot(z11, np.linalg.inv(s11)), t11), z11i)

    return(gx, hx)

        


def gxhx_inputdict(inputdict):

    # get policy functions
    # raise an error if I have the wrong number of states?
    from dsgesetup_func import printruntime
    printruntime(inputdict, 'Starting compute policy functions')
    if 'gxhxraise' not in inputdict:
        inputdict['gxhxraise'] = True
    inputdict['gx'], inputdict['hx'] = gxhx(inputdict['nfxe'], inputdict['nfxep'], inputdict['nfy'], inputdict['nfyp'], raiseerror = inputdict['gxhxraise'])

    return(inputdict)

def polfunc_inputdict(inputdict):
    """
    Get policy function through a dictionary.
    Necessary arguments: equations_plus, replacedict, states_plus, controls, 
    Optional: lambdifyok, evalequals, gxhxraise, continuous
    """
    from dsgediff_func import getfxefy_inputdict
    inputdict = getfxefy_inputdict(inputdict)
    from dsgediff_func import getnfxenfy_inputdict
    inputdict = getnfxenfy_inputdict(inputdict)
    inputdict = gxhx_inputdict(inputdict)

    return(inputdict)


# Policy Functions:{{{1
def gxhx_splitbyshocks(gx, hx, numshocks):
    """
    gxhx matrices include shocks. Get matrices without these.

    (i,j) shows response of variable i to state value j
    """
    numstatesshocks = hx.shape[0]
    numstates = numstatesshocks - numshocks
    gx_noshocks = gx[:, : numstates]
    gx_shocks = gx[:, numstates: ]
    hx_noshocks = hx[: numstates, : numstates]
    # still want to look at response of non-shock states (to shocks)
    hx_shocks = hx[: numstates, numstates: ]
    return(gx_noshocks, gx_shocks, hx_noshocks, hx_shocks)


def irmatrix(gx, hx, X0, T = None):
    """
    Return as X Y concatenated matrix for simplicity.

    BASE THIS ON STATESPACE FUNCTION INSTEAD.
    """
    import numpy as np
    
    # adjust defaults
    if T is None:
        T = 40

    gxshape = np.shape(gx)
    nx = gxshape[1]
    ny = gxshape[0]

    X = np.empty([T, nx])
    Y = np.empty([T, ny])

    for t in range(T):
        if t == 0:
            X[t] = np.reshape(X0, [nx])
        else:
            X[t] = np.reshape(np.dot(hx, X[t - 1]), [nx])

        Y[t] = np.reshape(np.dot(gx, X[t]), [ny])

    XY = np.concatenate((X, Y), axis = 1)
    return(XY)
    

# Policy Functions Inputdict:{{{1
def interpretpolfunc_inputdict(inputdict):

    # Parse arguments:{{{
    if 'printmain' not in inputdict:
        inputdict['printmain'] = True

    # use separate shocks dict here since I might want to include states or exclude certain shocks
    if 'irfshocks' not in inputdict:
        inputdict['irfshocks'] = inputdict['shocks']
    if 'irfshockdict' not in inputdict:
        inputdict['irfshockdict'] = {}
    for shock in inputdict['irfshocks']:
        if shock not in inputdict['irfshockdict']:
            inputdict['irfshockdict'][shock] = 1
    if 'showirfs' not in inputdict:
        inputdict['showirfs'] = inputdict['irfshocks']


    # number of periods for shocks to IRF
    if 'irf_T' not in inputdict:
        inputdict['irf_T'] = 40

    if 'savefolderlatexname' not in inputdict:
        # the name I give to the savefolder in Latex files
        inputdict['savefolderlatexname'] = inputdict['savefolder']
    # Parse arguments:}}}

    # Policy Functions Post-Compute Adjustments:{{{
    from dsgesetup_func import printruntime
    printruntime(inputdict, 'Starting adjust policy functions')

    # policy function options
    if 'adjustpolfunc' not in inputdict:
        inputdict['adjustpolfunc'] = True

    # adjust policy function
    # set terms close to 0 to be 0
    if inputdict['adjustpolfunc'] is True:
        inputdict['hx'][np.absolute(inputdict['hx']) < 1e-15] = 0
        inputdict['gx'][np.absolute(inputdict['gx']) < 1e-15] = 0

    if inputdict['printmain'] is True:
        print('\nPolicy Functions:')
        print('hx - state matrix')
        print(inputdict['hx'])
        print('gx - control matrix')
        print(inputdict['gx'])
    # Policy Functions Post-Compute General:}}}

    alltex = ''

    inputdict['gx_noshocks'], inputdict['gx_shocks'], inputdict['hx_noshocks'], inputdict['hx_shocks'] = gxhx_splitbyshocks(inputdict['gx'], inputdict['hx'], len(inputdict['shocks']))

    # Policy Functions Latex:{{{
    if 'polfunclatexdecimal' not in inputdict:
        inputdict['polfunclatexdecimal'] = 3
    if len(inputdict['states']) > 0 and inputdict['savefolder'] is not None:
        from dsgesetup_func import printruntime
        printruntime(inputdict, 'Starting write up policy functions for latex')

        inputdict['hx_onlystates'] = inputdict['hx'][0: len(inputdict['states']), 0: len(inputdict['states'])]
        inputdict['hx_shocks'] = inputdict['hx'][len(inputdict['states']): , 0: len(inputdict['states'])]
        inputdict['hx_shocks'] = inputdict['hx'][0: len(inputdict['states']), len(inputdict['states']): ]
        inputdict['gx_onlystates'] = inputdict['gx'][:, 0: len(inputdict['states'])]
        inputdict['gx_shocks'] = inputdict['gx'][:, len(inputdict['states']): ]

        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        hxlatex = genbasicmatrix(inputdict['hx'])
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        gxlatex = genbasicmatrix(inputdict['gx'])

        p = '\\begin{equation}'
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix([state + '\\_p' for state in inputdict['states']])
        p = p + ' = '
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix(inputdict['hx_onlystates'], decimalpoints = inputdict['polfunclatexdecimal'])
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix(inputdict['states'])
        if len(inputdict['shocks']) > 0:
            p = p + ' + '
            sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
            from tab_general_func import genbasicmatrix
            p = p + genbasicmatrix(inputdict['hx_shocks'], decimalpoints = inputdict['polfunclatexdecimal'])
            sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
            from tab_general_func import genbasicmatrix
            p = p + genbasicmatrix(inputdict['shocks'])
        p = p + '\\end{equation}'
        polfunclatex_states = p
        
        p = '\\begin{equation}'
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix(inputdict['controls'])
        p = p + ' = '
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix(inputdict['gx_onlystates'], decimalpoints = inputdict['polfunclatexdecimal'])
        sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
        from tab_general_func import genbasicmatrix
        p = p + genbasicmatrix(inputdict['states'])
        if len(inputdict['shocks']) > 0:
            p = p + ' + '
            sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
            from tab_general_func import genbasicmatrix
            p = p + genbasicmatrix(inputdict['gx_shocks'], decimalpoints = inputdict['polfunclatexdecimal'])
            sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
            from tab_general_func import genbasicmatrix
            p = p + genbasicmatrix(inputdict['shocks'])
        p = p + '\\end{equation}'
        polfunclatex_controls = p

        with open(Path(inputdict['savefolder']) / Path('polfunclatex_basic.tex'), 'w+') as f:
            f.write(polfunclatex_states + '\n\n')
            f.write(polfunclatex_controls + '\n\n')

        alltex = alltex + polfunclatex_states  + '\n\n' + polfunclatex_controls + '\n\n'

    # End Policy functions latex:}}}

    # IRFS:{{{
    from dsgesetup_func import printruntime
    printruntime(inputdict, 'Starting IRFs')

    # get IRFs
    inputdict['irf_XY'] = {}
    for shock in inputdict['irfshocks']:
        X0 = np.zeros(len(inputdict['states_plus']))
        # set up shock in irf to be 1 unless otherwise specified
        X0[inputdict['stateshockcontrolposdict'][shock]] = inputdict['irfshockdict'][shock]
        XY = irmatrix(inputdict['gx'], inputdict['hx'], X0, T = inputdict['irf_T'])
        # save overall irf
        inputdict['irf_XY'][shock] = XY

        irfvars = [inputdict['stateshockcontrolposdict'][varname] for varname in inputdict['mainvars']]
        irfnames = inputdict['mainvarnames']
        XY2 = XY[:, irfvars]
        if shock in inputdict['showirfs']:
            pltshow = True
        else:
            pltshow = False
        if inputdict['savefolder'] is not None:
            pltsavename = Path(inputdict['savefolder']) / Path('irf_' + shock + '.png')
        else:
            pltsavename = None
        if pltshow is True:
            print('Showing IRF for ' + shock + '.')
        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/statespace')))
        from statespace_func import irgraphs
        irgraphs(XY2, names = irfnames, pltshow = pltshow, pltsavename = pltsavename)

        if inputdict['savefolder'] is not None:
            alltex = alltex + '\\begin{figure}[H]\n'
            alltex = alltex + '\\centering\n'
            alltex = alltex + '\\caption{IRF of the Response to a Shock of ' + str(inputdict['irfshockdict'][shock]) + ' to ' + shock + '}\n'
            alltex = alltex + '\\includegraphics[width=0.8\\linewidth]{' + inputdict['savefolderlatexname'] + 'irf_' + shock + '.png}\n'
            alltex = alltex + '\\end{figure}\n\n'
    # }}}

    if inputdict['savefolder'] is not None:
        with open(Path(inputdict['savefolder']) / Path('summary.tex'), 'w+') as f:
            f.write(alltex + '\n\n')


# Full Run Inputdict:{{{1
def discretelineardsgefull(inputdict):
    """
    Run full DSGE model starting with equations and shocks inputted.
    Generates policy function, IRFs.
    Similar to Dynare (I prefer using this since I like writing in Python, I think the functions are easy to follow and I like using my own codes).
    """
    # get basic model
    inputdict = getmodel_inputdict(inputdict)

    # get basic model (after potential for python2dynare)
    if inputdict['savefolder'] is not None:
        # only add shocksddict for python2dynare
        from getshocks_func import getshocksddict_inputdict
        getshocksddict_inputdict(inputdict)

        # standard simulation for Dynare
        inputdict['python2dynare_simulation'] = 'stoch_simul(order=1);'

        from python2dynare_func import python2dynare_inputdict
        python2dynare_inputdict(inputdict)

    # add policy functions
    inputdict = polfunc_inputdict(inputdict)

    # interpret policy functions
    interpretpolfunc_inputdict(inputdict)
    
    return(inputdict)


# IRF Plot of Multiple Models:{{{1
def irfmultiplemodels(inputdicts, linenames, varnames, shockvar, T = 40, shocksize = 1, vardescs = None, pltsavename = None, pltshow = False, linestyles = None):
    """
    labelslist is the list of labels I want to give in the time plot
    inputdicts should be a list of inputdicts on which I have not yet run getmodel etc.
    shockvar is the var I want to shock in the IRF
    shocksize is the size of the shock I want to apply
    """
    if vardescs is None:
        vardescs = varnames

    XYfull = np.empty([len(inputdicts), T, len(varnames)])

    # get basic models
    for i in range(len(inputdicts)):
        inputdict = inputdicts[i]

        # run DSGE model if not already done
        if 'hx' not in inputdict:
            getmodel_inputdict(inputdict)
            polfunc_inputdict(inputdict)

        X0 = np.zeros(len(inputdict['states_plus']))
        # set up shock in irf to be 1 unless otherwise specified
        X0[inputdict['stateshockcontrolposdict'][shockvar]] = shocksize
        XY = irmatrix(inputdict['gx'], inputdict['hx'], X0, T = T)

        irfvars = [inputdict['stateshockcontrolposdict'][varname] for varname in varnames]
        XY2 = XY[:, irfvars]
        XYfull[i, :, :] = XY2

    irgraphs_multiplelines(XYfull, linenames = linenames, graphnames = vardescs, pltsavename = pltsavename, pltshow = pltshow, linestyles = linestyles)

