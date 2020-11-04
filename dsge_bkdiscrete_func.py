#!/usr/bin/env python3
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

import numpy as np

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

    # STEP ONE: GET NUMBER OF STATES

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

    # get number of states
    try:
        # reorder states in desired order
        ret_temp = ordqz(A, B)
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

        ret_temp = ordqz(A, B)
    AA = ret_temp[0]
    BB = ret_temp[1]
    alpha = ret_temp[2]
    beta = ret_temp[3]
    nk = 0
    for i in range(Nvar):
        if np.absolute(beta[i]) < np.absolute(alpha[i]):
            nk = nk + 1

    # STEP TWO: REORDER

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
    q = ret[4]
    z = ret[5]

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
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute policy functions')
    if 'gxhxraise' not in inputdict:
        inputdict['gxhxraise'] = True
    inputdict['gx'], inputdict['hx'] = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'gxhx')(inputdict['nfxe'], inputdict['nfxep'], inputdict['nfy'], inputdict['nfyp'], raiseerror = inputdict['gxhxraise'])

    return(inputdict)

def polfunc_inputdict(inputdict):
    """
    Get policy function through a dictionary.
    Necessary arguments: equations_plus, replacedict, states_plus, controls, 
    Optional: lambdifyok, evalequals, gxhxraise, continuous
    """
    inputdict = importattr(__projectdir__ / Path('dsgediff_func.py'), 'getfxefy_inputdict')(inputdict)
    inputdict = importattr(__projectdir__ / Path('dsgediff_func.py'), 'getnfxenfy_inputdict')(inputdict)
    inputdict = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'gxhx_inputdict')(inputdict)

    return(inputdict)


# Policy Functions:{{{1
def gxhx_splitbyshocks(gx, hx, numshocks):
    """
    gxhx matrices include shocks. Get matrices without these.
    """
    numstatesshocks = hx.shape[0]
    numstates = numstatesshocks - numshocks
    gx_noshocks = gx[:, : numstates]
    gx_shocks = gx[:, numstates: ]
    hx_noshocks = hx[: numstates, : numstates]
    hx_shocks = hx[numstates: , numstates: ]
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
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting adjust policy functions')

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

    inputdict['gx_noshocks'], inputdict['gx_shocks'], inputdict['hx_noshocks'], inputdict['hx_shocks'] = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'gxhx_splitbyshocks')(inputdict['gx'], inputdict['hx'], len(inputdict['shocks']))

    # Policy Functions Latex:{{{
    if 'polfunclatexdecimal' not in inputdict:
        inputdict['polfunclatexdecimal'] = 3
    if len(inputdict['states']) > 0 and inputdict['savefolder'] is not None:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting write up policy functions for latex')

        inputdict['hx_onlystates'] = inputdict['hx'][0: len(inputdict['states']), 0: len(inputdict['states'])]
        inputdict['hx_shocks'] = inputdict['hx'][len(inputdict['states']): , 0: len(inputdict['states'])]
        inputdict['hx_shocks'] = inputdict['hx'][0: len(inputdict['states']), len(inputdict['states']): ]
        inputdict['gx_onlystates'] = inputdict['gx'][:, 0: len(inputdict['states'])]
        inputdict['gx_shocks'] = inputdict['gx'][:, len(inputdict['states']): ]

        hxlatex = importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['hx'])
        gxlatex = importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['gx'])

        p = '\\begin{equation}'
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')([state + '\_p' for state in inputdict['states']])
        p = p + ' = '
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['hx_onlystates'], decimalpoints = inputdict['polfunclatexdecimal'])
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['states'])
        if len(inputdict['shocks']) > 0:
            p = p + ' + '
            p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['hx_shocks'], decimalpoints = inputdict['polfunclatexdecimal'])
            p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['shocks'])
        p = p + '\\end{equation}'
        polfunclatex_states = p
        
        p = '\\begin{equation}'
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['controls'])
        p = p + ' = '
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['gx_onlystates'], decimalpoints = inputdict['polfunclatexdecimal'])
        p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['states'])
        if len(inputdict['shocks']) > 0:
            p = p + ' + '
            p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['gx_shocks'], decimalpoints = inputdict['polfunclatexdecimal'])
            p = p + importattr(__projectdir__ / Path('submodules/python-texoutput/regoutput_func.py'), 'genbasicmatrix')(inputdict['shocks'])
        p = p + '\\end{equation}'
        polfunclatex_controls = p

        with open(Path(inputdict['savefolder']) / Path('polfunclatex_basic.tex'), 'w+') as f:
            f.write(polfunclatex_states + '\n\n')
            f.write(polfunclatex_controls + '\n\n')

        alltex = alltex + polfunclatex_states  + '\n\n' + polfunclatex_controls + '\n\n'

    # End Policy functions latex:}}}

    # IRFS:{{{
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting IRFs')

    # get IRFs
    inputdict['irf_XY'] = {}
    for shock in inputdict['irfshocks']:
        X0 = np.zeros(len(inputdict['states_plus']))
        # set up shock in irf to be 1 unless otherwise specified
        X0[inputdict['stateshockcontrolposdict'][shock]] = inputdict['irfshockdict'][shock]
        XY = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'irmatrix')(inputdict['gx'], inputdict['hx'], X0, T = inputdict['irf_T'])
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
        importattr(__projectdir__ / Path('submodules/python-math-func/statespace/statespace_func.py'), 'irgraphs')(XY2, names = irfnames, pltshow = pltshow, pltsavename = pltsavename)

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
    inputdict = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getmodel_inputdict')(inputdict)

    # get basic model (after potential for python2dynare)
    if inputdict['savefolder'] is not None:
        # only add shocksddict for python2dynare
        importattr(__projectdir__ / Path('getshocks_func.py'), 'getshocksddict_inputdict')(inputdict)

        # standard simulation for Dynare
        inputdict['python2dynare_simulation'] = 'stoch_simul(order=1);'

        importattr(__projectdir__ / Path('python2dynare_func.py'), 'python2dynare_inputdict')(inputdict)

    # add policy functions
    inputdict = polfunc_inputdict(inputdict)

    # interpret policy functions
    interpretpolfunc_inputdict(inputdict)
    
    return(inputdict)


# IRF Plot of Multiple Models:{{{1
def irfmultiplemodels(linenameslist, inputdictlist, plotvars, shockvar, T = 40, shocksize = 1, plotnames = None, graphswithlegend = None, pltsavename = None):
    """
    labelslist is the list of labels I want to give in the time plot
    inputdictlist should be a list of inputdicts on which I have not yet run getmodel etc.
    shockvar is the var I want to shock in the IRF
    shocksize is the size of the shock I want to apply
    """
    if plotnames is None:
        plotnames = plotvars

    XYfull = np.empty([len(inputdictlist), T, len(plotvars)])

    # get basic models
    for i in range(len(inputdictlist)):
        inputdict = inputdictlist[i]

        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getmodel_inputdict')(inputdict)
        importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'polfunc_inputdict')(inputdict)

        X0 = np.zeros(len(inputdict['states_plus']))
        # set up shock in irf to be 1 unless otherwise specified
        X0[inputdict['stateshockcontrolposdict'][shockvar]] = shocksize
        XY = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'irmatrix')(inputdict['gx'], inputdict['hx'], X0, T = T)

        irfvars = [inputdict['stateshockcontrolposdict'][varname] for varname in plotvars]
        XY2 = XY[:, irfvars]
        XYfull[i, :, :] = XY2

    importattr(__projectdir__ / Path('submodules/python-math-func/statespace/statespace_func.py'), 'irgraphs_multiplelines')(XYfull, linenames = linenameslist, graphnames = plotnames, pltsavename = pltsavename, graphswithlegend = graphswithlegend)

    