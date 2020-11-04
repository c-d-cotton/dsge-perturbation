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

import copy
import numpy as np
import pickle
import re
import sympy

# Defaults:{{{1
futureending_default = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'futureending_default')

# Convert string list into sympy list:{{{1
def convertstringlisttosympy(eqs_string):
    """
    Realised can just do directly with sympy.sympify.
    """
    from sympy.abc import _clash

    eqs = sympy.zeros(len(eqs_string),1)

    for i in range(len(eqs_string)):
        eqs[i,0] = sympy.sympify(eqs_string[i], locals = _clash)

    return(eqs)


# Computing Steady State:{{{1
def get_ss_eqs(equations, replacedict):
    """
    Evaluates the equations at steady state and returns the result.
    With a standard model, this should be zero.
    But if I'm adding a second regime where a constraint binds occasionally or something like this, some of the equations may be nonzero.
    """
    # equations_copy = copy.deepcopy(equations)
    # equations_copy = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'replacevarsinstringlist')(equations_copy, replacedict)
    # for i in range(len(equations_copy)):
    #     equations_copy[i] = importattr(__projectdir__ / Path('submodules/python-math-func/stringvar_func.py'), 'evalstring')(equations_copy[i])

    equations_copy = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'subsympymatrix_string')(equations, replacedict, inputtype = 'listofstrings')

    return(equations_copy)


# Main Derivative Decomposition:{{{1
def dsgeanalysisdiff(Et_eqs, statesshocks, controls, futureending = futureending_default):

    # sympy.diff(Et_eqs, states)
    fxe = Et_eqs.jacobian([sympy.Symbol(var) for var in statesshocks])
    fxep = Et_eqs.jacobian([sympy.Symbol(var + futureending) for var in statesshocks])
    fy = Et_eqs.jacobian([sympy.Symbol(var) for var in controls])
    fyp = Et_eqs.jacobian([sympy.Symbol(var + futureending) for var in controls])
    
    return(fxe, fxep, fy, fyp)
    

def dsgeanalysisdiff_split(Et_eqs, states, controls, shocks, futureending = futureending_default):
    # sympy.diff(Et_eqs, states)
    fx = Et_eqs.jacobian([sympy.Symbol(var) for var in states])
    fxp = Et_eqs.jacobian([sympy.Symbol(var + futureending) for var in states])
    fy = Et_eqs.jacobian([sympy.Symbol(var) for var in controls])
    fyp = Et_eqs.jacobian([sympy.Symbol(var + futureending) for var in controls])
    if len(shocks) > 0:
        fe = Et_eqs.jacobian([sympy.Symbol(var) for var in shocks])
        fep = Et_eqs.jacobian([sympy.Symbol(var + futureending) for var in shocks])
    else:
        fe = np.empty([len(Et_eqs), 0])
        fep = np.empty([len(Et_eqs), 0])

    return(fx, fxp, fy, fyp, fe, fep)
    
    
def convertsplitxe(fxe, fxep, numstates):
    fx = fxe[:, 0: numstates]
    fe = fxe[:, numstates: ]

    fxp = fxep[:, 0: numstates]
    fep = fxep[:, numstates: ]

    return(fx, fxp, fe, fep)


def convertjoinxe(fx, fxp, fe, fep):
    if isinstance(fx, np.ndarray):
        # numpy matrices
        fxe = np.concatenate((fx, fe), axis = 1)
        fxep = np.concatenate((fxp, fep), axis = 1)
    else:
        # sympy matrices
        fxe = fx.row_join(fe)
        fxep = fxp.row_join(fep)

    return(fxe, fxep)


# Main Derivative Decomposition Inputdict Functions:{{{1
def getfxefy_inputdict(inputdict):
    # check equations correct length
    if len(inputdict['equations']) != len(inputdict['states']) + len(inputdict['controls']):
        print('Number of equations: ' + str(len(inputdict['equations'])))
        print('Number of states + controls: ' + str(len(inputdict['states']) + len(inputdict['controls'])))
        raise ValueError('Number of equations does not match the number of states and controls.')

    # Et_eqs
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting add Et_eqs')

    # determine which coefficients and variables use
    if 'fxefy_cancelparams' not in inputdict:
        inputdict['fxefy_cancelparams'] = True
    if inputdict['fxefy_cancelparams'] is True:
        equations = 'equations_noparams_plus'
    else:
        equations = 'equations_plus'

    inputdict['Et_eqs'] = importattr(__projectdir__ / Path('dsgediff_func.py'), 'convertstringlisttosympy')(inputdict[equations])

    # find analytical derivatives
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute analytical derivatives')
    inputdict['fxe'], inputdict['fxep'], inputdict['fy'], inputdict['fyp'] = importattr(__projectdir__ / Path('dsgediff_func.py'), 'dsgeanalysisdiff')(inputdict['Et_eqs'], inputdict['states_plus'], inputdict['controls'])

    return(inputdict)

        
def getnfxenfy_inputdict(inputdict):
    """
    Only use this without doing gxhx when doing loglin and lin check
    """


    inputdict = importattr(__projectdir__ / Path('dsgediff_func.py'), 'getfxefy_inputdict')(inputdict)

    if inputdict['fxefy_cancelparams'] is True:
        replacedict = 'varfplusonlyssdict'
    else:
        replacedict = 'replaceuseddict'


    # use a method that works with missing params if necessary
    # lambdify method does work but converts the matrices into 
    # if inputdict['missingparams'] is True:
    #     inputdict['fxefy_evalf'] = True

    # apply replacedict to get numerical derivatives - if necessary
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute numerical derivatives')
    matriceslist = [inputdict['fxe'], inputdict['fxep'], inputdict['fy'], inputdict['fyp']]
    returnlist = []
    if inputdict['loglineareqs'] is True and inputdict['fxefy_cancelparams'] is True and inputdict['missingparams'] is False:
        # don't bother doing numerical replace if loglineareqs since then the equations should have been linear to begin with so there should be no symbols in matriceslist
        for matrix in matriceslist:
            nmatrix = np.array(matrix)
            returnlist.append(nmatrix)
    else:
        for matrix in matriceslist:
            # allow for different replace methods since replacing sympy variables can be slow
            if 'fxefy_evalf' in inputdict and inputdict['fxefy_evalf'] is True:
                # this does sympy.evalf
                nmatrix = matrix.evalf(subs = inputdict[replacedict])
            elif 'fxefy_subquick' in inputdict and inputdict['fxefy_subquick'] is True:
                # this basically just applies the usual sympy replace but only for used elements
                nmatrix = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'matrixsubsquick')(matrix, inputdict[replacedict])
            elif 'fxefy_substringsympy' in inputdict and inputdict['fxefy_substringsympy'] is True:
                # replace by converting matrix to string and then replacing
                # seems to work most quickly but I worry about errors when doing the conversion
                nmatrix = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'subsympymatrix_string')(matrix, inputdict[replacedict])
            else:
                # this converts the matrix to a function and then inputs the relevant arguments to the function
                # may not work with <Python 3.7 since before then lambdify only allowed <99 args
                nmatrix = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'lambdifysubs')(matrix, inputdict[replacedict])
                
            returnlist.append(nmatrix)

    inputdict['nfxe'], inputdict['nfxep'], inputdict['nfy'], inputdict['nfyp'] = returnlist

    return(inputdict)


# Partial Evaluation of Derivatives:{{{1
def funcstoeval_inputdict(inputdict):

    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute numerical derivative functions')

    if 'fxefy_f_usenumerical' not in inputdict:
        inputdict['fxefy_f_usenumerical'] = False

    if inputdict['fxefy_f_usenumerical'] is True:
        inputlist = [inputdict['nfxe'], inputdict['nfxep'], inputdict['nfy'], inputdict['nfyp']]
    else:
        inputlist = [inputdict['fxe'], inputdict['fxep'], inputdict['fy'], inputdict['fyp']]

    retlist = []
    for matrix in inputlist:
        retlist.append(importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'lambdifysubs_lambdifyonly')(matrix))

    inputdict['fxe_f'], inputdict['fxep_f'], inputdict['fy_f'], inputdict['fyp_f'] = [ret[0] for ret in retlist]
    inputdict['fxe_sortedvars'], inputdict['fxep_sortedvars'], inputdict['fy_sortedvars'], inputdict['fyp_sortedvars'] = [ret[1] for ret in retlist]

    return(inputdict)

def funcstoeval_inputdict_save(inputdict, savefolder):
    """
    sympy lambdify functions don't pickle
    I use dill with the recurse option to save them separately
    I can then load them using funcstoeval_inputdict_load
    """
    import dill
    # this option needed to save sympy lambdify functions
    dill.settings['recurse'] = True

    savelambdifylist = ['fxe_f', 'fxep_f', 'fy_f', 'fyp_f']
    for savelambdify in savelambdifylist:
        with open(os.path.join(savefolder, savelambdify + '.dill'), 'wb') as f:
            dill.dump(inputdict[savelambdify], f)
        # need to delete them to prevent the rest of the inputdict pickle from failing
        del inputdict[savelambdify]

    with open(os.path.join(savefolder, 'inputdict.pickle'), 'wb') as f:
        pickle.dump(inputdict, f)


def funcstoeval_inputdict_load(savefolder):
    """
    sympy lambdify functions don't pickle
    I use dill with the recurse option to save them separately in funcstoeval_inputdict_save
    This function loads the full inputdict from a savefolder
    """
    import dill

    with open(os.path.join(savefolder, 'inputdict.pickle'), 'rb') as f:
        inputdict = pickle.load(f)


    savelambdifylist = ['fxe_f', 'fxep_f', 'fy_f', 'fyp_f']
    for savelambdify in savelambdifylist:
        with open(os.path.join(savefolder, savelambdify + '.dill'), 'rb') as f:
            inputdict[savelambdify] = dill.load(f)

    return(inputdict)


def partialtofulleval_quick_inputdict(inputdict, fullparamssdict):
    """
    Applies a set of parameters to fxe, fxep, fy, fyp

    This is like partialtofulleval_inputdict
    However, I only create just the replacedict necessary to do the numerical evaluation and I don't put the results into inputdict
    """

    # complete the replacedict
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'completereplacedict_inputdict')(inputdict, fullparamssdict)

    matriceslist = [inputdict['fxe_f'], inputdict['fxep_f'], inputdict['fy_f'], inputdict['fyp_f']]
    sortedvarslist = [inputdict['fxe_sortedvars'], inputdict['fxep_sortedvars'], inputdict['fy_sortedvars'], inputdict['fyp_sortedvars']]
    newmatriceslist = []
    for i in range(len(matriceslist)):
        
        newmatrix = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'lambdifysubs_subsonly')(matriceslist[i], sortedvarslist[i], fullparamssdict)

        newmatriceslist.append(newmatrix)

    return(newmatriceslist)


def partialtofulleval_inputdict(inputdict, fullparamssdict):
    """
    Like partialtofulleval_inputdict except that I basically just back out the full inputdict that I would have got if I'd just done everything at the same time.
    """

    # now have a complete paramssdict so not missingparams anymore
    # this is necessary to get the dicts describing parameters correctly
    inputdict['missingparams'] = False
    inputdict['useparamssdictonly'] = True
    inputdict['paramssdict'] = fullparamssdict
    for var in inputdict['shocksplusoptional']:
        # replace if not exist (since most shocks are 0 so saves time) of if using log linear equations (when all shocks should be zero)
        if var + inputdict['ssending1'] not in inputdict['paramssdict']:
            inputdict['paramssdict'][var + inputdict['ssending1']] = 0
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getfullreplacedict_inputdict')(inputdict)

    matriceslist = [inputdict['fxe_f'], inputdict['fxep_f'], inputdict['fy_f'], inputdict['fyp_f']]
    sortedvarslist = [inputdict['fxe_sortedvars'], inputdict['fxep_sortedvars'], inputdict['fy_sortedvars'], inputdict['fyp_sortedvars']]
    newmatriceslist = []
    for i in range(len(matriceslist)):
        
        newmatrix = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'lambdifysubs_subsonly')(matriceslist[i], sortedvarslist[i], inputdict['replaceuseddict'])

        newmatriceslist.append(newmatrix)

    inputdict['nfxe'], inputdict['nfxep'], inputdict['nfy'], inputdict['nfyp'] = newmatriceslist

    return(inputdict)


# Reshaping Derivative Decompositions:{{{1
def get_ABCDE_form(fx, fxp, fy, fyp, fe, fep, ss_eqs):
    """
    Convert model into ABCDE format
    fx, fxp etc. need to be numerical i.e. no symbols

    Note that I may need to input fx etc. excluding the shock equations i.e. equations rather than equations_plus
    """
    nstates = np.shape(fx)[1]
    ncontrols = np.shape(fy)[1]
    nZ = nstates + ncontrols
    Azero = np.zeros([nZ, nZ])

    A = copy.deepcopy(Azero)
    A[:, 0: nstates] = np.array(fx)
    
    B = copy.deepcopy(Azero)
    B[:, 0: nstates] = np.array(fxp)
    B[:, nstates: ] = np.array(fy)

    C = copy.deepcopy(Azero)
    C[:, nstates: ] = np.array(fyp)

    E = np.array(fe)

    # getting constant D
    # we just replace in the ssdict into the equations and then evaluate
    D = np.array(ss_eqs)

    # convert to floats since numpy has problems working with integer arrays
    # this is particularly an issue with the E matrix
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)
    D = D.astype(float)
    E = E.astype(float)

    return(A, B, C, D, E)


def ABCDE_form_full(equations, states, controls, shocks, varfplusonlyssdict):
    """
    Express equations as AZ_{t - 1} + BZ_t + CZ_{t + 1} + D + E\epsilon_t
    Z_t = [states_{t + 1} controls_t], \epsilon_t = vector of shocks at t

    Doesn't work if not linear.
    """
    Et_eqs = importattr(__projectdir__ / Path('dsgediff_func.py'), 'convertstringlisttosympy')(equations)

    fx, fxp, fy, fyp, fe, fep = importattr(__projectdir__ / Path('dsgediff_func.py'), 'dsgeanalysisdiff_split')(Et_eqs, states, controls, shocks)

    # solve for ss_eqs
    ss_eqs = importattr(__projectdir__ / Path('dsgediff_func.py'), 'get_ss_eqs')(equations, varfplusonlyssdict)

    A, B, C, D, E = importattr(__projectdir__ / Path('dsgediff_func.py'), 'get_ABCDE_form')(fx, fxp, fy, fyp, fe, fep, ss_eqs)

    return(A, B, C, D, E)


# Check Computation of Policy Functions:{{{1
def checksame_inputdict(inputdict1, inputdict2):
    """
    This function allows me to compare the results from a linearized model to a non-lineariezed model to verify that I have not made a mistake in the linearized model.
    I can also use this to compare a log model with a log-linearized model.

    Equations need to be in same order and use the same variables
    """
    # ensure checking steady state - since I sometimes turn this off
    inputdict1['skipsscheck'] = False
    inputdict2['skipsscheck'] = False

    # get the basic model
    inputdict1 = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getmodel_inputdict')(inputdict1)
    inputdict2 = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getmodel_inputdict')(inputdict2)

    # add fx fxp fy fyp
    # nonlinearised
    inputdict1 = getnfxenfy_inputdict(inputdict1)
    # linearised
    inputdict2 = getnfxenfy_inputdict(inputdict2)

    # combine relevant vectors
    fullvars = inputdict1['states_plus'] + [state + '_p' for state in inputdict1['states_plus']] + inputdict1['controls'] + [control + '_p' for control in inputdict1['controls']]

    fullmat1 = np.concatenate((inputdict1['nfxe'], inputdict1['nfxep'], inputdict1['nfy'], inputdict1['nfyp']), axis = 1)
    fullmat2 = np.concatenate((inputdict2['nfxe'], inputdict2['nfxep'], inputdict2['nfy'], inputdict2['nfyp']), axis = 1)

    numeq = np.shape(fullmat1)[0]
    numvars = np.shape(fullmat1)[1]
    sameall = True
    for row in range(numeq):
        # get first element of row that does not equal approximately zero
        colstar = None
        for col in range(numvars):
            # verify first that I can get float versions of each row and column of fullmat
            try:
                float(fullmat1[row][col])
            except Exception:
                print('Following equation failed:')
                print(inputdict1['equations'][row])
                print('For following element:')
                print(fullvars[col])
                print('Returned:')
                print(fullmat1[row][col])
            try:
                float(fullmat2[row][col])
            except Exception:
                print('Following equation failed:')
                print(inputdict2['equations'][row])
                print('For following element:')
                print(fullvars[col])
                print('Returned:')
                print(fullmat2[row][col])
        
            # get which column is nonzero for fullmat2 so scaling does not fail
            if colstar is None and np.abs(fullmat2[row][col]) > 1e-8:
                colstar = col
        if colstar is not None:
            # scale row by difference between nonlinear and linear matrix
            fullmat2[row] = fullmat2[row] * fullmat1[row][colstar] / fullmat2[row][colstar]
        else:
            print('Failed row due to no variables for model2: ' + str(row))

        # same is False when rows are different
        same = True
        for col in range(numvars):
            if abs(fullmat1[row][col] - fullmat2[row][col]) > 1e-8:
                same = False
                break
        
        # print if same is False
        if same is False:
            sameall = False
            print('')

            # print equation
            print(inputdict1['equations_plus'][row])
            print(inputdict2['equations_plus'][row])


            diffcols = [col for col in range(numvars) if np.abs(fullmat1[row][col] - fullmat2[row][col]) > 1e-8]
            print([fullvars[col] for col in diffcols])
            print([fullmat1[row][col] - fullmat2[row][col] for col in diffcols])




            # nonzerocols_nonlin = [col for col in range(numvars) if np.abs(fullmat1[row][col]) > 1e-8]
            # print([fullvars[col] for col in nonzerocols_nonlin])
            # print([fullmat1[row][col] for col in nonzerocols_nonlin])
            #
            # nonzerocols_lin = [col for col in range(numvars) if np.abs(fullmat2[row][col]) > 1e-8]
            # print([fullvars[col] for col in nonzerocols_lin])
            # print([fullmat2[row][col] for col in nonzerocols_lin])

    if sameall is True:
        # print('Two models are the same.')
        None
    else:
        raise ValueError('ERROR: TWO MODELS ARE DIFFERENT.')

    


def checksame_inputdict_test():
    inputdict = {}
    states = ['X_tm1']
    controls = ['A', 'B', 'X']
    varssdict = {'X': 1, 'A': 2, 'B': 0.5, 'X_tm1': 1}
    basicinputdict = {'states': states, 'controls': controls, 'varssdict': varssdict}

    nonlinequations = [
    'X = X_tm1'
    ,
    'A = 2'
    ,
    'A * B = X_tm1'
    ,
    'X ** 1.5 = 1'
    ]
    linequations = [
    'X = X_tm1'
    ,
    'A = 0' # correct
    # 'B = 0' # incorrect
    ,
    'A + B = X_tm1'
    ,
    '1.5 * X = 0'
    ]

    inputdict_nonlin = copy.deepcopy(basicinputdict)
    inputdict_nonlin['equations'] = nonlinequations
    inputdict_nonlin['logvars'] = True
    inputdict_lin = copy.deepcopy(basicinputdict)
    inputdict_lin['equations'] = linequations
    inputdict_lin['loglineareqs'] = True

    importattr(__projectdir__ / Path('dsgediff_func.py'), 'checksame_inputdict')(inputdict_nonlin, inputdict_lin)

