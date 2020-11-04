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
import pathlib
import re

# Defaults:{{{1
dotending_default = '_dot'
ssending1_default = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'ssending1_default')
ssending2_default = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'ssending2_default')

# DSGE Setup Functions:{{{1
def convertlogvariables_string_cont(equations, samenamelist = [], diffnamedict = None, varssdict = None, ssending1 = ssending1_default, ssending2 = ssending2_default, dotending = dotending_default):
    """
    This works with strings unlike next function which works with sympy vars.
    It also allows dynare inputs. The varssdict returned in the Dynare case would work with Dynare but not with my gxhx functions which require a s.s. defined for each period.

    diffnamedict['C'] = 'c' means replace C with exp(c) in equations.
    If c in samenamelist then replace c with exp(c).

    If specify varssdict then also convert varssdict. Consequently return variable number of arguments.
    Will want to convert varssdict as well if I computed varssdict before doing log conversion.
    ssending1 allows me to specify what the steady state variables in varssdict end in.
    """
    import numpy as np

    def subbasic(oldname, newname, string):
        # basic replace of A(t) with e^{a(t)}
        string = re.sub('(?<![a-zA-Z0-9_])' + oldname + '(?![a-zA-Z0-9\(_])', 'exp(' + newname + ')', string)
        return(string)

    def subdot(oldname, newname, string):
        # need to replace dA(t)/dt with da(t)/dt e^{a(t)}
        # oldname is A
        # newname is a
        string = re.sub('(?<![a-zA-Z0-9_])' + oldname + dotending + '(?![a-zA-Z0-9\(_])', 'exp(' + newname + ') * ' + newname + dotending, string)
        return(string)

    if diffnamedict is None:
        diffnamedict = {}
    for var in samenamelist:
        if var in diffnamedict:
            raise ValueError('ERROR: ' + var + ' is already contained in diffnamedict')
        diffnamedict[var] = var

    for var in diffnamedict:
        for i in range(len(equations)):
            equations[i] = subbasic(var, diffnamedict[var], equations[i])
            equations[i] = subdot(var, diffnamedict[var], equations[i])

            # substitute ss references to variable:
            if ssending2 != '':
                equations[i] = subbasic(var + ssending2, diffnamedict[var] + ssending2, equations[i])

        if varssdict is not None:
            if varssdict[var + ssending1] <= 0:
                raise ValueError('Converting log variable with steady state < 0: Variable: ' + str(var) + '. Value: ' + str(var))
            varssdict[diffnamedict[var] + ssending1] = np.log(varssdict[var + ssending1])

    if varssdict is not None:
        return(equations, varssdict)
    else:
        return(equations)


def completereplacedict_continuous(replacedict, variables, ssending1 = ssending1_default, ssending2 = ssending2_default, dotending = dotending_default, loglineareqs = False, missingparams = False):
    """
    Add additional terms for replacedict
    i.e. in replacedict I might specify replacedict['C'] = 1 where C is a variable
    But I also need to introduce C_ss, C_p to have all necessary terms available
    """
    for var in variables:
        if var in replacedict:
            ssterm = replacedict[var + ssending1]
            del replacedict[var + ssending1]
            replacedict[var + ssending2] = ssterm

            if loglineareqs is True:
                replacedict[var] = 0
            else:
                replacedict[var] = ssterm
            replacedict[var + dotending] = 0
        else:
            if missingparams is False:
                raise ValueError('variable ' + var + ' not given in replacedict.')
    return(replacedict)


def getfullreplacedict_continuous(dictlist, variables, used, ssending1 = ssending1_default, ssending2 = ssending2_default, dotending = dotending_default, loglineareqs = False, missingparams = False):
    """
    dictlist is a list of dictionaries each with a parameter and value (which may be symbolic)
    If the value for k is included and k is in variables, also add same value for k_p.

    ssending1 is the ending of dictlist for variables.

    DICTIONARIES TO CREATE:
    replacedict - all possible params and variables I could replace
    replaceuseddict - all possible params and variables I could replace that are in the equations

    paramonlyssdict - only parameters without actual controls + states
    paramonlyusedssdict - only parameters that are used in the equations without actual controls + states

    varonlyssdict - only states + controls steady states(without future)
    vardonlyssdict - only states + controls steady states(with future)
    """

    # # when allowed input of multiple dicts in dictlist:
    if isinstance(dictlist, list):
        replacedict = {}
        for thedict in dictlist:
            for var in thedict:
                replacedict[var] = thedict[var]
    else:
        replacedict = copy.deepcopy(dictlist)

    # replace replacedict variables with float
    # need to do this otherwise can get error when input into lambdify function
    # values are sympy values beforehand since in convertlogvariables, I converted the ss into a sympy log ss
    badvars = []
    badvarss = False
    for var in list(replacedict):
        try:
            replacedict[var] = float(replacedict[var])
        except Exception:
            if var in used:
                print('Bad variable steady state: Variable: ' + var + '. Steady state: ' + str(replacedict[var]) + '.')
                badvarss = True
            del replacedict[var]

    if badvarss is True:
        raise ValueError('Missing needed steady states.')


    importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'completereplacedict_continuous')(replacedict, variables, ssending1 = ssending1, ssending2 = ssending2, dotending = dotending, loglineareqs = loglineareqs, missingparams = missingparams)



    # compare used variables with variables in replacedict
    # want to check whether all variables used
    used_defined = [var for var in used if var in replacedict]
    if missingparams is False:
        if set(used) != set(used_defined):
            print('Missing variables:')
            print(set(used) - set(used_defined))
            raise ValueError('Variables in equations are not defined in replacedict.')

    # get retdict
    # from now on just adjusting arguments for retdict
    retdict = {}
    retdict['replacedict'] = replacedict

    # need to save list of parameters used in equations so I know which ones to delete if I have only partial equations
    retdict['usedparamsvarslist'] = used
    retdict['useddefinedparamsvarslist'] = used_defined
    retdict['missingparamslist'] = set(used) - set(used_defined)

    # now add other dictionaries
    retdict['replaceuseddict'] = { your_key: retdict['replacedict'][your_key] for your_key in used_defined }

    variables = set(variables)
    variablesd = variables.union(set([var + dotending for var in variables]))


    retdict['varonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in variables if your_key in replacedict }
    retdict['vardonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in variablesd if your_key in replacedict }

    retdict['paramonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in variablesd }
    retdict['paramonlyusedssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in variablesd and your_key in used_defined }

    return(retdict)


# DSGE Setup Inputdict Functions:{{{1
def completereplacedict_inputdict(inputdict, replacedict, missingparams = False):
    """
    Complete a replacedict using arguments in inputdict
    This is useful if I have already run getmodel_inputdict with missingparams = True
    Then I might want to get the completed dictionary without missingparams in which case I can complete that dictionary using this function
    """
    importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'completereplacedict_continuous')(replacedict, inputdict['states'] + inputdict['controls'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], dotending = inputdict['dotending'], loglineareqs = inputdict['loglineareqs'], missingparams = missingparams)


def getfullreplacedict_inputdict(inputdict):
    if 'use_paramssdict_only' in inputdict and inputdict['use_paramssdict_only'] is True:
        dicts = inputdict['paramssdict']
    else:
        dicts = [inputdict['paramssdict'], inputdict['varssdict']]
    retdict = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getfullreplacedict_continuous')(dicts, inputdict['states'] + inputdict['controls'], inputdict['termsinequations'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], loglineareqs = inputdict['loglineareqs'], missingparams = inputdict['missingparams'])
    for ssdictname in retdict:
        inputdict[ssdictname] = retdict[ssdictname]


def getmodel_continuous_inputdict(inputdict):

    # Basic Parse:{{{

    if 'printrundetails' not in inputdict:
        inputdict['printrundetails'] = False
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting basic parse')
        
    if 'equations' not in inputdict:
        print('No equations specified.')
        sys.exit(1)


    # how specify steady state in varssdict:
    if 'ssending1' not in inputdict:
        inputdict['ssending1'] = ssending1_default
    # how specify steady state in equations:
    if 'ssending2' not in inputdict:
        inputdict['ssending2'] = ssending2_default
    if 'dotending' not in inputdict:
        inputdict['dotending'] = dotending_default

    # steady states
    # paramssdict is steady state for parameters
    # varssdict is steady state for variables
    # however if only specify paramssdict then take varssdict to be paramssdict as well
    if 'paramssdict' not in inputdict:
        inputdict['paramssdict'] = {}

    if 'varssdict' not in inputdict:
        # note that I link them directly so that I have no issue when I change variables to logs (or something like this) i.e. the value in paramssdict also updates as well.
        inputdict['varssdict'] = inputdict['paramssdict']
        inputdict['use_paramssdict_only'] = True
    

    if 'loglineareqs' not in inputdict:
        inputdict['loglineareqs'] = False
    if 'loglineareqs' in inputdict and inputdict['loglineareqs'] is True and 'logvars' in inputdict and (inputdict['logvars'] is True or len(inputdict['logvars']) > 0):
        raise ValueError('You should not specify both loglineareqs and logvars.')
    if 'logvars' not in inputdict:
        inputdict['logvars'] = []

    if 'savefolder' not in inputdict:
        inputdict['savefolder'] = None
    if isinstance(inputdict['savefolder'], pathlib.Path):
        inputdict['savefolder'] = str(inputdict['savefolder'])
    if 'replacesavefolder' not in inputdict:
        if 'savefolder' in inputdict:
            inputdict['replacesavefolder'] = True
        else:
            inputdict['replacesavefolder'] = False

    if inputdict['savefolder'] is not None:
        if inputdict['deletesavefolder'] is True:
            if os.path.isdir(inputdict['savefolder']):
                shutil.rmtree(inputdict['savefolder'])
            os.mkdir(inputdict['savefolder'])

    if 'states' not in inputdict:
        inputdict['states'] = []
        print('Warning: No states specified')

    if 'controls' not in inputdict:
        inputdict['controls'] = []
        print('Warning: No controls specified')

    if 'mainvars' not in inputdict:
        inputdict['mainvars'] = inputdict['states'] + inputdict['controls']

    if 'mainvarnames' not in inputdict:
        inputdict['mainvarnames'] = inputdict['mainvars']

    if 'missingparams' not in inputdict:
        inputdict['missingparams'] = False

    if 'alreadyparsed' not in inputdict:
        inputdict['alreadyparsed'] = True
    else:
        if inputdict['alreadyparsed'] is True:
            print('WARNING: Already parsed inputdict.')

    # Basic Parse:}}}

    # checks on variables{{{
    # verify that variables are all defined and used
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting check for badly defined variables')

    # check equations don't contain stuff that is unspecified
    if inputdict['missingparams'] is False:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'checkeqs_string')(inputdict['equations'], inputdict['states'] + inputdict['controls'], list(inputdict['paramssdict']), varendings = [inputdict['dotending'], inputdict['ssending1'], inputdict['ssending2']])

    allvars = inputdict['states'] + inputdict['controls']

    duplicates = set([x for x in allvars if allvars.count(x) > 1])
    if len(duplicates) > 0:
        raise ValueError('Duplicates in states and controls: ' + str(duplicates) + '.')

    # verify that any variables in mainvars or irfshocks are included in states/controls/shocks
    if 'mainvars' in inputdict and inputdict['mainvars'] is not None:
        missingvars = set(inputdict['mainvars']) - set(allvars)
        if len(missingvars) > 0:
            raise ValueError('Variables in mainvars are not defined in allvars: ' + str(missingvars))
    if 'irfshocks' in inputdict and inputdict['irfshocks'] is not None:
        missingvars = set(inputdict['irfshocks']) - set(allvars)
        if len(missingvars) > 0:
            raise ValueError('Variables in irfshocks are not defined in allvars: ' + str(missingvars))

    # also verify that all variables are used
    variablesused = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'variablesinstring')(inputdict['equations'])
    variablesnotused = []
    for var in allvars:
        if var not in variablesused and var + inputdict['dotending'] not in variablesused:
            variablesnotused.append(var)
    if len(variablesnotused) > 0:
        raise ValueError('Variables not used in equations: ' + str(variablesnotused))

    inputdict['controls_dot'] = []
    inputdict['controls_nondot'] = []
    for var in inputdict['controls']:
        if var + inputdict['dotending'] in variablesused:
            inputdict['controls_dot'].append(var)
        else:
            inputdict['controls_nondot'].append(var)
    if inputdict['controls'] != inputdict['controls_dot'] + inputdict['controls_nondot']:
        print('Controls_dot: ' + str(inputdict['controls_dot']))
        print('Controls_nondot: ' + str(inputdict['controls_nondot']))
        print('WARNING: Changed order of controls so that ordered as _dot then _nondot.')
        inputdict['controls'] = inputdict['controls_dot'] + inputdict['controls_nondot']

    # checks on variables}}}

    # replace equals sign
    # need to do after add optional variables
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting replace equals')
    inputdict['equations'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'replaceequals_string')(inputdict['equations'])

    # convert log variables{{{
    # adjusting logvars
    # need to do here rather than later in case I include log variables in optional variables
    if inputdict['logvars'] is True:
        inputdict['logvars'] = inputdict['states'] + inputdict['controls']
    # convert log variables in main equations
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting convert log variables')

    inputdict['equations'] = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'convertlogvariables_string_cont')(inputdict['equations'], samenamelist = inputdict['logvars'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], dotending = inputdict['dotending'])

    # adjust steady state
    for var in inputdict['logvars']:
        if inputdict['varssdict'][var + inputdict['ssending1']] > 0:
            inputdict['varssdict'][var + inputdict['ssending1']] = np.log(inputdict['varssdict'][var + inputdict['ssending1']])
        else:
            print('Value of ' + var + ': ' + str(inputdict['varssdict'][var + inputdict['ssending1']]) + '.')
            raise ValueError('Cannot take logs of variable since its non-log steady state value is non-positive: ' + var + '.')
    # convert log variables}}}

    # get posdicts{{{
    inputdict['stateposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'])
    inputdict['controlposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['controls'])
    inputdict['statecontrolposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'] + inputdict['controls'])

    inputdict['statescontrols_dot'] = inputdict['states'] + inputdict['controls'] + [statecontrol + inputdict['dotending'] for statecontrol in inputdict['states'] + inputdict['controls']]
    # get posdicts}}}

    # replacedict:{{{
    
    # get list of terms used in equations
    inputdict['termsinequations'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'variablesinstring')(inputdict['equations'])

    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting add replacedicts')
    getfullreplacedict_inputdict(inputdict)

    # add equations_noparams (equations string with parameters replaced)
    # only leave controls and states
    inputdict['equations_noparams'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'replacevarsinstringlist')(inputdict['equations'], inputdict['paramonlyusedssdict'])

    # replacedict:}}}

    # check steady state{{{
    if 'skipsscheck' not in inputdict:
        if inputdict['missingparams'] is True:
            inputdict['skipsscheck'] = True
        else:
            inputdict['skipsscheck'] = False
    if inputdict['skipsscheck'] is False:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting check steady state')
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'checkss_string')(inputdict['equations_noparams'], inputdict['vardonlyssdict'], tolerance = 1e-9, originaleqs = inputdict['equations'])
    else:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Skipped check steady state')
    # check steady state}}}

    return(inputdict)


# Rewrite as Linear System Functions:{{{1
def dsgeanalysisdiff_cont(Et_eqs, states, controls_dot, controls_nondot, dotending = dotending_default):
    """
    If a variable appears in this equation that is not in this list, return an error.
    Useful to check for spelling mistakes etc.
    """
    import sympy
    import sys

    dotvars = states + controls_dot
    nondotvars = controls_nondot

    f1 = Et_eqs.jacobian([sympy.Symbol(var + '_dot') for var in dotvars])
    f2 = Et_eqs.jacobian([sympy.Symbol(var) for var in dotvars])
    f3 = Et_eqs.jacobian([sympy.Symbol(var) for var in nondotvars])
    
    return(f1, f2, f3)


def removenondot(nf1, nf2, nf3):
    """
    Returns f4 * Zdot = f5 * Z and Y = f6 * Zdot + f7 * Z
    """

    # avoid issues with removing dot
    nf1[np.abs(nf1) < 1e-10] = 0
    nf2[np.abs(nf2) < 1e-10] = 0
    nf3[np.abs(nf3) < 1e-10] = 0

    # rewrite matrices as [nf1, nf2] [Xdot; X] = [-nf3] [Y2]
    # in this form, I can solve for Y as a function of [Xdot; X] and an equation for matrix * [Xdot; X] = 0
    Zmatrix, Ymatrix = importattr(__projectdir__ / Path('submodules/python-math-func/matrix-reduce_func.py'), 'reducematrix')(np.column_stack((nf1, nf2)), -nf3)

    # number of dot variables
    ndot = nf1.shape[1]

    nf4 = Zmatrix[:, 0: ndot]
    nf5 = -Zmatrix[:, ndot: ]
    nf6 = Ymatrix[:, 0: ndot]
    nf7 = Ymatrix[:, ndot: ]

    # adjust nf4 - nf7 to avoid terms that should be zero being nonzero (which seems to have the potential to screw up the Blanchard Kahn conditions)
    nf4[np.abs(nf4) < 1e-10] = 0
    nf5[np.abs(nf5) < 1e-10] = 0
    nf6[np.abs(nf6) < 1e-10] = 0
    nf7[np.abs(nf7) < 1e-10] = 0

    return(nf4, nf5, nf6, nf7)



# Rewrite as Linear System Inputdict Functions:{{{1
def getfx_continuous_inputdict(inputdict):
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
        equations = 'equations_noparams'
    else:
        equations = 'equations'

    inputdict['Et_eqs'] = importattr(__projectdir__ / Path('dsgediff_func.py'), 'convertstringlisttosympy')(inputdict[equations])

    # find analytical derivatives
    f1, f2, f3 = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'dsgeanalysisdiff_cont')(inputdict['Et_eqs'], inputdict['states'], inputdict['controls_dot'], inputdict['controls_nondot'])
    inputdict['f1'] = f1
    inputdict['f2'] = f2
    inputdict['f3'] = f3

    return(inputdict)

        
def getnfx_continuous_inputdict(inputdict):

    inputdict = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getfx_continuous_inputdict')(inputdict)

    if inputdict['fxefy_cancelparams'] is True:
        replacedict = 'vardonlyssdict'
    else:
        replacedict = 'replaceuseddict'


    # apply replacedict to get numerical derivatives - if necessary
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute numerical derivatives')
    matriceslist = [inputdict['f1'], inputdict['f2'], inputdict['f3']]
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

    inputdict['nf1'], inputdict['nf2'], inputdict['nf3'] = returnlist

    return(inputdict)


def getnfxadj_continuous_inputdict(inputdict):

    inputdict = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getnfx_continuous_inputdict')(inputdict)

    nf4, nf5, nf6, nf7 = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'removenondot')(inputdict['nf1'], inputdict['nf2'], inputdict['nf3'])

    inputdict['nf4'] = nf4
    inputdict['nf5'] = nf5
    inputdict['nf6'] = nf6
    inputdict['nf7'] = nf7

    # print(inputdict['nf1'])
    # print('')
    # print(inputdict['nf2'])
    # print('')
    # print(inputdict['nf3'])
    # print('')
    #
    # print('\n\n\n')
    # print('nf4')
    # print(nf4)
    # print('')
    # print('nf5')
    # print(nf5)
    # print('')
    # print('nf6')
    # print(nf6)
    # print('')
    # print('nf7')
    # print(nf7)
    # print('')

    return(inputdict)


# Policy Function Functions:{{{1
def gxhx_lt0(A, B, numstates, raiseerror = True):
    """
    This is basically the same as the gxhx case in discrete time except the dividing line between states and controls is whether the eigenvalue < 0 or not.
    """
    import numpy as np
    from scipy.linalg import ordqz
    # from np.lin_alg import matrix_rank

    shapeA = np.shape(A)
    if shapeA[0] != shapeA[1]:
        raise ValueError('Shape of A is incorrect.')

    Nvar = shapeA[0]

    # Schmitt-Grohe Uribe codes do 2 step Schur stuff:
    # 1. find number of eigenvalues by non-ordered eigenvalues to get number of states
    # 2. reorder eigenvalues

    # get number of states
    try:
        # reorder states in desired order
        ret_temp = ordqz(A, B)
    except:
        print('ordqz failed.')
        print('fx:')
        print(fx)
        print('fxp:')
        print(fxp)
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
        print(beta[i])
        print(alpha[i])
        if (beta[i] < 0 and alpha[i] >= 0) or (beta[i] > 0 and alpha[i] < 0):
            nk = nk + 1

    # reorder
    try:
        # need to put <0 first
        ret = ordqz(A, B, sort = 'lhp')
    except:
        print('ordqz failed.')
        print('fx:')
        print(fx)
        print('fxp:')
        print(fxp)
        print('fy:')
        print(fy)
        print('fyp:')
        print(fyp)

        # need to put <0 first
        ret = ordqz(A, B, sort = 'lhp')
    s = ret[0]
    t = ret[1]
    q = ret[4]
    z = ret[5]

    if nk < numstates:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(numstates)
        print('No local equilibrium exists.')
        if raiseerror is True:
            sys.exit(1)
            # raise ValueError('')
    if nk > numstates:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(numstates)
        print('The equilibrium is locally indeterminate.')
        if raiseerror is True:
            sys.exit(1)
            # raise ValueError('')

    # Here I go ahead and use dimensions of states I specified rather than dimensions suggested by eigenvalues.

    z21 = z[(numstates):, 0: numstates]
    z11 = z[0: numstates, 0: numstates]


    # return error if can't invert
    if np.linalg.matrix_rank(z11) < numstates:
        print(z11)
        raise ValueError('Invertibility condition violated.')


    s11 = s[0: numstates, 0: numstates]
    t11 = t[0: numstates, 0: numstates]

    z11i = np.linalg.inv(z11)

    gx = np.dot(z21, z11i)
    hx = np.dot(np.dot(np.dot(z11, np.linalg.inv(s11)), t11), z11i)

    return(gx, hx)


def gxhx_lt0(A, B, numstates, raiseerror = True):
    """
    This is basically the same as the gxhx case in discrete time except the dividing line between states and controls is whether the eigenvalue < 0 or not.
    """
    import numpy as np
    from scipy.linalg import ordqz

    shapeA = np.shape(A)
    if shapeA[0] != shapeA[1]:
        raise ValueError('Shape of A is incorrect.')

    Nvar = shapeA[0]

    # a special sort function is needed
    # this is because otherwise scipy can set that when alpha = -1e-17 and beta = 1, the eigenvalue should count as a state
    # (matlab did not count this as a state)
    def sortfunction(alpha, beta):
        retlist = []
        for i in range(len(alpha)):
            # exclude cases where alpha is effectively zero but not actually zero
            if (beta[i] < 0 and alpha[i] >= 0) or (beta[i] > 0 and alpha[i] < -1e-10):
                retlist.append(True)
            else:
                retlist.append(False)
        return(retlist)


    # reorder
    try:
        # need to put <0 first
        ret = ordqz(A, B, sort = sortfunction)
    except:
        print('ordqz failed.')
        print('fx:')
        print(fx)
        print('fxp:')
        print(fxp)
        print('fy:')
        print(fy)
        print('fyp:')
        print(fyp)

        ret = ordqz(A, B, sort = sortfunction)
        raise ValueError('ordqz failed')

    s = ret[0]
    t = ret[1]
    alpha = ret[2]
    beta = ret[3]
    q = ret[4]
    z = ret[5]

    nk = 0
    for i in range(Nvar):
        if (beta[i] < 0 and alpha[i] >= 0) or (beta[i] > 0 and alpha[i] < -1e-10):
            nk = nk + 1

    if nk < numstates:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(numstates)
        print('No local equilibrium exists.')
        if raiseerror is True:
            sys.exit(1)
            # raise ValueError('')
    if nk > numstates:
        print('States solved for:')
        print(nk)
        print('States specified at beginning:')
        print(numstates)
        print('The equilibrium is locally indeterminate.')
        if raiseerror is True:
            sys.exit(1)
            # raise ValueError('')

    # Here I go ahead and use dimensions of states I specified rather than dimensions suggested by eigenvalues.

    z21 = z[(numstates):, 0: numstates]
    z11 = z[0: numstates, 0: numstates]


    # return error if can't invert
    if np.linalg.matrix_rank(z11) < numstates:
        print(z11)
        raise ValueError('Invertibility condition violated.')


    s11 = s[0: numstates, 0: numstates]
    t11 = t[0: numstates, 0: numstates]

    z11i = np.linalg.inv(z11)

    gx = np.dot(z21, z11i)
    hx = np.dot(np.dot(np.dot(z11, np.linalg.inv(s11)), t11), z11i)

    return(gx, hx)


def getfullgx(tildegx, hx, f6, f7):
    """
    Back out gx for controls without a dot process in the equations from the substituted matrices.
    """
    tildegxshape = np.shape(tildegx)
    nx = tildegxshape[1]
    ny1 = tildegxshape[0]

    tildegx2 = f6.dot( np.row_stack((hx, np.zeros((ny1, nx)))) ) + f7.dot( np.row_stack((np.eye(nx), tildegx)) )

    gx = np.row_stack((tildegx, tildegx2))

    return(gx)



# Policy Function Inputdict Functions:{{{1
def polfunc_continuous_inputdict(inputdict):
    """
    Get policy function through a dictionary.
    Necessary arguments: equations_plus, replacedict, states, controls, 
    Optional: lambdifyok, evalequals, gxhxraise, continuous
    """

    # get policy functions
    # raise an error if I have the wrong number of states?
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting compute policy functions')
    # get policy functions for dot variables
    inputdict['tildegx'], inputdict['hx'] = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'gxhx_lt0')(inputdict['nf4'], inputdict['nf5'], numstates = len(inputdict['states']))
    # extend to general policy functions (including nondot variables)
    inputdict['gx'] = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getfullgx')(inputdict['tildegx'], inputdict['hx'], inputdict['nf6'], inputdict['nf7'])

    return(inputdict)


def interpretpolfunc_inputdict(inputdict):

    # Parse arguments:{{{
    if 'printmain' not in inputdict:
        inputdict['printmain'] = True

    # use separate shocks dict here since I might want to include states or exclude certain shocks
    if 'irfshocks' not in inputdict:
        inputdict['irfshocks'] = []
    if 'irfshockdict' not in inputdict:
        inputdict['irfshockdict'] = {}
    for shock in inputdict['irfshocks']:
        if shock not in inputdict['irfshockdict']:
            inputdict['irfshockdict'][shock] = 1
    if 'showirfs' not in inputdict:
        inputdict['showirfs'] = inputdict['irfshocks']

    if 'irfperiod_length' not in inputdict:
        inputdict['irfperiod_length'] = 1


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

    # IRFS:{{{
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting IRFs')

    # get hx2 - the discrete time version of the continuous time hx
    inputdict['hx2'] = np.eye(len(inputdict['states'])) + inputdict['irfperiod_length'] * inputdict['hx']

    # get IRFs
    inputdict['irf_XY'] = {}
    for shock in inputdict['irfshocks']:
        X0 = np.zeros(len(inputdict['states']))
        # set up shock in irf to be 1 unless otherwise specified
        X0[inputdict['statecontrolposdict'][shock]] = inputdict['irfshockdict'][shock]
        XY = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'irmatrix')(inputdict['gx'], inputdict['hx2'], X0, T = inputdict['irf_T'])
        # save overall irf
        inputdict['irf_XY'][shock] = XY

        irfvars = [inputdict['statecontrolposdict'][varname] for varname in inputdict['mainvars']]
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


def checksame_inputdict_cont(inputdict1, inputdict2):
    """
    This function allows me to compare the results from a linearized model to a non-lineariezed model to verify that I have not made a mistake in the linearized model.
    I can also use this to compare a log model with a log-linearized model.

    Equations need to be in same order and use the same variables
    """
    # ensure checking steady state - since I sometimes turn this off
    inputdict1['skipsscheck'] = False
    inputdict2['skipsscheck'] = False

    # get the basic model
    inputdict1 = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getmodel_continuous_inputdict')(inputdict1)
    inputdict2 = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getmodel_continuous_inputdict')(inputdict2)

    # add fx fxp fy fyp
    # nonlinearised
    inputdict1 = getnfx_continuous_inputdict(inputdict1)
    # linearised
    inputdict2 = getnfx_continuous_inputdict(inputdict2)

    # verify both models have the same dot controls
    if inputdict1['controls_dot'] != inputdict2['controls_dot']:
        print('Model 1 dot controls: ' + str(inputdict1['controls_dot']) + '.')
        print('Model 2 dot controls: ' + str(inputdict2['controls_dot']) + '.')
        raise ValueError('Models have different dot controls.')

    # combine relevant vectors
    dotvars = inputdict1['states'] + inputdict1['controls_dot']
    nondotvars = inputdict1['controls_nondot']
    fullvars = [var + inputdict1['dotending'] for var in dotvars] + dotvars + nondotvars

    fullmat1 = np.concatenate((inputdict1['nf1'], inputdict1['nf2'], inputdict1['nf3']), axis = 1)
    fullmat2 = np.concatenate((inputdict2['nf1'], inputdict2['nf2'], inputdict2['nf3']), axis = 1)

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
            print(inputdict1['equations'][row])
            print(inputdict2['equations'][row])


            diffcols = [col for col in range(numvars) if np.abs(fullmat1[row][col] - fullmat2[row][col]) > 1e-8]
            print([fullvars[col] for col in diffcols])
            print([fullmat1[row][col] - fullmat2[row][col] for col in diffcols])


    if sameall is True:
        # print('Two models are the same.')
        None
    else:
        raise ValueError('ERROR: TWO MODELS ARE DIFFERENT.')

    


# All Inputdict Function:{{{1
def continuouslineardsgefull(inputdict):
    
    # get basic model
    inputdict = importattr(__projectdir__ / Path('dsge_continuous_func.py'), 'getmodel_continuous_inputdict')(inputdict)

    # rewrite as linear model
    getnfxadj_continuous_inputdict(inputdict)
    
    # add policy functions
    polfunc_continuous_inputdict(inputdict)

    # interpret policy functions
    interpretpolfunc_inputdict(inputdict)
    
