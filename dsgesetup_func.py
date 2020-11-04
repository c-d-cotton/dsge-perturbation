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
import datetime
import numpy as np
import os
import pathlib
import pickle
import re
import shutil

# Defaults:{{{1
ssending1_default = '' # the ending in varssdict
ssending2_default = '_ss' # the ending in terms that appear in equations
futureending_default = '_p'
# these are words that if I find in eqsstrings should be doing some math operation and should not be a variable
mathoperations = set(['exp', 'log', 'max', 'min'])
keyelements_inputdict = ['states', 'controls', 'shocks', 'equations', 'equations_noparams', 'equations_plus', 'equations_noparams_plus', 'paramssdict', 'varssdict', 'replacedict', 'replaceuseddict', 'paramonlyssdict', 'stateposdict', 'stateshockposdict', 'statecontrolposdict', 'stateshockcontrolposdict', 'termsinequations', 'logvars', 'ssending1', 'ssending1', 'futureending', 'irfshocks', 'optionalstatedict', 'optionalcontroldict', 'optionalshockslist', 'statesplusoptional', 'controlsplusoptional', 'shocksplusoptional', 'variablesplusoptional', 'mainvars', 'loglineareqs']

# String Variable Functions:{{{1
def variablesinstring(eqsstring):
    """
    Get list of variables in a string list of equations.
    """
    solveeqsvars = set()
    # notice that this requires the variable starts with a-zA-Z
    revar = re.compile('(?<![a-zA-Z0-9_])([a-zA-Z][a-zA-Z0-9_]*)(?![a-zA-Z0-9_])')
    for eq in eqsstring:
        while True:
            match = revar.search(eq)
            if match is not None:
                # add match to list of variables
                solveeqsvars.add(match.group(1))
                # only replace first match
                eq = eq.replace(match.group(1), '', 1)
            else:
                break

    # remove mathoperations from solveeqsvars
    solveeqsvars = solveeqsvars.difference(mathoperations)

    return(solveeqsvars)


def replacevarsinstringlist(eqsstring, replacedict):
    # make copy
    eqsstring = copy.deepcopy(eqsstring)
    for var in replacedict:
        for i in range(len(eqsstring)):
            # I could add ( to the lookahead to prevent matching exp and log etc.
            eqsstring[i] = re.sub('(?<![a-zA-Z0-9_])' + var + '(?![a-zA-Z0-9_])', '' + str(replacedict[var]), eqsstring[i])
    return(eqsstring)        


# Get posdicts:{{{1
def getposdict(variables):
    """
    General function to get posdict.
    Posdict is dictionary that gives position of variable in list.
    """
    posdict = {}
    for i in range(len(variables)):
        posdict[variables[i]] = i
    return(posdict)


# Add Equations:{{{1
def addoptionalcontrols(eqs_string, controls, optionalcontroldict, futureoptionalcontroldict = None, futureending = futureending_default, allvars = None, optionalstatedict = None):
    """
    This function allows me to reduce the size of the number of equations I am considering.
    optionalcontroldict contains the equation for variables that do not strictly need to be included in eqs_string and could be substituted out
    If a control is not in controls but is in optionalcontroldict then substitute it out

    Substitution:
    Substitute out x using optionalcontroldict['x']
    Substitute out x_p using futureoptionalcontroldict['x'] if it is defined. If it is not then subsstitute it out by replacing every current variable y with y_p in optionalcontroldict['x'] - if y_p is already defined in optionalcontroldict['x'], this yields an error.
    I make this substitution in eqs_string but also in optionalcontroldict (since this may contain 'x'). I also do this with optionalstatedict if it is specified.

    Also, note that I need to specify allvars (a list of all variables to replace with their future version) if I want to get the future substitution without using futureoptionalcontroldict.
    """
    if futureoptionalcontroldict is None:
        futureoptionalcontroldict = {}

    allvars = copy.deepcopy(allvars)

    # verify that all controls in varinclude are in optionalcontroldict
    optionalcontrolsinclude = []
    for control in controls:
        if control in optionalcontroldict:
            optionalcontrolsinclude.append(control)

    # add in vars that I want to include
    for var in optionalcontrolsinclude:
        eqs_string.append(var + ' = ' + optionalcontroldict[var])

    # function to add in futurevardict
    # only use this if I have a var x to replace and x_p is defined in equations
    # only works if no future variables defined in optionalcontroldict otherwise raise error
    def addfutureoptionalcontroldict(removedcontrol):
        # start with present value of removedcontrol
        futureeq = optionalcontroldict[removedcontrol]
        if allvars is None:
            raise ValueError('Found future version of variable in optionalcontroldict. Need to specify allvars option so that can replace future variable with equivalent future variable definition. Or use futureoptionalcontroldict.')

        # need to remove removedcontrol from allvars so don't replace it as well
        allvars.remove(removedcontrol)

        # replace control with control_p in futureeqs
        for var in allvars:
            match = re.compile('(?<![a-zA-Z0-9_])' + var + '_p' + '(?![a-zA-Z0-9\(_])').search(futureeq)
            if match is not None:
                raise ValueError('Cannot create futurevariable definition for variable: ' + removedcontrol + ' because the definition of control includes ' + str(var + '_p'))
            
            futureeq = re.sub('(?<![a-zA-Z0-9_])' + var + '(?![a-zA-Z0-9\(_])', '' + var + '_p', futureeq)
        futureoptionalcontroldict[removedcontrol] = futureeq

    def replacevar(eq_string, var):
        eq_string = re.sub('(?<![a-zA-Z0-9_])' + var + '(?![a-zA-Z0-9\(_])', '(' + optionalcontroldict[var] + ')', eq_string)

        futurevarstring = '(?<![a-zA-Z0-9_])' + var + futureending + '(?![a-zA-Z0-9\(_])'

        # add var to futureoptionalcontroldict if necessary
        if var not in futureoptionalcontroldict:
            match = re.compile(futurevarstring).search(eq_string)
            if match is not None:
                addfutureoptionalcontroldict(var)            
        if var in futureoptionalcontroldict:
            eq_string = re.sub(futurevarstring, '(' + futureoptionalcontroldict[var] + ')', eq_string)

        return(eq_string)
    
    # substitute vars that I don't want to include
    notincluded = [var for var in optionalcontroldict if var not in optionalcontrolsinclude]
    for excludedcontrol in notincluded:

        for i in range(len(eqs_string)):
            # replace present variables
            eqs_string[i] = replacevar(eqs_string[i], excludedcontrol)

        for optionalcontrol in optionalcontroldict:
            optionalcontroldict[optionalcontrol] = replacevar(optionalcontroldict[optionalcontrol], excludedcontrol)
        for optionalstate in optionalstatedict:
            optionalstatedict[optionalstate] = replacevar(optionalstatedict[optionalstate], excludedcontrol)


    return(eqs_string)


def addoptionalcontrols_test():
    def run(includey = False):
        eqs_string = ['x + y + y_p = 1']
        controls = ['x', 'v']
        states = ['z']
        optionalcontroldict = {'y': '2*x + z', 'v': 'y_p'}
        optionalstatedict = {'J': 'y_p'}
        if includey is True:
            controls.append('y')
        
        eqs_string = addoptionalcontrols(eqs_string, controls, optionalcontroldict, futureoptionalcontroldict = None, allvars = controls + states + list(optionalcontroldict), optionalstatedict = optionalstatedict)
        print(eqs_string)
        print(controls)
        print(optionalstatedict)

    print('Include y')
    run(includey = True)

    print('\nNot include y')
    run(includey = False)


def addoptionalstates(eqs_string, states, optionalstatedict, futureending = futureending_default):
    """
    Doesn't do replace (which wouldn't make sense anyway for states).
    Just leaves out states that I don't need.
    Meant for exogenous variables like shocks i.e. u_p = 0 or exogenous processes like productivity i.e. A_p = RHO * A.

    """
    optionalstatesinclude = []
    for state in states:
        if state in optionalstatedict:
            optionalstatesinclude.append(state)

    # include variables that are included
    for state in optionalstatesinclude:
        eqs_string.append(state + futureending + ' = ' + optionalstatedict[state])
        # states.append(state)

    return(eqs_string)


def addoptionalstates_test():
    optionalstatedict = {'A': 'RHO * A + epsilon_A'}
    states = ['A']
    eqs_string = []
    allvars = ['A', 'epsilon_A']

    # should work
    eqs_string = addoptionalstates(eqs_string, states, optionalstatedict)
    print('A added:')
    print(eqs_string)


    optionalstatedict = {'A': 'RHO * A + epsilon_A'}
    states = []
    eqs_string = []
    allvars = ['A', 'epsilon_A']

    # should return an error since shouldn't exclude when epsilon_A is a variable
    eqs_string = addoptionalstates(eqs_string, states, optionalstatedict)
    print('\nA not added:')
    print(eqs_string)


def replaceequals_string(equations):
    for i in range(len(equations)):
        equation = equations[i]
        numberofequals = equation.count('=')
        if numberofequals == 0:
            continue
        elif numberofequals == 1:
            # so change from 1 = 1 to 1 -( 1)
            equations[i] = equation.replace('=', '-(') + ')'
        else:
            raise ValueError('Too many equations in equation row number: ' + str(i) + '. Equation: ' + str(equation))

    return(equations)


# Convert to logs:{{{1
def convertlogvariables_string(equations, samenamelist = [], diffnamedict = None, varssdict = None, ssending1 = ssending1_default, ssending2 = ssending2_default, futureending = futureending_default, dynareformat = False):
    """
    It also allows dynare inputs. The varssdict returned in the Dynare case would work with Dynare but not with my gxhx functions which require a s.s. defined for each period.

    diffnamedict['C'] = 'c' means replace C with exp(c) in equations.
    If c in samenamelist then replace c with exp(c).

    If specify varssdict then also convert varssdict. Consequently return variable number of arguments.
    Will want to convert varssdict as well if I computed varssdict before doing log conversion.
    ssending1 allows me to specify what the steady state variables in varssdict end in.
    """
    import numpy as np

    def subbasic(oldname, newname, string):
        string = re.sub('(?<![a-zA-Z0-9_])' + oldname + '(?![a-zA-Z0-9\(_])', 'exp(' + newname + ')', string)
        return(string)

    def subparantheses_dynare(oldname, newname, string):
        string = re.sub('(?<![a-zA-Z0-9_])' + oldname + '(\([0-9\+\-]*\))' '(?![a-zA-Z0-9_])', 'exp(' + newname + '\g<1>' + ')', string)
        return(string)


    if diffnamedict is None:
        diffnamedict = {}
    for var in samenamelist:
        if var in diffnamedict:
            raise ValueError('ERROR: ' + var + ' is already contained in diffnamedict')
        diffnamedict[var] = var

    for var in diffnamedict:
        for i in range(len(equations)):
            # discrete replace future
            if dynareformat is True:
                equations[i] = subparantheses_dynare(var, diffnamedict[var], equations[i])
            else:
                equations[i] = subbasic(var + futureending, diffnamedict[var] + futureending, equations[i])

            equations[i] = subbasic(var, diffnamedict[var], equations[i])

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


# Basic check to ensure all variables specified:{{{1
def checkeqs_string(eqsstring, listvars, params, varendings = [futureending_default, ssending1_default, ssending2_default]):
    """
    Check that the variables in the equations make sense.
    Add ending

    (If I run this for the steady state equations after substitution, I only need to include listvars since then I'll have removed futurevars and specified params. That's why they're optional)
    """

    # get list of ok string variables
    okstringvars = set()
    okstringvars = okstringvars.union(listvars)
    for ending in varendings:
        okstringvars = okstringvars.union([var + ending for var in listvars])

    if params is not None:
        okstringvars = okstringvars.union(params)

    okstringvars = okstringvars.union(mathoperations)

    # verify that each line does not contain bad variables
    # do line one at a time to make fixing errors easier
    stop = False
    for line in eqsstring:
        varsinstring = importattr(__projectdir__ / Path('submodules/python-math-func/stringvar_func.py'), 'varsinstring')(line)
        badvars = varsinstring.difference(okstringvars)
        if len(badvars) > 0:
            print('\nBad variables: ' + str(badvars))
            print(line)
            stop = True
    if stop is True:
        raise ValueError('Non-defined terms used in string.')




# Check steady state specified correctly:{{{1
def checkss_string(eqs, replacedict, tolerance = 1e-9, originaleqs = None):
    """
    Apply replacedict to eqs and ensure that it equals zero.

    originaleqs is the original equations which are not adjusted which I can print as well as eqs to help me find the error in the steady state.
    """
    eqsold = eqs

    # go through each line and convert to steady state - return error if fail
    badconvert = False
    sseqs = []
    for i in range(len(eqsold)):
        # replace strings with numbers
        try:
            eqreplaced = importattr(__projectdir__ / Path('submodules/python-math-func/stringvar_func.py'), 'replacevardict')(eqsold[i], replacedict)
        except Exception:
            print('Failed equation:')
            print(eqsold[i])
            badconvert = True
            continue
        # evaluate string of numbers
        try:
            eqnum = importattr(__projectdir__ / Path('submodules/python-math-func/stringvar_func.py'), 'evalstring')(eqreplaced)
        except Exception:
            print('Failed equation:')
            print(eqsold[i])
            print(eqreplaced)
            badconvert = True
            continue
        # add result to list
        sseqs.append(eqnum)
    if badconvert is True:
        raise ValueError('Conversion for steady state failed.')

    # eqs = copy.deepcopy(eqs)
    # eqs = replacevarsinstringlist(eqs, replacedict)
    # eqs = evaleqs(eqs)


    badeqs = []
    for i in range(len(sseqs)):
        try:
            if abs(sseqs[i]) > 1e-9:
                badeqs.append(i)
        except Exception:
            badeqs.append(i)
    
    if len(badeqs) > 0:
        print('Equations:')
        print(sseqs)
        print('')
        print('Replacedict:')
        print(replacedict)
        print('')
        print('Bad Equations:')
        for badi in badeqs:
            print(eqsold[badi])
            print(sseqs[badi])
        if originaleqs is not None:
            print('')
            print('Original Bad Equations:')
            for badi in badeqs:
                print(originaleqs[badi])
                print(sseqs[badi])
        raise ValueError('Steady State Incorrect.')


# Replace Dict:{{{1
def completereplacedict(replacedict, variables, ssending1 = ssending1_default, ssending2 = ssending2_default, futureending = futureending_default, loglineareqs = False, missingparams = False, shocksplusoptional = None):
    """
    Add additional terms for replacedict
    i.e. in replacedict I might specify replacedict['C'] = 1 where C is a variable
    But I also need to introduce C_ss, C_p to have all necessary terms available
    """
    # add direct shocks to varssdict:
    if shocksplusoptional is not None:
        for var in shocksplusoptional:
            # replace if not exist (since most shocks are 0 so saves time) of if using log linear equations (when all shocks should be zero)
            if var + ssending1 not in replacedict:
                replacedict[var + ssending1] = 0

    for var in variables:
        if var in replacedict:
            ssterm = replacedict[var + ssending1]
            del replacedict[var + ssending1]
            replacedict[var + ssending2] = ssterm
            if loglineareqs is True:
                replacedict[var] = 0
                replacedict[var + futureending] = 0
            else:
                replacedict[var] = ssterm
                replacedict[var + futureending] = ssterm
        else:
            if loglineareqs is True:
                replacedict[var] = 0
                replacedict[var + futureending] = 0

        # REMOVED THIS 20200210 since it throws an error unless I specify every variable's steady state but this isn't what I want if I only consider a loglinear model
        # else:
        #     if missingparams is False:
        #         raise ValueError('variable ' + var + ' not given in replacedict.')
    return(replacedict)


def getfullreplacedict_discrete(dictlist, variables, used, statescontrols, shocks, ssending1 = ssending1_default, ssending2 = ssending2_default, futureending = futureending_default, loglineareqs = False, missingparams = False, shocksplusoptional = None):
    """
    dictlist is a list of dictionaries each with a parameter and value (which may be symbolic)
    If the value for k is included and k is in variables, also add same value for k_p.

    ssending1 is the ending of dictlist for variables.

    DICTIONARIES TO CREATE:
    replacedict - all possible params and variables I could replace
    replaceuseddict - all possible params and variables I could replace that are in the equations

    paramonlyssdict - only parameters without actual controls + states + shocks
    paramplusonlyssdict - only parameters without actual controls + states
    paramonlyusedssdict - only parameters that are used in the equations without actual controls + states + shocks
    paramplusonlyusedssdict - only parameters that are used in the equations without actual controls + states

    varonlyssdict - only states + controls steady states(without future)
    varplusonlyssdict - only states + controls + shocks s.s. (witout future)
    varfonlyssdict - only states + controls steady states(with future)
    varfplusonlyssdict - only states + controls + shocks s.s. (with future)


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


    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'completereplacedict')(replacedict, variables, ssending1 = ssending1, ssending2 = ssending2, futureending = futureending, loglineareqs = loglineareqs, shocksplusoptional = shocksplusoptional, missingparams = missingparams)



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

    statescontrols = set(statescontrols)
    shocks = set(shocks)

    # define statescontrols including future variables
    statescontrolsf = statescontrols.union(set([var + futureending for var in statescontrols]))
    # define shocks including future variables
    shocksf = shocks.union(set([var + futureending for var in shocks]))


    retdict['varonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in statescontrols if your_key in replacedict }
    retdict['varfonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in statescontrolsf if your_key in replacedict }

    retdict['varplusonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in statescontrols.union(shocks) if your_key in replacedict }
    retdict['varfplusonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in statescontrolsf.union(shocksf) if your_key in replacedict }

    # this includes shocks in parameters
    retdict['paramplusonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in statescontrolsf }
    retdict['paramplusonlyusedssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in statescontrolsf and your_key in used_defined }
    
    # this excludes shocks from parameters
    retdict['paramonlyssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in statescontrolsf.union(shocksf) }
    retdict['paramonlyusedssdict'] = { your_key: retdict['replacedict'][your_key] for your_key in retdict['replacedict'] if your_key not in statescontrolsf.union(shocksf) and your_key in used_defined }

    return(retdict)


# Inputdict Specific Functions:{{{1
def printruntime(inputdict, message = None):
    if 'printrundetails' in inputdict and inputdict['printrundetails'] is True:
        if 'startruntime' not in inputdict:
            inputdict['startruntime'] = datetime.datetime.now()
        if message is not None:
            print('Stage: ' + message + '. Time: ' + str(datetime.datetime.now() - inputdict['startruntime']))

def completereplacedict_inputdict(inputdict, replacedict, missingparams = False):
    """
    Complete a replacedict using arguments in inputdict
    This is useful if I have already run getmodel_inputdict with missingparams = True
    Then I might want to get the completed dictionary without missingparams in which case I can complete that dictionary using this function
    """
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'completereplacedict')(replacedict, inputdict['variablesplusoptional'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], futureending = inputdict['futureending'], loglineareqs = inputdict['loglineareqs'], missingparams = missingparams, shocksplusoptional = inputdict['shocksplusoptional'])


def getfullreplacedict_inputdict(inputdict):
    if 'use_paramssdict_only' in inputdict and inputdict['use_paramssdict_only'] is True:
        dicts = inputdict['paramssdict']
    else:
        dicts = [inputdict['paramssdict'], inputdict['varssdict']]
    retdict = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getfullreplacedict_discrete')(dicts, inputdict['variablesplusoptional'], inputdict['termsinequations'], inputdict['states'] + inputdict['controls'], inputdict['shocks'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], loglineareqs = inputdict['loglineareqs'], missingparams = inputdict['missingparams'], shocksplusoptional = inputdict['shocksplusoptional'])
    for ssdictname in retdict:
        inputdict[ssdictname] = retdict[ssdictname]


# Get Model Full:{{{1
def getmodel_inputdict(inputdict):
    """
    Parse inputdict to adjust a set of equations to get them ready to use as a model (so replace optional variables, adjust log variables).
    Also check that arguments make sense i.e. check that parameters/variables in equations make sense, check steady state.
    """

    # basic parse:{{{
    if 'printrundetails' not in inputdict:
        inputdict['printrundetails'] = False
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting basic parse')
        
    if 'equations' not in inputdict not in inputdict:
        print('No equations specified.')
        sys.exit(1)

    if 'shocks' not in inputdict:
        inputdict['shocks'] = []

    # ending for future variables in equations
    if 'futureending' not in inputdict:
        inputdict['futureending'] = futureending_default

    # how specify steady state in varssdict:
    if 'ssending1' not in inputdict:
        inputdict['ssending1'] = ssending1_default
    # how specify steady state in equations:
    if 'ssending2' not in inputdict:
        inputdict['ssending2'] = ssending2_default

    # steady states
    # paramssdict is steady state for parameters
    # varssdict is steady state for variables
    # however if only specify paramssdict then take varssdict to be paramssdict as well
    if 'paramssdict' not in inputdict:
        inputdict['paramssdict'] = {}

    if 'varssdict' not in inputdict:
        if 'varssdict' not in inputdict and 'paramssdict' in inputdict:
            # note that I link them directly so that I have no issue when I change variables to logs (or something like this) i.e. the value in paramssdict also updates as well.
            inputdict['varssdict'] = inputdict['paramssdict']
            inputdict['use_paramssdict_only'] = True
        else:
            raise ValueError('Neither varssdict nor paramssdict specified.')

    # treat equations as already log linearised
    if 'loglineareqs' not in inputdict:
        inputdict['loglineareqs'] = False
    # check I haven't defined both logvars and loglineareqs
    if 'loglineareqs' in inputdict and inputdict['loglineareqs'] is True and 'logvars' in inputdict and (inputdict['logvars'] is True or len(inputdict['logvars']) > 0):
        raise ValueError('You should not specify both loglineareqs and logvars.')

    # need to adjust log variables at start since I call these in python2dynare
    # log variable conversion:
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
        if inputdict['replacesavefolder'] is True:
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

    # basic parse:}}}

    # optional states, controls and shocks:{{{
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting add optional variables')
    # add optionalcontroldict/optionalstatedict
    # add None options
    if 'optionalcontroldict' not in inputdict:
        inputdict['optionalcontroldict'] = None
    if 'optionalstatedict' not in inputdict:
        inputdict['optionalstatedict'] = None
    if 'futureoptionalcontroldict' not in inputdict:
        inputdict['futureoptionalcontroldict'] = None
    if 'optionalshockslist' not in inputdict:
        inputdict['optionalshockslist'] = None

    # get full list of variables
    # this allows me to get the steady state dictionary
    # so if I have an optional variable that I didn't include, it will still be included in the steady state dictionary
    if inputdict['optionalstatedict'] is not None:
        inputdict['statesplusoptional'] = sorted(list(set(inputdict['states']).union(set(inputdict['optionalstatedict']))))
    else:
        inputdict['statesplusoptional'] = copy.deepcopy(inputdict['states'])
    if inputdict['optionalcontroldict'] is not None:
        inputdict['controlsplusoptional'] = sorted(list(set(inputdict['controls']).union(set(inputdict['optionalcontroldict']))))
    else:
        inputdict['controlsplusoptional'] = copy.deepcopy(inputdict['controls'])
    if inputdict['optionalshockslist'] is not None:
        inputdict['shocksplusoptional'] = sorted(list(set(inputdict['shocks']).union(set(inputdict['optionalshockslist']))))
    else:
        inputdict['shocksplusoptional'] = copy.deepcopy(inputdict['shocks'])
    inputdict['variablesplusoptional'] = inputdict['statesplusoptional'] + inputdict['controlsplusoptional'] + inputdict['shocksplusoptional']

    # # now add in optionalstatedict/optionalcontroldict
    if inputdict['optionalcontroldict'] is not None:
        addoptionalcontrols(inputdict['equations'], inputdict['controls'], inputdict['optionalcontroldict'], futureoptionalcontroldict = inputdict['futureoptionalcontroldict'], futureending = inputdict['futureending'], allvars = inputdict['variablesplusoptional'], optionalstatedict = inputdict['optionalstatedict'])

    if inputdict['optionalstatedict'] is not None:
        addoptionalstates(inputdict['equations'], inputdict['states'], inputdict['optionalstatedict'], futureending = inputdict['futureending'])

    # optional states, controls and shocks:}}}

    # checks on variables{{{
    # verify that variables are all defined and used
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting check for badly defined variables')

    # check equations don't contain stuff that is unspecified
    if inputdict['missingparams'] is False:
        checkeqs_string(inputdict['equations'], inputdict['variablesplusoptional'], list(inputdict['paramssdict']), varendings = [inputdict['futureending'], inputdict['ssending1'], inputdict['ssending2']])

    allvars = inputdict['states'] + inputdict['controls'] + inputdict['shocks']

    duplicates = set([x for x in allvars if allvars.count(x) > 1])
    if len(duplicates) > 0:
        raise ValueError('Duplicates in states, controls and shocks: ' + str(duplicates) + '.')

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
    variablesused = variablesinstring(inputdict['equations'])
    variablesnotused = []
    for var in allvars:
        if var not in variablesused and var + inputdict['futureending'] not in variablesused:
            variablesnotused.append(var)
    if len(variablesnotused) > 0:
        raise ValueError('Variables not used in equations: ' + str(variablesnotused))

    # checks on variables}}}
        
    # adjust shocks but don't add to equations yet{{{
    # don't add them since I want them separate in some cases

    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting adjust shocks and states')

    # generate equations for shocks
    # need to do after set steady state for shocks
    inputdict['shocks_equations'] = []
    for shock in inputdict['shocks']:
        # adding shock_p - shock_ss
        inputdict['shocks_equations'].append(shock + inputdict['futureending'] + ' - ' + '0')

    # add direct shocks to states
    inputdict['states_plus'] = inputdict['states'] + inputdict['shocks']
    # adjust shocks but don't add to equations yet}}}

    # get posdicts{{{
    inputdict['stateposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'])
    inputdict['controlposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['controls'])
    inputdict['shockposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['shocks'])
    inputdict['statecontrolposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'] + inputdict['controls'])
    inputdict['stateshockposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'] + inputdict['shocks'])
    inputdict['stateshockcontrolposdict'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'getposdict')(inputdict['states'] + inputdict['shocks'] + inputdict['controls'])

    inputdict['statescontrols_p'] = inputdict['states'] + inputdict['controls'] + [statecontrol + inputdict['futureending'] for statecontrol in inputdict['states'] + inputdict['controls']]
    inputdict['statescontrolsshocks_p'] = inputdict['states'] + inputdict['controls'] + inputdict['shocks'] + [statecontrolshock + inputdict['futureending'] for statecontrolshock in inputdict['states'] + inputdict['controls'] + inputdict['shocks']]

    # get posdicts}}}

    # replace equals sign
    # need to do after add optional variables
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting replace equals')
    inputdict['equations'] = replaceequals_string(inputdict['equations'])

    # convert log variables{{{
    if inputdict['missingparams'] is True and inputdict['logvars'] != []:
        print("WARNING: note this won't work well with missingparams - as things stand, this will return an error if a varss is missing which is likely if parameters are missing")
        print("that being said, it probably doesn't make sense to use logvars when parameters are missing since I'm likely to be doing Bayesian estimation in which case I should probably use the log-linearized form")

    # adjusting logvars
    # need to do here rather than later in case I include log variables in optional variables
    if inputdict['logvars'] is True:
        inputdict['logvars'] = inputdict['states'] + inputdict['controls']
    # convert log variables in main equations
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting convert log variables')
    inputdict['equations'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'convertlogvariables_string')(inputdict['equations'], samenamelist = inputdict['logvars'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], futureending = inputdict['futureending'])
    # convert log variables in shock equations
    inputdict['shocks_equations'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'convertlogvariables_string')(inputdict['shocks_equations'], samenamelist = inputdict['logvars'], ssending1 = inputdict['ssending1'], ssending2 = inputdict['ssending2'], futureending = inputdict['futureending'])
    # adjust steady state
    for var in inputdict['logvars']:
        if inputdict['varssdict'][var + inputdict['ssending1']] > 0:
            inputdict['varssdict'][var + inputdict['ssending1']] = np.log(inputdict['varssdict'][var + inputdict['ssending1']])
        else:
            print('Value of ' + var + ': ' + str(inputdict['varssdict'][var + inputdict['ssending1']]) + '.')
            raise ValueError('Cannot take logs of variable since its non-log steady state value is non-positive: ' + var + '.')
    # convert log variables}}}

    # get replacedicts i.e. dict that will replace all symbols with{{{

    # get list of terms used in equations
    inputdict['termsinequations'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'variablesinstring')(inputdict['equations'] + inputdict['shocks_equations'])

    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting add replacedicts')
    getfullreplacedict_inputdict(inputdict)

    # add equations_noparams (equations string with parameters replaced)
    # only leave controls, states and shocks
    inputdict['equations_noparams'] = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'replacevarsinstringlist')(inputdict['equations'], inputdict['paramonlyusedssdict'])

    # get replacedicts i.e. dict that will replace all symbols with}}}

    # check steady state{{{
    if 'skipsscheck' not in inputdict:
        if inputdict['missingparams'] is True:
            inputdict['skipsscheck'] = True
        else:
            inputdict['skipsscheck'] = False
    if inputdict['skipsscheck'] is False:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting check steady state')
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'checkss_string')(inputdict['equations_noparams'], inputdict['varfplusonlyssdict'], tolerance = 1e-9, originaleqs = inputdict['equations'])
    else:
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Skipped check steady state')
    # check steady state}}}

    # Add shocks into equations
    # get full equations for case where include shocks directly
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting add shocks to state variables')
    inputdict['equations_plus'] = inputdict['equations'] + inputdict['shocks_equations']
    inputdict['equations_noparams_plus'] = inputdict['equations_noparams'] + inputdict['shocks_equations']

    # Save filename
    if 'pickleinputdict_filename' in inputdict and inputdict['pickleinputdict_filename'] is not None:
        if 'pickleinputdict_filename' is True:
            if 'savefolder' in inputdict:
                inputdict['pickleinputdict_filename'] = os.path.join(inputdict['savefolder'], 'inputdict.pickle')
            else:
                raise ValueError("Need to specify savefolder if inputdict['pickleinputdict'] is True")
        with open(inputdict['pickleinputdict_filename'], 'wb') as f:
            pickle.dump(inputdict, f)

    return(inputdict)


def savemodel_inputdict(inputdict, picklesavename):
    """
    This saves only the key elements from inputdict.
    """
    inputdict = {key: inputdict[key] for key in keyelements_inputdict if key in inputdict}
    with open(picklesavename, 'wb') as f:
        pickle.dump(inputdict, f)
