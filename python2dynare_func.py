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
import os
import re
import scipy.io as sio
import subprocess

# Defaults:{{{1
ssending1_default = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'ssending1_default') # the ending in varssdict
futureending_default = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'futureending_default')


# Create Dynare Script:{{{1
def python2dynare(equations_string, paramssdict, dynareformatvariables = None, states = None, controls = None, shocks = None, shocksddict = None, shockpath = None, varssdict = None, loadfilename = None, savefilename = None, futureending = futureending_default, ssending1 = ssending1_default, simulation = None):
    """
    Code to make a basic Dynare script from stuff in Python.

    dynareformatvariables = states + controls but not shocks
    predeterminedvariables = states when equations is in SGU form and = None when equations is in Dynare form i.e. with k(-1), k(+1)
    Need to input either dynareformatvariables or both states and controls. If dynareformatvariables is used then the equations_string should be in the dynare time form otherwise it should be in the SGU form.

    If filename_delete is not False, just replace the relevant blocks of the file, not the full file.
    """
    # make copies to prevent issues
    equations_string = copy.deepcopy(equations_string)

    if shocks is None:
        shocks = []

    # not in Dynare variable form
    if states is not None or controls is not None:
        if states is None:
            states = []
        if controls is None:
            controls = []

        for var in states + controls:
            for i in range(len(equations_string)):
                equations_string[i] = re.sub('(?<![a-zA-Z0-9_])' + var + futureending + '(?![a-zA-Z0-9_])', var + '(+1)', equations_string[i])


    # define variables
    if dynareformatvariables is not None:
        variables = dynareformatvariables
    else:
        variables = states + controls

    # replace power term which works in Python but not Dynare
    for i in range(len(equations_string)):
        equations_string[i] = equations_string[i].replace('**', '^')

    # regexes for replacing text in file
    re_var = re.compile('\nvar (.*);\n')
    re_varexo = re.compile('\n(|// )varexo (.*);\n')
    re_predetermined_variables = re.compile('\n(|// )predetermined_variables (.*);\n')
    re_parameters = re.compile('\n(|// )parameters (.*);\n')
    re_parameters_define = re.compile('\n// parameters_define;\n([\s\S]*?)\n// parameters_define_end;\n')
    re_model = re.compile('\nmodel;\n([\s\S]*?)\nend;\n')
    re_initval = re.compile('\ninitval;\n([\s\S]*?)\nend;\n')
    re_shocks = re.compile('\n(|// )shocks;\n([\s\S]*?)\n(|// )end;\n')
    re_simulation = re.compile('\n// simulation_command;\n([\s\S]*?)\n// simulation_command_end;\n')

    # load filename:{{{
    if loadfilename is None:
        loadfilename = __projectdir__ / Path('dynare/template.mod')
    # get text to replace
    with open(loadfilename) as f:
        text = f.read()

    # verify that this file has what we need for replace
    if re_var.search(text) is None:
        raise ValueError('var block missing')

    if states is not None and len(states) > 0 and re_predetermined_variables.search(text) is None:
        raise ValueError('predetermined_variables block missing')

    if len(shocks) > 0 and re_varexo.search(text) is None:
        raise ValueError('varexo block missing')

    if len(paramssdict) is not None and re_parameters.search(text) is None:
        raise ValueError('parameters block missing')

    if len(paramssdict) is not None and re_parameters_define.search(text) is None:
        raise ValueError('parameters_define block missing')

    if re_model.search(text) is None:
        raise ValueError('model block missing')

    if re_initval.search(text) is None:
        raise ValueError('initval block missing')

    if re_simulation.search(text) is None:
        raise ValueError('simulation block missing')
    # load filename:}}}

    # replace var
    text = re.sub(re_var, '\nvar ' + ' '.join(variables) + ';\n', text, count = 1)

    # replace varexo
    if len(shocks) > 0:
        text = re.sub(re_varexo, '\nvarexo ' + ' '.join(shocks) + ';\n', text, count = 1)
    else:
        # if I don't add something here I seem to get an error with checking the covariance matrices so I may want to add a random shock if I haven't specified one
        text = re.sub(re_varexo, '\n// varexo ;\n', text, count = 1)
        # text = re.sub(re_varexo, '\nvarexo randomexovar;\n', text, count = 1)

    # replace predetermined_variables
    if states is not None and len(states) > 0:
        text = re.sub(re_predetermined_variables, '\npredetermined_variables ' + ' '.join(states) + ';\n', text, count = 1)
    else:
        text = re.sub(re_predetermined_variables, '\n// predetermined_variables ;\n', text, count = 1)

    # replace parameters
    if len(paramssdict) > 0:
        text = re.sub(re_parameters, '\nparameters ' + ' '.join(sorted(list(paramssdict))) + ';\n', text, count = 1)
    else:
        text = re.sub(re_parameters, '\n// parameters ;\n', text, count = 1)

    # replace parameters_define
    if len(paramssdict) is not None:
        parameters_define_string = '\n'.join([param + '=' + str(paramssdict[param]) + ';' for param in sorted(paramssdict)])
        text = re.sub(re_parameters_define, '\n// parameters_define;\n' + parameters_define_string + '\n// parameters_define_end;\n', text, count = 1)
    else:
        text = re.sub(re_parameters_define, '\n// parameters_define;\n\n// parameters_define_end;\n', text, count = 1)

    # replace model
    model_string = '\n'.join([equation + ';' for equation in equations_string])
    text = re.sub(re_model, '\nmodel;\n' + model_string + '\nend;\n', text, count = 1)

    # replace initval
    initval_string = '\n'.join([var + '=' + str(varssdict[var + ssending1]) + ';' for var in varssdict])
    text = re.sub(re_initval, '\ninitval;\n' + initval_string + '\nend;\n', text, count = 1)

    # replace shocks
    if len(shocks) > 0:
        if shockpath is not None:
            # case where I specify shocks
            if np.shape(shockpath)[1] != len(shocks):
                raise ValueError('shockpaths should be in format T x numvars. np.shape(shockpath)[0] = ' + str(np.shape(shockpath)[1]) + ' and len(shocks) = ' + str(len(shocks)) + ' so something has failed.')

            shocks_string = ''
            for i in range(len(shocks)):
                shock = shocks[i]
                # note that I define periods to be 1 - 30 rather than 0 - 29 since Dynare doesn't do shocks at time 0
                shocks_string = shocks_string + 'var ' + shock + ';\nperiods ' + ' '.join([str(var) for var in list(range(1, np.shape(shockpath)[0] + 1))]) + ';\nvalues ' + ' '.join([str(var) for var in shockpath[:, i]]) + ';\n'
        elif shocksddict is not None:
            # case where I specify s.d. of shocks but not actual shocks
            shocks_string = '\n'.join(['var ' + shock + '; stderr ' + str(shocksddict[shock]) + ';' for shock in shocks]) + '\n'
        else:
            raise ValueError('Shocks has nonzero length but I have not specified how I want to deal with shocks in the simulation. Dynare will yield an error.')

        # replace shocks section
        text = re.sub(re_shocks, '\nshocks;\n' + shocks_string + 'end;\n', text, count = 1)
    else:
        # replace shocks section with no shocks
        text = re.sub(re_shocks, '\n// shocks;\n// var e; stderr 1;\n// end;\n', text, count = 1)

    # replace simulation
    if simulation is None:
        # simulation = 'stoch_simul(order=1);'
        simulation = '// ;'
    text = re.sub(re_simulation, '\n// simulation_command;\n' + simulation + '\n// simulation_command_end;\n', text, count = 1)


    if savefilename is not None:
        with open(savefilename, 'w+') as f:
            f.write(text)

    return(text)



def python2dynare_inputdict(inputdict):
    """
    Run python2dynare with inputdict argument.
    Can specify filename separately to save a little time.
    """
    # print time
    importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting python2dynare')

    if 'python2dynare_loadfilename' not in inputdict:
        inputdict['python2dynare_loadfilename'] = None

    if 'python2dynare_savefilename' not in inputdict:
        if 'savefolder' in inputdict and inputdict['savefolder'] is not None:
            inputdict['python2dynare_savefilename'] = os.path.join(inputdict['savefolder'], 'main.mod')
        else:
            if 'python2dynare_nofilename' not in inputdict or inputdict['python2dynare_nofilename'] is not True:
                raise ValueError('Need to specify savefolder or python2dynare_savefilename')

    # simulation variable
    if 'python2dynare_simulation' not in inputdict:
        inputdict['python2dynare_simulation'] = None

    # determine which equations and parameters I use
    if 'python2dynare_noparams' in inputdict and inputdict['python2dynare_noparams'] is True:
        # replace all parameters in equations
        equations = inputdict['equations_noparams']
        paramssdict = {}
    elif 'python2dynare_usedparams' in inputdict and inputdict['python2dynare_usedparams'] is True:
        # use only parameters that are in equation
        equations = inputdict['equations']
        paramssdict = inputdict['paramonlyusedssdict']
    else:
        # use all parameters
        equations = inputdict['equations']
        paramssdict = inputdict['paramonlyssdict']

    if 'shocksddict' in inputdict:
        shocksddict = inputdict['shocksddict']
    else:
        shocksddict = None
    if 'shockpath' in inputdict:
        shockpath = inputdict['shockpath']
    else:
        shockpath = None

    dynaretext = importattr(__projectdir__ / Path('python2dynare_func.py'), 'python2dynare')(equations, paramssdict = paramssdict, varssdict = inputdict['varonlyssdict'], states = inputdict['states'], controls = inputdict['controls'], shocks = inputdict['shocks'], shocksddict = shocksddict, shockpath = shockpath, loadfilename = inputdict['python2dynare_loadfilename'], savefilename = inputdict['python2dynare_savefilename'], simulation = inputdict['python2dynare_simulation'])

    return(dynaretext)

# Run Dynare:{{{1
def gendynarematlabscript(savefolder, dynarefilename = 'main.mod', matlabfilename = 'rundynare.m', dynarepath = None):
    matlabfiletext = ''

    if dynarepath is not None:
        matlabfiletext = matlabfiletext + "addpath('" + dynarepath + "');\n"

    matlabfiletext = matlabfiletext + 'dynare ' + dynarefilename + ';'

    with open(os.path.join(savefolder, 'rundynare.m'), 'w+') as f:
        f.write(matlabfiletext)
    

def gendynarematlabscript_inputdict(inputdict):
    if 'python2dynare_savefilename' not in inputdict:
        inputdict['python2dynare_savefilename']

    if 'dynarepath' not in inputdict:
        if os.path.isfile(__projectdir__ / Path('paths/dynarepath.txt')):
            with open(__projectdir__ / Path('paths/dynarepath.txt')) as f:
                dynarepath = f.read()
            if dynarepath[-1] == '\n':
                dynarepath = dynarepath[: -1]
            inputdict['dynarepath'] = dynarepath
        else:
            inputdict['dynarepath'] = None

    importattr(__projectdir__ / Path('python2dynare_func.py'), 'gendynarematlabscript')(os.path.dirname(inputdict['python2dynare_savefilename']), os.path.basename(inputdict['python2dynare_savefilename']), dynarepath = inputdict['dynarepath'])


def rundynare(dynarefolder, dynarerunfile = 'rundynare.m', runwithoctave = False):
    cwd = os.getcwd()
    os.chdir(dynarefolder)

    if runwithoctave is True:
        matlaboroctave = 'octave'
    else:
        matlaboroctave = 'matlab'

    subprocess.call([matlaboroctave, dynarerunfile])

    os.chdir(cwd)


def rundynare_inputdict(inputdict):
    importattr(__projectdir__ / Path('python2dynare_func.py'), 'gendynarematlabscript_inputdict')(inputdict)

    if 'runwithoctave' in inputdict:
        runwithoctave = inputdict['runwithoctave']
    else:
        if os.path.isfile(__projectdir__ / Path('paths/rundynarewithoctave.txt')):
            runwithoctave = True
        else:
            runwithoctave = False

    importattr(__projectdir__ / Path('python2dynare_func.py'), 'rundynare')(os.path.dirname(inputdict['python2dynare_savefilename']), runwithoctave = runwithoctave)


# IRFS:{{{1
def getirfs_dynare(dynarefilename):
    """
    I can use this function to call IRFs after running a perfect foresight solver in Dynare i.e. simul
    """
    matlabsave = dynarefilename[: -4] + '_results.mat'

    # get names of variables
    data = sio.loadmat(matlabsave)
    names = data['M_']['endo_names'][0][0]
    names = [name.strip() for name in names]

    importattr(__projectdir__ / Path('submodules/python-math-func/matlab_func.py'), 'irf_matlab')(matlabsave, names, matlabsavefunc = lambda x: x['oo_']['endo_simul'][0][0])


def getirfs_dynare_inputdict(inputdict):
    importattr(__projectdir__ / Path('python2dynare_func.py'), 'getirfs_dynare')(inputdict['python2dynare_savefilename'])
