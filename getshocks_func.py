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

# Add shocksddict:{{{1
def getshocksddict(shocks, shocksddict = None):
    if shocksddict is None:
        shocksddict = {}
    for shock in shocks:
        if shock not in shocksddict:
            shocksddict[shock] = 1
    return(shocksddict)


def getshocksddict_inputdict(inputdict):
    if 'shocksddict' not in inputdict:
        inputdict['shocksddict'] = None
    inputdict['shocksddict'] = getshocksddict(inputdict['shocks'], shocksddict = inputdict['shocksddict'])
    return(inputdict)


# Add shockpath:{{{1
def getshockpath(pathsimperiods, shocks, shocksddict, pathsimperiods_noshockend = None, addmissing = False):
    """
    Produces array of form T x numshocks
    """

    # allow for zeros after sim - useful for IRFs
    if pathsimperiods_noshockend is None:
        pathsimperiods_noshockend = 0

    if shocksddict is None:
        shocksddict = {}

    for shock in shocks:
        if shock not in shocksddict:
            shocksddict[shock] = 1

    shockpath = np.zeros([pathsimperiods + pathsimperiods_noshockend, len(shocks)])
    for i in range(len(shocks)):
        shock = shocks[i]
        shockpath[0: pathsimperiods, i] = shocksddict[shock] * np.random.normal(size = pathsimperiods)

    return(shockpath)


def getshockpath_inputdict(inputdict):
    
    # add arguments
    if 'pathsimperiods' not in inputdict:
        raise ValueError("getshockpath_inputdict requires the argument inputdict['pathsimperiods']")
    if 'pathsimperiods_noshockend' not in inputdict:
        inputdict['pathsimperiods_noshockend'] = None

    # add shocksddict
    inputdict = importattr(__projectdir__ / Path('getshocks_func.py'), 'getshocksddict_inputdict')(inputdict)

    inputdict['shockpath'] = importattr(__projectdir__ / Path('getshocks_func.py'), 'getshockpath')(inputdict['pathsimperiods'], inputdict['shocks'], inputdict['shocksddict'], pathsimperiods_noshockend = inputdict['pathsimperiods_noshockend'])

    return(inputdict)
