#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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
    inputdict = getshocksddict_inputdict(inputdict)

    inputdict['shockpath'] = getshockpath(inputdict['pathsimperiods'], inputdict['shocks'], inputdict['shocksddict'], pathsimperiods_noshockend = inputdict['pathsimperiods_noshockend'])

    return(inputdict)
