#!/usr/bin/env python3
"""
Simulate out DSGE model given a path of shocks.
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np

def simpathlinear(gx, hx, shocks, X0 = None):
    """
    shocks should be in form T x numshocks

    Returns matrix of form T x numvars (ordered states, shocks, controls)
    """
    T = np.shape(shocks)[0]
    ny = np.shape(gx)[0]
    numstatesshocks = np.shape(gx)[1]
    numshocks = np.shape(shocks)[1]
    numstates = numstatesshocks - numshocks
    
    if X0 is None:
        X0 = np.zeros(numstates)
    
    X = np.empty([T, numstatesshocks])
    Y = np.empty([T, ny])

    for t in range(T):
        if t == 0:
            X[t, : numstates] = X0
        else:
            X[t, : numstates] = np.dot(hx[: numstates, :], X[t - 1])

        X[t, numstates: ] = shocks[t]
            
        Y[t] = np.dot(gx, X[t])

    XY = np.concatenate((X, Y), axis = 1)

    return(XY)


def simpathlinear_inputdict(inputdict):
    """
    Run after running polfunc_inputdict
    """
    inputdict['varpath'] = simpathlinear(inputdict['gx'], inputdict['hx'], inputdict['shockpath'])

    # get list of steady states in order of states, shocks, controls
    # ssending2 is "_ss" - note that otherwise if I use loglineareqs then the variable will be 0
    ssvec = np.array([inputdict['replacedict'][var + inputdict['ssending2']] for var in inputdict['states'] + inputdict['shocks'] + inputdict['controls']])

    # get list of steady states in order of states, shocks, controls
    # ssending2 is "_ss" - note that otherwise if I use loglineareqs then the variable will be 0
    ssvec = []
    for var in inputdict['states'] + inputdict['shocks'] + inputdict['controls']:
        ss = inputdict['replacedict'][var + inputdict['ssending2']]
        if var in inputdict['logvars']:
            ss = np.exp(ss)
        ssvec.append(ss)
    ssvec = np.array(ssvec)

    inputdict['expvarpath'] = np.exp(inputdict['varpath'])
    # note that [[0, 1], [2, 3]] + [2, 4] in numpy = [[2, 4], [5, 7]]
    inputdict['ssexpvarpath'] = ssvec * inputdict['expvarpath']

    return(inputdict)

