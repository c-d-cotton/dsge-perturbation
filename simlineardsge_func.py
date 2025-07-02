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
    ssvec = np.array([inputdict['varfplusonlyssdict'][var] for var in inputdict['states'] + inputdict['shocks'] + inputdict['controls']])

    # get x rather than xhat = x - xbar
    # note that [[0, 1], [2, 3]] + [2, 4] in numpy = [[2, 4], [5, 7]]
    # thus each column of the matrix is multiplied by the element of the vector
    # this is what I need since every column represents the path of a variable and I want to multiply it by the steady state
    inputdict['ssvarpath'] = inputdict['varpath'] + ssvec

    # if I log-linearized variables then I'll want to take the exponential i.e. X = exp(x)
    # note that if I did log-linearize then the ss above will be the ss of the log variable so I need to take the exponential of that as well
    inputdict['expssvarpath'] = np.exp(inputdict['ssvarpath'])


    return(inputdict)

