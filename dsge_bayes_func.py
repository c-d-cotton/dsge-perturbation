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
import sympy

# Omega Matrix:{{{1
def Omega_fromsdvec(vector):
    """
    Convert a vector of strings into a sympy matrix
    Assume no correlation between different shocks
    i.e. ['sigma_a', 'sigma_b'] becomes [['sigma_a', 0], [0, 'sigma_b']]
    """
    matrix = sympy.zeros(len(vector), len(vector))
    for i in range(len(vector)):
        # have to square to go from sdvec to variance
        matrix[i, i] = sympy.Symbol(vector[i]) ** 2

    return(matrix)


def Omega_convertfunc(sympymatrix):
    matrixfunc, parameters = importattr(__projectdir__ / Path('submodules/python-sympy-extra/subs/subs_func.py'), 'lambdifysubs_lambdifyonly')(sympymatrix)
    return(matrixfunc, parameters)


def Omega_test():
    vector = ['sigma_a', 'sigma_b']
    sympymatrix = Omega_fromvec(vector)
    Omegafunc, parameters = Omega_convertfunc(sympymatrix)

    Omega = Omegafunc(0.1, 0.2)


# Get Log Likelihood of Data Given Parameters:{{{1
def getbayes_dsge_logl_aux(inputdict_partiallyevaluated, replacedictfunc, y, varnames, paramval, Omega_funcparamnames = None, solveP0_iterate = False):
    """
    This function derives the log-likelihood for a given set of parameters
    y is the observed data. should have dimensions T x numobervedvars. Input as np.array

    Optional argument:
    Omega_funcparams = [Omegafunc, Omegaparamnames]. I should then be able to get Omega = Omegafunc([p[paramname] for paramname in Omegaparamnames])
    Omega then allows me to specify a variance for the shocks. This is useful if I didn't directly specify the variance in the equations.
    """
    # get parameters
    p = replacedictfunc(paramval)
    
    # get fxe, fxep, fy, fyp
    retlist = importattr(__projectdir__ / Path('dsgediff_func.py'), 'partialtofulleval_quick_inputdict')(inputdict_partiallyevaluated, p)

    # get gxhx
    gx, hx = importattr(__projectdir__ / Path('dsge_bkdiscrete_func.py'), 'gxhx')(retlist[0], retlist[1], retlist[2], retlist[3])

    # break down into ABCD
    # we know that (X_t \\ \epsilon_{t + 1}) = hx (X_{t - 1} \\ \epsilon_t )
    # therefore we split hx into A, B where X_t = AX_{t - 1} + B\epsilon_t
    numstates = len(inputdict_partiallyevaluated['states'])
    numshocks = len(inputdict_partiallyevaluated['shocks'])
    A = hx[: numstates, : numstates]
    B = hx[: numstates, numstates: ]
    # we compute C, D only for the observed variables
    # note that this yields Y_t = CX_t + D\epsilon_t
    # if we observe a state then it will just equal to X_{i,t} = 1X_{i,t} + 0\epsilon_t
    observedvarsindices = [inputdict_partiallyevaluated['statecontrolposdict'][varname] for varname in varnames]
    stategx = np.concatenate([np.eye(numstates), np.zeros([numstates, numshocks])], axis = 1)
    allgx = np.concatenate([stategx, gx], axis = 0)
    C = allgx[observedvarsindices, : numstates]
    D = allgx[observedvarsindices, numstates: ]

    # add Omega if specified
    # Omega specifies a variance for the shock vectors
    # this is an alternative to specifying the variance of shocks directly in the equations
    if Omega_funcparamnames is not None:
        Omega = Omega_funcparamnames[0](*[p[paramname] for paramname in Omega_funcparamnames[1]])
    else:
        Omega = None

    # get kalman filter
    if solveP0_iterate is True:
        P0 = importattr(__projectdir__ / Path('submodules/python-math-func/statespace/statespace_func.py'), 'solvevariance_quick')(A, B, Omega = Omega, crit = 1e-6)
    else:
        P0 = None
    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = importattr(__projectdir__ / Path('submodules/python-math-func/statespace/statespace_func.py'), 'kalmanfilter')(y, A, B, C, D, Omega = Omega, P0 = P0)

    # get log likelihood
    ll = importattr(__projectdir__ / Path('submodules/python-math-func/statespace/statespace_func.py'), 'logl_prop_kalmanfilter')(y, y_t_tm1, Q_t_tm1)

    return(ll)






