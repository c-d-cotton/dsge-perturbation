#!/usr/bin/env python3
"""
If running multiple codes at the same time, run "export MKL_NUM_THREADS=1" on the terminal before
Prevents multithreading in numpy
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import numpy as np
import scipy

# Simulation given regimes:{{{1
def getpolicygivenregimes(regimeABCDElist, regimesovertime, H_Tplus, I_regime0):
    """
    regimesABCDElist = [[A,B,C,D,E]_{for normal regime to which revert in long-run without shocks}, [A,B,C,D,E]_{other regime}]
    regimesovertime = [0,0,1,0,0] where this would mean use the first regime for all but the second period when use the second regime
    H_Tplus is the policy function i.e. X_{t} = HX_{t - 1} when t \geq T
    H_Tplus is also the policy function that would occur if we were only in the first regime

    Can speed up the results a lot by noting if the system is in the first regime for all t >= T then the policy function at T is given by H_Tplus

    """

    T = len(regimesovertime)
    # get number of variables by looking at dimensions of A in nonbinding regime
    numvar = np.shape(regimeABCDElist[0][0])[0]

    # use this variable to state whether all future iterations are regime0
    futureregime0only = True

    # Garray is just the array for the constant
    # make any given constant a numvar x 1 matrix to match with other variables
    Garray = np.empty([T, numvar])
    # Harray - array of all policy function values over time
    # Harray[0] is H_0 where X_{t} = H_0X_{-1} so X_{T - 1} = Harray[-1]X_{T - 2} (since we know Harray[-1] = H_{T - 1})
    Harray = np.empty([T, numvar, numvar])
    # start working backwards from t = T - 1
    for t in reversed(range(T)):
        if t == T - 1:
            Gp = [0] * numvar
            Hp = H_Tplus
        else:
            Gp = Garray[t + 1]
            Hp = Harray[t + 1]

        regime = regimesovertime[t]
        if regime != 0:
            futureregime0only = False

        # need to get coeff to compute Et
        if futureregime0only is True:
            # know that policy function will be normal
            # doing this saves time
            Garray[t] = [0] * numvar
            Harray[t] = H_Tplus

            if t == 0:
                I0 = I_regime0
        else:
            At = regimeABCDElist[regime][0]
            Bt = regimeABCDElist[regime][1]
            Ct = regimeABCDElist[regime][2]
            Dt = regimeABCDElist[regime][3]

            # I tried solve instead of inv but it's slower because we're multiplying the inverse by a square matrix not a vector
            coeff = - np.linalg.inv(Bt + Ct.dot(Hp))
            Garray[t] = coeff.dot(Ct.dot(Gp) + Dt)
            Harray[t] = coeff.dot(At)



            if t == 0:
                Et = regimeABCDElist[regime][4]
                I0 = coeff.dot(Et)

                # this line i.e. coeff.dot(Et) can take a surprisingly long time unless E is a float array

    return(Garray, Harray, I0)


def simulatepath(Garray, Harray, I0, Zm1, epsilon0, numstates):
    """
    Zarray = [states_t; controls_{t - 1}]
    XYarray = [states_t; controls_t]
    """
    T = np.shape(Harray)[0]
    numvar = np.shape(Zm1)[0]

    Zarray = np.empty([T, numvar])
    for t in range(T):
        if t == 0:
            # add list since otherwise get error that 4x1 cannot be broadcast to 4
            Zarray[t] = Garray[t] + Harray[t].dot(Zm1).reshape([numvar]) + I0.dot(epsilon0).reshape([numvar])

        else:
            # add list since otherwise get error that 4x1 cannot be broadcast to 4
            Zarray[t] = list(Garray[t] + Harray[t].dot(Zarray[t - 1]))

    # add XYarray
    XYarray = np.empty([T, numvar])
    XYarray[0, 0: numstates] = Zm1[0: numstates]
    XYarray[1: , 0: numstates] = Zarray[0: T - 1, 0: numstates]
    XYarray[:, numstates: ] = Zarray[:, numstates: ]

    return(Zarray, XYarray)
        

# Solve Switching Regimes:{{{1
def simulatepathfull(ABCDElist, regimelist, H_Tplus, I_regime0, numstates, Zm1 = None, epsilon0 = None):
    """
    Get policy function and then compute simulation from this.
    Really the only thing this function does is put everything together.
    """
    # get policy function
    Garray, Harray, I0 = getpolicygivenregimes(ABCDElist, regimelist, H_Tplus, I_regime0)

    numvar = np.shape(Garray)[1]
    numshock = np.shape(I0)[1]

    if Zm1 is None:
        Zm1 = np.zeros(numvar)

    if epsilon0 is None:
        epsilon0 = np.zeros(numshock)


    # given the policy function, get path of variables over time
    _, XYarray = simulatepath(Garray, Harray, I0, Zm1, epsilon0, numstates)

    return(XYarray)


def regimechange(inputdictlist, regimelist, irf = False, Zm1 = None, epsilon0 = None):
    """
    The variables should be the same in each inputdict.

    The system models what happens to the variables as they change to regime modelled by regimelist

    System returns to inputdictlist[0] after regimelist ends.
    """
    ABCDElist = []
    for i in range(len(inputdictlist)):
        inputdict = inputdictlist[i]
        if i > 0:
            inputdict['skipsscheck'] = True
        from dsgesetup_func import getmodel_inputdict
        inputdictlist[i] = getmodel_inputdict(inputdict)
        from dsgediff_func import ABCDE_form_full
        ABCDElist.append(ABCDE_form_full(inputdict['equations_noparams'], inputdict['states'], inputdict['controls'], inputdict['shocks'], inputdict['varfplusonlyssdict']))

    # get H_Tplus i.e. policy function after long enough period without shocks
    inputdict2 = copy.deepcopy(inputdictlist[0])
    from dsge_bkdiscrete_func import polfunc_inputdict
    inputdict2 = polfunc_inputdict(inputdict2)
    from dsge_bkdiscrete_func import gxhx_splitbyshocks
    gx_noshocks, gx_shocks, hx_noshocks, hx_shocks = gxhx_splitbyshocks(inputdict2['gx'], inputdict2['hx'], len(inputdict2['shocks']))
    nstates = len(inputdict2['states'])
    ncontrols = len(inputdict2['controls'])
    nvars = nstates + ncontrols
    H_Tplus = np.zeros([nvars, nvars])
    H_Tplus[0: nstates, 0: nstates] = hx_noshocks
    H_Tplus[nstates: nvars, 0: nstates] = gx_noshocks

    I_regime0 = np.vstack((hx_shocks, gx_shocks))

    # solve for path of variables
    XYarray = simulatepathfull(ABCDElist, regimelist, H_Tplus, I_regime0, nstates, Zm1 = Zm1, epsilon0 = epsilon0)

    # IRFs
    if irf is True:
        pos = [inputdict['statecontrolposdict'][var] for var in inputdictlist[0]['mainvars']]

        XYarray2 = XYarray[:, pos]

        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from matplotlib_func import gentimeplots_basic
        gentimeplots_basic(XYarray2.transpose(), inputdict['mainvars'])

    return(XYarray)


