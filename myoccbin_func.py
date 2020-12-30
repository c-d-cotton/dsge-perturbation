#!/usr/bin/env python3
"""
Functions for solving inequality constraint DSGE problems in an occbin-like manner.
"""
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import copy
import datetime
import numpy as np
import pickle
import shutil
import sympy

# Test if Constraint Binds:{{{1
def testconstraint(Xarray, constraintvars, constraint_noparams, allposdict):
    """
    Test whether a constraint holds or not at each period until T.
    Returns list of True/False.
    """
    T = np.shape(Xarray)[0]
    constraintholdslist = []

    for t in range(T):
        constraint_noparams_eval = constraint_noparams
        for var in constraintvars:
            sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
            from stringvar_func import replacestring
            constraint_noparams_eval = replacestring(constraint_noparams_eval, var, Xarray[t][allposdict[var]])

        # evaluating 1 < 2, 1 < 0 should yield True/False list
        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from stringvar_func import evalstring
        constraintholdslist.append(evalstring(constraint_noparams_eval))

    return(constraintholdslist)
    


    

# Regime Updating Functions:{{{1
def regimeupdate_occbin(inputregime_old, outputregime_old, triedalready, printwarnings = True):
    """
    Use the same regime updating as in occbin.
    So take the regime from the previous gu

    triedalready needs to be an empty list.
    """
    if inputregime_old == outputregime_old:
        stop = True
    else:
        if outputregime_old in triedalready:
            stop = True
            if printwarnings is True:
                print('Warning: Already tried the regime that was output in the last iteration. Will just take this output regime to be correct.')
        else:
            stop = False


    if stop is True:
        inputregime_new = inputregime_old
    else:
        # add old regime to list of regimes I've tried
        triedalready.append(outputregime_old)
        inputregime_new = outputregime_old

    return(inputregime_new, stop, triedalready)


def regimeupdate_binduntilnot(inputregime_old, outputregime_old, exitbindat):
    """
    Here the central bank sets i = 0 until they would not hit the binding constraint without additional shocks.
    exitbindat is the period when the central bank returns to the normal regime again
    exitzlbat should be initialised to zero (and the initial inputregime should be all in the nonbinding regime).

    Algorithm tries [0,0,0,0] then [1,0,0,0], [1,1,0,0] etc. until it finds a case where the constraint is slack in every period where the regime in slack.
    For the ZLB case, this implies that the central bank will hold i = 0 until it reaches a period when setting i at the normal rule will not cause the zlb to hit without further shocks.
    """
    T = len(inputregime_old)

    # get last period when constraint binds
    bindingperiods = [period for period in range(len(outputregime_old)) if outputregime_old[period] == 1]
    if len(bindingperiods) > 0:
        lastbindperiod = np.max(bindingperiods)
    else:
        lastbindperiod = -1

    if lastbindperiod < exitbindat:
        stop = True
        inputregime_new = inputregime_old
    else:
        stop = False
        exitbindat = exitbindat + 1
        inputregime_new = [1] * exitbindat + [0] * (T - exitbindat)


    return(inputregime_new, stop, exitbindat)


# Full Get Simulation Functions:{{{1
def oneperiodshocksolve(Zm1, epsilon0, ABCDE_bind, ABCDE_nobind, H_Tplus, I_regime0, constraintvars, slackconstraint_noparams, allposdict, numstates, regimeupdatefunc = 'occbin', maxiter = 100, solverperiods = 100, printdetails = False, printvars = None, starttime = None, printwarnings = True):
    """
    Solve for one period shock.
    """
    numvars = len(Zm1)

    # set up stuff for regimeupdatefunc
    if regimeupdatefunc == 'occbin':
        triedalready = []
    elif regimeupdatefunc == 'binduntilnot':
        exitbindat = 0

    for i in range(maxiter):
        if printdetails is True:
            if starttime is None:
                print('\nStarting iteration ' + str(i + 1) + ':')
            else:
                print('\nStarting iteration ' + str(i + 1) + '. Time: ' + str(datetime.datetime.now() - starttime))

        # add inputregime if i == 0
        if i == 0:
            # we just consider the case where the constraint never binds initially
            inputregime = [0] * solverperiods
        if printdetails is True:
            print('Input regime: ' + str(inputregime))

        # get the policy function given the specified regimes
        if i == 0:
            if printdetails is True:
                if starttime is not None:
                    print('Generating policy functions using H_Tplus. Time: ' + str(datetime.datetime.now() - starttime))
                
            Garray = np.zeros([solverperiods, numvars])
            Harray = np.empty([solverperiods, numvars, numvars])
            for t in range(solverperiods):
                Harray[t] = H_Tplus
            I0 = I_regime0
        else:
            if starttime is not None:
                print('Generating policy functions based upon regimes. Time: ' + str(datetime.datetime.now() - starttime))
            from regime_func import getpolicygivenregimes
            Garray, Harray, I0 = getpolicygivenregimes([ABCDE_nobind, ABCDE_bind], inputregime, H_Tplus, I0)
            if starttime is not None:
                print('Finished generating policy functions based upon regimes. Time: ' + str(datetime.datetime.now() - starttime))

        # given the policy function, get path of variables over time
        if starttime is not None:
            print('Iterating over the policy function. Time: ' + str(datetime.datetime.now() - starttime))

        from regime_func import simulatepath
        Zarray, XYarray = simulatepath(Garray, Harray, I0, Zm1, epsilon0, numstates)


        # get whether the constraint is binding or slack at each period
        if starttime is not None:
            print('Testing whether the constraint is slack. Time: ' + str(datetime.datetime.now() - starttime))
        slackconstraintlist = testconstraint(XYarray, constraintvars, slackconstraint_noparams, allposdict)
        # if slackconstraint is True then output regime is 0
        # if slackconstraint is False then output regime is 1
        outputregime = [int(not slackconstraint) for slackconstraint in slackconstraintlist]

        if printdetails is True:
            if printvars is None:
                printvars = []
            for var in printvars:
                varval = [XYarray[t, allposdict[var]] for t in range(solverperiods)]
                print('Values for variable ' + var + ': ' + str(varval))
            print('Constraint is slack: ' + str(slackconstraintlist))

        # update the regime
        if regimeupdatefunc == 'occbin':
            inputregime, stop, triedalready = regimeupdate_occbin(inputregime, outputregime, triedalready, printwarnings = printwarnings)
        elif regimeupdatefunc == 'binduntilnot':
            inputregime, stop, exitbindat = regimeupdate_binduntilnot(inputregime, outputregime, exitbindat)
        else:
            raise ValueError('regimeupdatefunc specified does not correspond to an actual regime update function.')

        # stop if appropriate
        if stop is True:
            break

        if printdetails is True:
            if starttime is None:
                print('\nFinished iteration ' + str(i + 1) + '.')
            else:
                print('\nFinished iteration ' + str(i + 1) + '. Time: ' + str(datetime.datetime.now() - starttime))
    if i == maxiter - 1:
        if printwarnings is True:
            print('Warning: Reached maximumum iteration.')

    if inputregime[-1] != 0:
        if printwarnings is True:
            print('Warning: At T - 1, the constraint binds. This implies you might want to increase the number of solver periods.')

    return(Zarray, XYarray, inputregime)


def oneperiodshocksolve_binduntilnot_savepolicy(Zm1, epsilon0, ABCDE_bind, I_regime0, constraintvars, slackconstraint_noparams, allposdict, numstates, polfuncsaved, solverperiods = 100, printdetails = False, printvars = None, starttime = None):
    """
    
    """

    numvar = len(Zm1)
    T = solverperiods

    bindsfornumperiods = 0
    # continue until solverperiods + 1 since when i == 0, bind 0 times and thus when i == solverperiods - 1, only binds solverperiods - 1, not solverperiods times
    for bindsfornumperiods in range(solverperiods + 1):
        if printdetails is True:
            if starttime is None:
                print('\nStarting bindsfornumperiods ' + str(bindsfornumperiods) + ':')
            else:
                print('\nStarting bindsfornumperiods ' + str(bindsfornumperiods) + '. Time: ' + str(datetime.datetime.now() - starttime))

        # get relevant policy function
        if bindsfornumperiods > len(polfuncsaved):
            raise ValueError('Something is wrong. This should already be defined since I only raise bindsfornumperiods by 1 each period.')
        elif bindsfornumperiods == len(polfuncsaved):
            # need to add an additional period to polfuncsaved
            Garray = np.empty([T, numvar])
            Harray = np.empty([T, numvar, numvar])

            Garray[1:T, :] = polfuncsaved[-1][0][0: T - 1, :]
            Harray[1:T, :] = polfuncsaved[-1][1][0: T - 1, :]

            At = ABCDE_bind[0]
            Bt = ABCDE_bind[1]
            Ct = ABCDE_bind[2]
            Dt = ABCDE_bind[3]
            Et = ABCDE_bind[4]

            coeff = - np.linalg.inv(Bt + Ct.dot(Harray[1]))
            Garray[0] = coeff.dot(Ct.dot(Garray[1]) + Dt)
            Harray[0] = coeff.dot(At)
            I0 = coeff.dot(Et)

            polfuncsaved.append([Garray, Harray, I0])
        else:
            # necessary policy function for this number of bind periods already saved
            None

        # given the policy function, get path of variables over time
        if starttime is not None:
            print('Iterating over the policy function. Time: ' + str(datetime.datetime.now() - starttime))
        from regime_func import simulatepath
        Zarray, XYarray = simulatepath(polfuncsaved[bindsfornumperiods][0], polfuncsaved[bindsfornumperiods][1], polfuncsaved[bindsfornumperiods][2], Zm1, epsilon0, numstates)

        # get whether the constraint is binding or slack at each period
        if starttime is not None:
            print('Testing whether the constraint is slack. Time: ' + str(datetime.datetime.now() - starttime))
        slackconstraintlist = testconstraint(XYarray, constraintvars, slackconstraint_noparams, allposdict)

        if printdetails is True:
            if printvars is None:
                printvars = []
            for var in printvars:
                varval = [XYarray[t, allposdict[var]] for t in range(solverperiods)]
                print('Values for variable ' + var + ': ' + str(varval))
            print('Constraint is slack: ' + str(slackconstraintlist))

        
        bindsfornumperiods_simulation = 0
        for j in range(len(slackconstraintlist)):
            if bool(slackconstraintlist[j]) is False:
                # + 1 since if bind until period j then actually binds for j + 1 periods
                bindsfornumperiods_simulation = j + 1
        if bindsfornumperiods_simulation <= bindsfornumperiods:
            break

        if printdetails is True:
            if starttime is None:
                print('\nFinished bindsfornumperiods ' + str(bindsfornumperiods) + '.')
            else:
                print('\nFinished bindsfornumperiods ' + str(bindsfornumperiods) + '. Time: ' + str(datetime.datetime.now() - starttime))

    return(Zarray, XYarray, bindsfornumperiods)


def manyperiodshocksolve(Zm1, epsilon, ABCDE_bind, ABCDE_nobind, H_Tplus, I_regime0, constraintvars, slackconstraint_noparams, allposdict, numstates, regimeupdatefunc = 'occbin', maxiter = 100, solverperiods = 100, printdetails = False, printvars = None, starttime = None, printwarnings = True):
    """
    epsilon should be of dimenion  numshocks x numvar
    Zm1 should be of dimention numvar x 1 (or just a vector)
    """
    numvar = len(Zm1)
    numshocks = epsilon.shape[0]

    Zarray = np.empty([numshocks, numvar])
    regime = []

    if regimeupdatefunc == 'binduntilnot_savepolicy':
        # if I use 100 solverperiods only 100 possible policy function iterations
        # save these policy function iterations as I compute them so I don't need to compute them again
        # I can also save time by computing the policy function when the constraint binds i + 1 times from when it binds i times
        polfuncsaved = []

        # get initial array
        Garray0 = np.zeros([solverperiods, numvar])
        Harray0 = np.empty([solverperiods, numvar, numvar])
        for t in range(solverperiods):
            Harray0[t] = H_Tplus
        polfuncsaved.append([Garray0, Harray0, I_regime0])


    for numshock in range(numshocks):
        if printdetails is True:
            if starttime is None:
                print('\n\nPERIOD ' + str(numshock + 1))
            else:
                print('\n\nPERIOD ' + str(numshock + 1) + '. Time: ' + str(datetime.datetime.now() - starttime))
                
        epsilon0 = epsilon[numshock, :]
        
        # use computations from previous period if no shocks this period, its not the first period and Zarray_t has high enough solveperiods to do this
        if np.all(epsilon0 == np.zeros(len(epsilon0))) and numshock != 0 and numshock - numshocklastrun < np.shape(Zarray_t)[0]:
            if printdetails is True:
                print('Using computations from time ' + str(numshocklastrun) + ' for time ' + str(numshock) + ' to save time.')
            # take path of X for this period from the previous iteration
            # saves time on iterations when I have a lot of zeros in the shocks
            # Zarray_t last generated at time tlastrun so want the t - tlastrun time element of Zarray_t
            Zarray[numshock] = Zarray_t[numshock - numshocklastrun]
            if regimeupdatefunc != 'binduntilnot_savepolicy':
                regime.append(inputregime_t[numshock - numshocklastrun])
            else:
                # numshock - numshocklastrun is the index of the period I'm considering relative to the last simulation
                # if index was 1 then need bindsfornumperiods to be at least 2 if we are at the zlb
                if numshock - numshocklastrun < bindsfornumperiods:
                    regime.append(1)
                else:
                    regime.append(0)
        else:
            if numshock == 0:
                Xtm1 = Zm1
            else:
                Xtm1 = Zarray[numshock - 1]

            if regimeupdatefunc != 'binduntilnot_savepolicy':
                Zarray_t, XYarray_t, inputregime_t = oneperiodshocksolve(Xtm1, epsilon0, ABCDE_bind, ABCDE_nobind, H_Tplus, I_regime0, constraintvars, slackconstraint_noparams, allposdict, numstates, regimeupdatefunc = regimeupdatefunc, maxiter = maxiter, solverperiods = solverperiods, printdetails = printdetails, printvars = printvars, starttime = starttime, printwarnings = printwarnings)

                Zarray[numshock] = Zarray_t[0]
                regime.append(inputregime_t[0])
                numshocklastrun = numshock
            else:
                # binduntilnot_savepolicy

                # don't need ABCDE_nobind for this method
                Zarray_t, XYarray_t, bindsfornumperiods = oneperiodshocksolve_binduntilnot_savepolicy(Xtm1, epsilon0, ABCDE_bind, I_regime0, constraintvars, slackconstraint_noparams, allposdict, numstates, polfuncsaved, solverperiods = solverperiods, printdetails = printdetails, printvars = printvars, starttime = starttime)

                Zarray[numshock] = Zarray_t[0]
                if bindsfornumperiods > 0:
                    regime.append(1)
                else:
                    regime.append(0)
                numshocklastrun = numshock

    # replace Zarray with XYarray
    XYarray = np.empty([numshocks, numvar])
    XYarray[0, 0: numstates] = Zm1[0: numstates]
    XYarray[1: , 0: numstates] = Zarray[0: numshocks - 1, 0: numstates]
    XYarray[:, numstates: ] = Zarray[:, numstates: ]
                

    return(XYarray, regime)


# General:{{{1
def myoccbin(inputdict_nobind, inputdict_bind, slackconstraint, shockpath, savefolder = None, replacedir = True, solverperiods = None, printdetails = False, printvars = None, irf = False, printprobbind = False, regimeupdatefunc = 'occbin', partialeval_paramssdict = None, printwarnings = True):
    """
    slackconstraint needs a strict inequality i.e. I > 0, so that when when the constraint does bind i.e. I = 0, the slackconstraint is not satisfied.

    shockpath must be in format nshocks x T

    partialeval_paramssdict is not None: means that I've already run getmodel, getfxefy_inputdict, funcstoeval_inputdict on both inputdict_nobind and inputdict_bind. Now I just need to apply the paramssdict in partialeval_paramssdict to fxe_f, fxep_f, ... and also get 
    """
    if printdetails is True:
        start = datetime.datetime.now()
    else:
        start = None

    if solverperiods is None:
        solverperiods = 100

    if savefolder is not None:
        if replacedir is True:
            if os.path.isdir(savefolder):
                shutil.rmtree(savefolder)
        else:
            if os.path.isdir(savefolder):
                print('Folder already exists. Folder: ' + savefolder + '.')
                return(0)
        if not os.path.isdir(savefolder):
            os.mkdir(savefolder)

    
    # additional arguments for inputdict if I haven't run getmodel yet
    if partialeval_paramssdict is None:
        # don't solve for steady state in binding case
        inputdict_bind['skipsscheck'] = True

        # add starttime to inputdicts
        if printdetails is True:
            inputdict_nobind['startruntime'] = start
            inputdict_bind['startruntime'] = start
        else:
            inputdict_nobind['printrundetails'] = False
            inputdict_bind['printrundetails'] = False

    # solve for ABCDE func:{{{
    def solveABCDEinputdict(inputdict, nobind = True):
        if nobind is True:
            iterationstring = 'nobind'
        else:
            iterationstring = 'bind'

        if partialeval_paramssdict is not None:
            if printdetails is True:
                print('\nStage: Starting to get ' + iterationstring + ' partialtofull inputdict. Time: ' + str(datetime.datetime.now() - start))
            from dsgediff_func import partialtofulleval_inputdict
            inputdict = partialtofulleval_inputdict(inputdict, partialeval_paramssdict)

            if printdetails is True:
                print('\nStage: Starting to get ' + iterationstring + ' sseqs. Time: ' + str(datetime.datetime.now() - start))
            from dsgediff_func import get_ss_eqs
            ss_eqs = get_ss_eqs(inputdict['equations'], inputdict['replaceuseddict'])
        else:
            if printdetails is True:
                print('\nStage: Starting to get ' + iterationstring + ' model. Time: ' + str(datetime.datetime.now() - start))

            from dsgesetup_func import getmodel_inputdict
            inputdict = getmodel_inputdict(inputdict)

            if printdetails is True:
                print('\nStage: Get ABCDE for ' + iterationstring + ' model. Time: ' + str(datetime.datetime.now() - start))

            # analytical deriv
            from dsgediff_func import getfxefy_inputdict
            getfxefy_inputdict(inputdict)
            # numerical deriv
            from dsgediff_func import getnfxenfy_inputdict
            getnfxenfy_inputdict(inputdict)

            # solve for ss_eqs
            if printdetails is True:
                print('\nStage: Starting to get ' + iterationstring + ' sseqs. Time: ' + str(datetime.datetime.now() - start))
            from dsgediff_func import get_ss_eqs
            ss_eqs = get_ss_eqs(inputdict['equations_noparams'], inputdict['varfplusonlyssdict'])

        # split nfxe
        from dsgediff_func import convertsplitxe
        inputdict['nfx'], inputdict['nfxp'], inputdict['nfe'], inputdict['nfep'] = convertsplitxe(inputdict['nfxe'], inputdict['nfxep'], numstates = len(inputdict['states']))

        # convert to ABCDe
        # note that I have to use fx etc. for the equations without shocks i.e. equations rather than equations_plus
        nvars = len(inputdict['states'] + inputdict['controls'])
        from dsgediff_func import get_ABCDE_form
        ABCDE = get_ABCDE_form(inputdict['nfx'][: nvars, :], inputdict['nfxp'][: nvars, :], inputdict['nfy'][: nvars, :], inputdict['nfyp'][: nvars, :], inputdict['nfe'][: nvars, :], inputdict['nfep'][: nvars, :], ss_eqs)

        return(ABCDE)

    # solve for ABCDE func:}}}

    ABCDE_nobind = solveABCDEinputdict(inputdict_nobind)
    ABCDE_bind = solveABCDEinputdict(inputdict_bind)

    if savefolder is not None:
        # add option to save pickle
        from dsgesetup_func import savemodel_inputdict
        savemodel_inputdict(inputdict_nobind, os.path.join(savefolder, 'inputdict_nobind.pickle'))

    # compute H_Tplus:{{{
    if printdetails is True:
        print('Stage: Get policy function for nobind model. Time: ' + str(datetime.datetime.now() - start))

    from dsge_bkdiscrete_func import gxhx
    inputdict_nobind['gx'], inputdict_nobind['hx'] = gxhx(inputdict_nobind['nfxe'], inputdict_nobind['nfxep'], inputdict_nobind['nfy'], inputdict_nobind['nfyp'])
    from dsge_bkdiscrete_func import gxhx_splitbyshocks
    gx_noshocks, gx_shocks, hx_noshocks, hx_shocks = gxhx_splitbyshocks(inputdict_nobind['gx'], inputdict_nobind['hx'], len(inputdict_nobind['shocks']))
    nstates = len(inputdict_nobind['states'])
    ncontrols = len(inputdict_nobind['controls'])
    nvars = nstates + ncontrols
    H_Tplus = np.zeros([nvars, nvars])
    H_Tplus[0: nstates, 0: nstates] = hx_noshocks
    H_Tplus[nstates: nvars, 0: nstates] = gx_noshocks


        
    # get I_regime0 where X_t = G + HX_{t - 1} + Iepsilon_t
    # need for when do initial iteration - we guess initially that we never hit the constraint in which case we can just use H_Tplus and I_regime0 to do this
    I_regime0 = np.vstack((hx_shocks, gx_shocks))

    # get the variables in the constraint
    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    from stringvar_func import varsinstringanddict
    constraintvars = varsinstringanddict(slackconstraint, inputdict_nobind['statescontrolsshocks_p'])

    # remove parameters from the constraint
    if partialeval_paramssdict is True:
        paramssonlydict = {param: partialeval_paramssdict[param] for param in partialeval_paramssdict if param not in inputdict['statescontrolsshocks_p']}
    else:
        paramssonlydict = inputdict_nobind['paramonlyssdict']
    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    from stringvar_func import replacevardict
    slackconstraint_noparams = replacevardict(slackconstraint, paramssonlydict)

    # main solve function
    Zm1 = np.zeros([nvars, 1])
    if printdetails is True:
        print('\nStage: Starting main occbin solve. Time: ' + str(datetime.datetime.now() - start))
    Xarray, regime = manyperiodshocksolve(Zm1, shockpath, ABCDE_bind, ABCDE_nobind, H_Tplus, I_regime0, constraintvars, slackconstraint_noparams, inputdict_nobind['statecontrolposdict'], nstates, regimeupdatefunc = regimeupdatefunc, maxiter = 100, solverperiods = solverperiods, printdetails = printdetails, printvars = printvars, starttime = start, printwarnings = printwarnings)

    if savefolder is not None:
        with open(os.path.join(savefolder, 'Xarray.pickle'), 'wb+') as handle:
            pickle.dump(Xarray, handle)
        with open(os.path.join(savefolder, 'regime.pickle'), 'wb+') as handle:
            pickle.dump(regime, handle)

    # print prob constraint binds
    if printprobbind is True or printdetails is True:
        print('Probability constraint binds: ' + str(np.mean(regime)))

    # IRFs
    if irf is True:
        if printdetails is True:
            print('\nStage: Starting IRFs. Time: ' + str(datetime.datetime.now() - start))
        pos = [inputdict_nobind['statecontrolposdict'][var] for var in inputdict_nobind['mainvars']]

        Xarray2 = Xarray[:, pos]

        sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
        from matplotlib_func import gentimeplots_basic
        gentimeplots_basic(Xarray2.transpose(), inputdict_nobind['mainvars'])

    return(inputdict_nobind, inputdict_bind, Xarray, regime)

        



