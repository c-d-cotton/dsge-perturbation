#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')


# Get From Dynare Format:{{{1
def convertdynareformat(equations, variables, dvarssdict, dlogvars, futureending = futureending_default, ssending1 = ssending1_default, loglineareqs = False):
    """
    Note that I need to input the varssdict as dvarssdict
    """
    import numpy as np
    import re

    maxdict = {}
    mindict = {}
    for var in variables:
        varnums = []
        pattern = re.compile('(?<![a-z0-9_])' + var + '\(([0-9+-]+)\)')
        for i in range(0, len(equations)):
            # firstly replace just variable i.e. c with c(0)
            equations[i] = re.sub('(?<![a-z0-9_])' + var + '(?![a-z0-9_(])', var + '(0)', equations[i])
            
                
            # looking for c(+1), c(-1)
            matches = pattern.finditer(equations[i])
            for match in matches:
                varnums.append(int(match.group(1)))
        if varnums == []:
            raise ValueError('Variable: ' + var + ' is not used in the equations.')
        # get highest and lowest periods
        mindict[var] = np.min(varnums)
        maxdict[var] = np.max(varnums)
        # if only have c(0), raise max to 1
        if mindict[var] == maxdict[var]:
            maxdict[var] = maxdict[var] + 1

    for var in variables:
        for i in range(0, len(equations)):
            # replace a(-1) with a_m0, a(0) with a_0

            # firstly replace all except highest period of var
            for j in range(mindict[var], maxdict[var]):
                if j <= 0:
                    joldstr = str(j)
                else:
                    joldstr = '\+?' + str(j)
                if j < 0:
                    jnewstr = '_m' + str(abs(j))
                else:
                    jnewstr = '_' + str(j)

                equations[i] = re.sub('(?<![a-z0-9_])' + var + '\(' + joldstr + '\)', var + jnewstr, equations[i])

            # now replace highest period
            # a(+1) with a_0_p

            # oldstr is maxdict
            j = maxdict[var]
            if j <= 0:
                joldstr = str(j)
            else:
                joldstr = '\+?' + str(j)
            # newstr is maxdict - 1
            j = maxdict[var] - 1
            if j < 0:
                jnewstr = '_m' + str(abs(j)) + futureending
            else:
                jnewstr = '_' + str(j) + futureending
            
            # note the j+1
            equations[i] = re.sub('(?<![a-z0-9_])' + var + '\(' + joldstr + '\)', var + jnewstr, equations[i])

    if loglineareqs is True:
        for var in variables:
            dvarssdict[var] = 0

    # now sorting out the variables and dealing with the equations:
    states = []
    controls = []
    varssdict = {}
    logvars = []
    for var in variables:
        for i in range(mindict[var], maxdict[var]):
            # add states/controls
            # add varssdict
            if i < 0:
                istr = '_m' + str(abs(i))
                states.append(var + istr)
                varssdict[var + istr + ssending1] = dvarssdict[var + ssending1]
                if var in dlogvars:
                    logvars.append(var + istr)
            else:
                istr = '_' + str(i)
                controls.append(var + istr)
                varssdict[var + istr + ssending1] = dvarssdict[var + ssending1]
                if var in dlogvars:
                    logvars.append(var + istr)
            
            # adding future variable equations
            # i.e. c_1 = c_0_p
            if i != mindict[var]:
                if i < 0:
                    jhigher = 'm' + str(abs(i))
                else:
                    jhigher = str(i)
                if i - 1 < 0:
                    jlower = 'm' + str(abs(i - 1))
                else:
                    jlower = str(i - 1)

                equations.append(var + '_' + jhigher + ' - ' + var + '_' + jlower + '_p')

    return(equations, states, controls, varssdict, logvars)

def dynareadd_inputdict():

    if 'dynareequations' in inputdict:
        from dsgesetup_func import printruntime
        printruntime(inputdict, 'Starting Dynare format')

        if 'shocks' not in inputdict:
            inputdict['shocks'] = []

        # need to specify steady state since need to give steady state if define K_1, K_2, K_3
        if 'dynarevarssdict' not in inputdict:
            inputdict['dynarevarssdict'] = {}

        # need to specify logvars
        if 'dynarelogvars' not in inputdict:
            inputdict['dynarelogvars'] = []

        from dsgesetup_func import convertdynareformat
        retdict = convertdynareformat(inputdict['dynareequations'], inputdict['dynarevariables'], inputdict['dynarevarssdict'], inputdict['dynarelogvars'], futureending = inputdict['futureending'], ssending1 = inputdict['ssending1'], loglineareqs = inputdict['loglineareqs'], saveeqsfile = saveeqsfile)
        inputdict['equations'] = retdict[0]
        inputdict['states'] = retdict[1]
        inputdict['controls'] = retdict[2]
        inputdict['varssdict'] = retdict[3]
        inputdict['logvars'] = retdict[4]


