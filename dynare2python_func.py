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
        importattr(__projectdir__ / Path('dsgesetup_func.py'), 'printruntime')(inputdict, 'Starting Dynare format')

        if 'shocks' not in inputdict:
            inputdict['shocks'] = []

        # need to specify steady state since need to give steady state if define K_1, K_2, K_3
        if 'dynarevarssdict' not in inputdict:
            inputdict['dynarevarssdict'] = {}

        # need to specify logvars
        if 'dynarelogvars' not in inputdict:
            inputdict['dynarelogvars'] = []

        retdict = importattr(__projectdir__ / Path('dsgesetup_func.py'), 'convertdynareformat')(inputdict['dynareequations'], inputdict['dynarevariables'], inputdict['dynarevarssdict'], inputdict['dynarelogvars'], futureending = inputdict['futureending'], ssending1 = inputdict['ssending1'], loglineareqs = inputdict['loglineareqs'], saveeqsfile = saveeqsfile)
        inputdict['equations'] = retdict[0]
        inputdict['states'] = retdict[1]
        inputdict['controls'] = retdict[2]
        inputdict['varssdict'] = retdict[3]
        inputdict['logvars'] = retdict[4]


