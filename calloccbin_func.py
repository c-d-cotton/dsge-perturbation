#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np
import scipy.io as sio
import shutil
import subprocess

def calloccbin_oneconstraint(inputdict_nozlb, inputdict_zlb, bindingconstraint, slackconstraint, shocks, folder, occbinpath = None, deletedir = True, simperiods = None, run = False, irf = False):
    """
    Create the two necessary mod files and the matlab run file for occbin to run.

    simperiods is how many periods simulate forward. So at time t, get how agents act until t + simperiods. Need to not be in binding regime at t + simperiods.
    """
    if irf is True:
        run = True

    if simperiods is None:
        simperiods = 100

    if deletedir is True:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    if occbinpath is None:
        if os.path.isfile(os.path.join(__projectdir__, 'paths/occbin.txt')):
            with open(os.path.join(__projectdir__, 'paths/occbin.txt')) as f:
                occbinpath = f.read()
            if occbinpath[-1] == '\n':
                occbinpath = occbinpath[: -1]
        else:
            raise ValueError('Need to specify occbinpath or place occbinpath in dsge-perturbation/paths/occbinpath.txt.')

    # Get inputdicts:{{{

    # add python2dynare filename
    inputdict_nozlb['python2dynare_savefilename'] = os.path.join(folder, 'nozlb.mod')
    inputdict_zlb['python2dynare_savefilename'] = os.path.join(folder, 'zlb.mod')

    # add simulation to python2dynare for nozlb case
    inputdict_nozlb['python2dynare_simulation'] = 'stoch_simul(order=1,nocorr,nomoments,irf=0);'
    # shouldn't include this since will be indeterminate
    # inputdict_zlb['python2dynare_simulation'] = 'stoch_simul(order=1,nocorr,nomoments,irf=0,print);'

    # skip sscheck for zlb case (since will be incorrect)
    inputdict_zlb['skipsscheck'] = True

    # need to add to inputdict so adds to Dynare file
    inputdict_nozlb['shockpath'] = shocks
    inputdict_zlb['shockpath'] = shocks

    # }}}

    # get Dynare files{{{
    from dsgesetup_func import getmodel_inputdict
    getmodel_inputdict(inputdict_nozlb)
    from dsgesetup_func import getmodel_inputdict
    getmodel_inputdict(inputdict_zlb)

    from python2dynare_func import python2dynare_inputdict
    python2dynare_inputdict(inputdict_nozlb)
    from python2dynare_func import python2dynare_inputdict
    python2dynare_inputdict(inputdict_zlb)

    # }}}

    # now create matlab file{{{
    addoccbinpath = "addpath('" + occbinpath + "');\n"

    setupoccbin = 'setpathdynare;\n'


    # GET MATLAB ARRAY
    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    from matlab_func import getmatlabmatrixstring
    shockssequence = getmatlabmatrixstring(shocks)

    defineshockssequence = 'shockssequence = ' + shockssequence + '\n'
    irfshocks = '[' +  '; '.join(["'" + shock + "'" for shock in inputdict_nozlb['shocks']]) + ']'

    savematrices = '\n'.join(["save('save_" + name + ".mat', '" + name + "');" for name in ['zdatalinear', 'zdatapiecewise', 'zdatass', 'oobase', 'Mbase']]) + '\n'


    funcrun = "[zdatalinear zdatapiecewise zdatass oobase Mbase] = solve_one_constraint('nozlb', 'zlb', '" + bindingconstraint + "', '" + slackconstraint + "', shockssequence, " + irfshocks + ', ' + str(simperiods) + ');\n'

    string = addoccbinpath + setupoccbin + defineshockssequence + funcrun + savematrices

    with open(os.path.join(folder, 'calloccbin.m'), 'w+') as f:
        f.write(string)

    # }}}

    if run is True:
        cwd = os.getcwd()
        os.chdir(folder)

        if os.path.isfile(__projectdir__ / Path('paths/rundynarewithoctave.txt')) is True:
            matlabordynare = 'octave'
        else:
            matlabordynare = 'matlab'

        subprocess.call([matlabordynare, 'calloccbin.m'])

        os.chdir(cwd)

    if irf is True:
        if run is True:
            getirfs_occbin(folder)
        else:
            raise ValueError('Need to set run to be True if irf is True.')


def getirfs_occbin(folder):
    from dsgediff_func import irf_matlab_inputdict
    # irf_matlab_inputdict(os.path.join(folder, 'nozlb_inputdict.pickle'), os.path.join(folder, 'save_zdatapiecewise.mat'), matlabsavefunc = lambda x: x['zdatapiecewise'], rowisvars = False)

    # get names of variables
    data = sio.loadmat(os.path.join(folder, 'nozlb_results.mat'))
    names = data['M_']['endo_names'][0][0]
    names = [name.strip() for name in names]

    sys.path.append(str(__projectdir__ / Path('submodules/python-math-func/')))
    from matlab_func import irf_matlab
    irf_matlab(os.path.join(folder, 'save_zdatapiecewise.mat'), names, matlabsavefunc = lambda x: x['zdatapiecewise'], rowisperiods = True)



