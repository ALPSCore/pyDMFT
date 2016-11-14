import pyalps 
#import pyalps.cthyb as cthyb # solver module
#import pyalps.mpi as mpi     # MPI library
import numpy as np
from numpy.linalg import inv
import sys
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess
import h5py
import fourier_transform as ft
import tempfile
from collections import OrderedDict
from lib import *
from h5dump import *

def write_parms(f, parms):
    for k,v in parms.iteritems():
        if isinstance(v,str):
            print >>f, k,"=",'"',v,'";'
        else:
            print >>f, k,"=", v, ';'

def call_hyb_matrix(app_parms, tau_mesh, hyb_tau, hyb, rotmat, mu):
    ntau = len(tau_mesh)-1
    norb = hyb_tau.shape[1]
    path_input = app_parms['prefix']+'_input_hyb'
    path_hyb = app_parms['prefix']+'_F'
    print "Using ", path_input, path_hyb

    #### impurity solver ####
    hyb_f = open(path_hyb,'w')
    for i in range(ntau+1):
        print >>hyb_f, " dummy 0.0 ", tau_mesh[i], " ",
        for iflavor in range(2*norb):
            iorb = iflavor/2
            isp = iflavor%2
            for iflavor2 in range(2*norb):
                iorb2 = iflavor2/2
                isp2 = iflavor2%2
                if isp==isp2:
                    print >>hyb_f, -hyb_tau[ntau-i,iorb,iorb2], " ",
                else:
                    print >>hyb_f, 0.0, " ",
        print >>hyb_f, ""
    hyb_f.close()

    hyb_f = open(path_hyb+'-nreal','w')
    hyb_f2 = open(path_hyb+'-nimag','w')
    for i in range(ntau):
        for iorb in range(norb):
            for iorb2 in range(norb):
                print >>hyb_f, hyb[i,iorb,iorb2].real, " ",
                print >>hyb_f2, hyb[i,iorb,iorb2].imag, " ",
        print >>hyb_f, ""
        print >>hyb_f2, ""
    hyb_f.close()
    hyb_f2.close()

    #rotation of hybfunction
    #evals,evecs=eigh_ordered(hyb[0,:,:])
    f = open(path_hyb+'-rot','w')
    for i in range(2*norb):
        iorb = i/2
        ispin = i%2
        for j in range(2*norb):
            jorb = j/2
            jspin = j%2
            if ispin==jspin:
                print >>f, i, j, rotmat[iorb,jorb]
            else:
                print >>f, i, j, 0.0
    f.close()

    #generating input files...
    input_f = open(path_input,'w')
    print >>input_f, "{"
    parms=OrderedDict()
    parms['N_TAU'] = app_parms['NMATSUBARA']
    parms['QUARTERBANDWIDTH'] = app_parms['t']
    parms['BETA'] = app_parms['BETA']
    parms['L'] = app_parms['N_ORB']
    parms['W'] = 1
    parms['SITES'] = app_parms['N_ORB']
    parms['SPINS'] = 2
    parms['FLAVORS'] = 2*app_parms['N_ORB']

    parms["U'"] = app_parms["U'"]
    parms["Uprime"] = app_parms["U'"]

    parms["t'"] = app_parms["t'"]
    parms["Tprime"] = app_parms["t'"]

    parms['J' ] = 0
    parms['JP' ] = 0
    parms['JH' ] = 0

    parms['mu'] = mu
    parms['MU'] = mu

    parms['F'] = path_hyb
    if 'BASIS_ROT' in app_parms and app_parms['BASIS_ROT']!=0:
        parms['ROTATE_F'] = path_hyb+'-rot'
    for i in range(norb):
        parms['E'+str(i)] = app_parms['E'+str(i)]
        parms['ME'+str(i)] = -app_parms['E'+str(i)]
        parms['U'+str(i)] = app_parms['U'+str(i)]
    for k,v in app_parms.items():
        m = re.search('^IMP_SLV_(.+)$',k)
        if m!=None:
            print k,v,m.group(0),m.group(1)
            parms[m.group(1)] = v
    write_parms(input_f, parms)
    print >>input_f, "}"
    input_f.close()

    cmd='parameter2xml -f '+path_input
    print cmd
    os.system(cmd)
    cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' --mpi --Tmin 10 --write-xml '+path_input+'.in.xml > output_'+path_input
    print cmd
    sys.stdout.flush()
    subprocess.call(cmd, shell=True)
    print "Finished hybridization"
    sys.stdout.flush()

    #Load measured observables
    result = {}
    foutput=path_input+'.task1.out.h5'

    #Load all observables
    keys,means,errors = load_observables("./"+foutput)
    obs = {}
    for i in range(len(keys)):
        obs[keys[i]+'_mean'] = means[i]
        obs[keys[i]+'_error'] = errors[i]

    sys.stdout.flush()

    #<Sign>
    sign = float(means[keys.index("Sign")])
    print "sign=",sign

    #<n_i>
    result["n"] = means[keys.index("n")]/sign
    result["n_rotated"] = means[keys.index("n_rotated")]/sign

    #Im G(tau)
    #Gij: (i,j,tau)
    G_tau = -1.0*(means[keys.index("Greens")]).reshape(2*norb,2*norb,ntau+1).transpose([2,0,1])/sign
    G_tau[0,:,:] *= 2 #because the bin size is a half at \tau=0 and beta.
    G_tau[ntau,:,:] *= 2
    for iflavor in range(2*norb):
        G_tau[0,iflavor,iflavor] = -(1.0-result["n"][iflavor])
        G_tau[ntau,iflavor,iflavor] = -1.0*result["n"][iflavor]
    result["Greens_imag_tau"] = G_tau


    sys.stdout.flush()
    return result, obs
