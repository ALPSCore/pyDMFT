#import pyalps 
import numpy as np
from numpy import cos,pi
from numpy.linalg import inv
import sys
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess
import fourier_transform as ft
from lib import *
import scipy.optimize as sciopt
from hyb_matrix import call_hyb_matrix,symmetrize_G_tau
from ct_int import call_ct_int
from h5dump import *
import copy
import h5py
import impurity_model
import hashlib


exclude_list = ['G_latt', 
    'Greens_Im_error', 
    'Greens_Im_mean', 
    'Greens_Re_error', 
    'Greens_Re_mean', 
    'Greens_legendre_Im_error', 
    'Greens_legendre_Re_error', 
    'Greens_legendre_rotated_Im_error', 
    'Greens_legendre_rotated_Im_mean', 
    'Greens_legendre_rotated_Re_error', 
    'Greens_legendre_rotated_Re_mean', 
    'Greens_rotated_Im_error', 
    'Greens_rotated_Im_mean', 
    'Greens_rotated_Re_error', 
    'Greens_rotated_Re_mean', 
    'hyb_n', 
    'hyb_tau']

#### setup parameters ####
app_parms = inp.read_input_parms(sys.argv[1]+'.h5')
app_parms['prefix'] = sys.argv[1]
h5f = h5py.File(sys.argv[1]+'.h5','r')
imp_model = impurity_model.OrbitalModel(h5f)
h5f.close()

#app_parms["PREFIX_IMP_SLV_WORK_FILE"] = "ct-hyb-"+str(random.randint(0,100000))
#app_parms["PREFIX_IMP_SLV_WORK_FILE"] = "ct-hyb-"+hashlib.sha224(app_parms['prefix']).hexdigest()
app_parms["PREFIX_IMP_SLV_WORK_FILE"] = app_parms['prefix'].replace('.', "_")
print "Using PREFIX_IMP_SLV_WORK_FILE ", app_parms["PREFIX_IMP_SLV_WORK_FILE"]

nsbl=app_parms["N_SUBLATTICE"]
norb=app_parms["N_ORB"]
nflavor=2*norb
nflavor_sbl=nflavor/nsbl
PM=True
vmix=1.0
vmu=app_parms["MU"]
tote=app_parms["N_ELEC"]
vbeta=app_parms["BETA"]
vconverged=app_parms["CONVERGED"]
ndiv_tau=app_parms["NMATSUBARA"]
cutoff_fourie=app_parms["CUTOFF_FOURIE"]

matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
tau=np.zeros((ndiv_tau+1,),dtype=float)

# uncorrelated lattice Green function
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta
for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it

#Fourie transformer
fourie_transformer = ft.FourieTransformer(imp_model)
Hk_mean = 1.*imp_model.get_moment(1)

if 'mu' in app_parms:
    raise RuntimeError("Do not use mu")

isc = 0

#compute G0
#G0,tote_tmp = imp_model.calc_Glatt(vbeta,matsubara_freq,np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex),vmu)
#np.save(app_parms["prefix"]+"-G0", G0)
#print "Computed G0"

#Local projectors
local_projectors = []
if 'N_LOCAL_PROJECTORS' in app_parms:
    nprj = app_parms['N_LOCAL_PROJECTORS']
    print "Number of local projectors ", nprj
    for iprj in xrange(nprj):
        local_projectors.append(app_parms['LOCAL_PROJECTOR'+str(iprj)])
    check_projectors(local_projectors)

#### self-consistent loop ####
def calc_diff(self_ene_dmp_in):
    time_start = time.time()
    global isc, vmu

    #Rescale self_energy
    self_ene_in = np.array(self_ene_dmp_in)
    for iflavor in range(nflavor):
        for iflavor2 in range(nflavor):
            self_ene_in[:,iflavor,iflavor2] = self_ene_dmp_in[:,iflavor,iflavor2]/dmp_fact

    #Symmetrize self energy
    #self_ene_in = self_ene_in.reshape((ndiv_tau, nsbl, nflavor_sbl, nsbl, nflavor_sbl))
    #for isbl in xrange(nsbl):
        #self_ene_in[:,isbl,:,isbl,:] = 1.0*symmetrize_G_tau(app_parms, self_ene_in[:,isbl,:,isbl,:])
        #print self_ene_in[0,isbl,0,isbl,2], self_ene_in[0,isbl,0,isbl,4]
    #self_ene_in = self_ene_in.reshape((ndiv_tau, nflavor, nflavor))

    #### Lattice Green function ####
    # Make sure we recompute G_lattice after updating vmu.
    # Otherwise, Delta(iomega) may not vanish at high frequencies.
    print "Computing lattice Green's function..."
    sys.stdout.flush()
    time1 = time.time()
    if ('OPT_MU' in app_parms and app_parms['OPT_MU'] > 0) and isc >= app_parms['FIX_CHEM']:
        nite = 5
        if 'N_CHEM_LOOP' in app_parms:
            nite = app_parms['N_CHEM_LOOP']
        for i_chem in range(nite): #By default, we do it 5 times.
            G_latt,tote_tmp = imp_model.calc_Glatt(vbeta,matsubara_freq,self_ene_in,vmu)
            G_latt_tau = fourie_transformer.G_freq_to_tau(G_latt,ndiv_tau,vbeta,cutoff_fourie)
    
            ntot = 0.0
            for ie in range(nflavor):
                ntot += -G_latt_tau[-1,ie,ie].real
                print ie, -G_latt_tau[-1,ie,ie].real
            vmu = (vmu-np.abs(app_parms['OPT_MU'])*(ntot-2*app_parms["N_ELEC"])).real
            print "tot_Glatt = ", ntot
            print "new mu = ", vmu
            sys.stdout.flush()
    G_latt,tote_tmp = imp_model.calc_Glatt(vbeta,matsubara_freq,self_ene_in,vmu)
    G_latt_tau = fourie_transformer.G_freq_to_tau(G_latt,ndiv_tau,vbeta,cutoff_fourie)
    ntot = 0.0
    for ie in range(nflavor):
        ntot += -G_latt_tau[-1,ie,ie].real
        print ie, -G_latt_tau[-1,ie,ie].real
    print "tot_Glatt = ", ntot
    time2 = time.time()
    print "Computing G_latt(tau) took ", time2-time1, " sec."

    print 'G_latt_tau(beta/2)=', np.sum([-G_latt_tau[ndiv_tau/2,iflavor,iflavor].real for iflavor in xrange(nflavor)])
    sys.stdout.flush()
    np.save(app_parms["prefix"]+"-G_latt", G_latt)
    np.save(app_parms["prefix"]+"-G_latt_tau", G_latt_tau)

    for it in range(ndiv_tau+1):
        assert is_hermitian(G_latt_tau[it,:,:])

    #### Cavity Green's function ####
    print "Computing cavity Green's function..."
    time1 = time.time()
    sys.stdout.flush()
    invG0 = np.zeros((nsbl,ndiv_tau,nflavor_sbl,nflavor_sbl),dtype=complex)
    for isbl in xrange(nsbl):
        for im in xrange(ndiv_tau):
            start = nflavor_sbl*isbl
            end = nflavor_sbl*(isbl+1)
            invG0[isbl,im,:,:]=inv(G_latt[im,start:end,start:end])+self_ene_in[im,start:end,start:end]

    time2 = time.time()
    print "Computing cavity Green's function took ", time2-time1, " sec."

    #### Hybridization function ####
    hyb=np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
    print "Computing Delta(omega_n)..."
    time1 = time.time()
    sys.stdout.flush()
    for isbl in xrange(nsbl):
        start = nflavor_sbl*isbl
        end = nflavor_sbl*(isbl+1)
        for im in range(ndiv_tau):
            hyb[im,start:end,start:end] = np.identity(nflavor_sbl)*(1J*matsubara_freq[im]+vmu)-Hk_mean[start:end,start:end]-invG0[isbl,im,:,:]
    time2 = time.time()
    print "Computing Delta(omega_n) took ", time2-time1, " sec."

    print "Transforming Delta(omega_n) to Delta(tau)..."
    time1 = time.time()
    sys.stdout.flush()
    hyb_tau,high_freq_tail,hyb_rest = fourie_transformer.hyb_freq_to_tau(hyb,ndiv_tau,vbeta,cutoff_fourie)
    if 'ASSUME_REAL' in app_parms and app_parms['ASSUME_REAL'] != 0:
        hyb_tau = np.array(hyb_tau.real, dtype=complex)
    time2 = time.time()
    print "Transforming Delta(omega_n) to Delta(tau) took ", time2-time1, " sec."
    np.save(app_parms["prefix"]+"-hyb", hyb)
    np.save(app_parms["prefix"]+"-hyb_tau", hyb_tau)
    np.save(app_parms["prefix"]+"-hyb_tail", high_freq_tail)
    np.save(app_parms["prefix"]+"-hyb_rest", hyb_rest)

    for it in range(ndiv_tau+1):
        assert is_hermitian(hyb_tau[it,:,:])

    sys.stdout.flush()

    #print hyb_rest[-1,:,:]
    #sys.exit(1)

    #### Impurity solver ####
    time3 = time.time()
    if app_parms['IMP_SOLVER']=='CT-HYB' or not app_parms.has_key('IMP_SOLVER'):
        imp_result,obs_meas = call_hyb_matrix(app_parms, imp_model, ft, tau, hyb_tau, hyb, invG0, vmu, local_projectors)
    elif app_parms['IMP_SOLVER']=='CT-INT':
        imp_result,obs_meas = call_ct_int(app_parms, imp_model, ft, tau, invG0, vmu)
    else:
        raise RuntimeError("Not implemented")

    G_imp = imp_result['G_imp']
    g_tau_im = imp_result['Greens_imag_tau']
    self_ene_out = imp_result['self_ene']
    time4 = time.time()
    print "Solving impurity problem took ", time4-time3, " sec."
    print "orbital_occ = ", imp_result["n_rotated"]
    print "tot_elec = ", np.sum(imp_result["n_rotated"]).real, imp_result["n_rotated"].real
    print 'G_tau(beta/2) = ', np.sum([-np.sum(g_tau_im[ndiv_tau/2-10:ndiv_tau/2+10,iflavor,iflavor].real) for iflavor in xrange(nflavor/nsbl)])
    sys.stdout.flush()

    np.save(app_parms["prefix"]+"-G_tau",g_tau_im)
    np.save(app_parms["prefix"]+"-G_imp",G_imp)
    np.save(app_parms["prefix"]+"-self_ene",self_ene_out)

    time1 = time.time()

    F = self_ene_out-self_ene_in
    for iflavor in range(nflavor):
        for iflavor2 in range(nflavor):
            F[:,iflavor,iflavor2] = F[:,iflavor,iflavor2]*dmp_fact

    print "max_diff", (np.absolute(F)).max()
    max_Gdiff = 0.0
    if 'MULTI_IMP' in app_parms and app_parms['MULTI_IMP'] != 0:
        for isbl in xrange(nsbl):
            start = isbl*nflavor_sbl
            end = (isbl+1)*nflavor_sbl
            max_Gdiff +=  (np.absolute(G_latt[:,start:end,start:end]-G_imp[:,start:end,start:end])).max()
    else:
        max_Gdiff =  (np.absolute(G_latt[:,0:nflavor/nsbl,0:nflavor/nsbl]-G_imp)).max()
    print "max_Gdiff", max_Gdiff

    if 'TOLERANCE_G' in app_parms and max_Gdiff<app_parms['TOLERANCE_G']:
        F *= 0.0
    if app_parms["MAX_IT"]==isc+1:
        F *= 0.0

    #update chemical potential
    if ('OPT_MU' in app_parms and app_parms['OPT_MU'] < 0) and isc >= app_parms['FIX_CHEM']:
        tote_imp = np.sum(imp_result["n_rotated"])*(nflavor_sbl/float(imp_result["n_rotated"].shape[0]))
        print "n_rotated", imp_result["n_rotated"]
        print "tote_imp = ", tote_imp
        vmu = vmu-np.abs(app_parms['OPT_MU'])*(tote_imp*nsbl-2*app_parms["N_ELEC"])
        print "new mu = ", vmu

    #Update results
    dmft_result.update(imp_model.get_moment(1), imp_model.get_moment(2), imp_model.get_moment(3), vmu, G_latt, G_imp, g_tau_im, self_ene_out, hyb, hyb_tau)
    for key in obs_meas.keys():
        dmft_result[key] = obs_meas[key]
    dump_results_modmft(app_parms, isc, dmft_result, exclude_list)

    time2 = time.time()
    print "Rest part took ", time2-time1, " sec."
    print "One sc loop took ", time2-time_start, " sec."

    sys.stdout.flush()
    isc += 1
    return F

self_ene_init=np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
if 'SIGMA_INPUT' in app_parms:
    print 'Reading ', app_parms['SIGMA_INPUT'], '...'
    self_ene_init=np.load(app_parms["SIGMA_INPUT"])

dmp_fact = matsubara_freq**(-2)
self_ene_dmp_init = np.array(self_ene_init)
for iflavor in range(nflavor):
    for iflavor2 in range(nflavor):
        self_ene_dmp_init[:,iflavor,iflavor2] = self_ene_dmp_init[:,iflavor,iflavor2]*dmp_fact

#symmetry
if 'SYMM_MAT' in app_parms:
    nsymm = app_parms['SYMM_MAT'].shape[0]
    nf_sbl = imp_model.get_nflavor()/imp_model.get_nsbl()
    for isymm in xrange(nsymm):
        assert is_unitary(app_parms['SYMM_MAT'][isymm,:,:])
        assert commute(imp_model.get_moment(1)[0:nf_sbl,0:nf_sbl], app_parms['SYMM_MAT'][isymm,:,:])

#sciopt.root(calc_diff,self_ene_init,method="anderson",options={'nit' : app_parms["MAX_IT"], 'fatol' : app_parms["CONVERGED"], 'disp': True, 'M': 10})
dmft_result = DMFTResult()
mix = 0.5
if 'mix' in app_parms:
    mix = app_parms['mix']
sciopt.linearmixing(calc_diff,self_ene_dmp_init,alpha=mix,iter=app_parms["MAX_IT"],f_rtol=app_parms["CONVERGED"],line_search=None,verbose=True)
#sciopt.anderson(calc_diff,self_ene_dmp_init,iter=app_parms["MAX_IT"], f_rtol=app_parms["CONVERGED"], verbose=True)
