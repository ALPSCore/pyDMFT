#import pyalps 
import numpy as np
from numpy.linalg import inv
import sys, copy, random
import time
import integration as intgr
import re 
import input_parms as inp
import os
import subprocess,shlex
import h5py
import fourier_transform as ft
import tempfile
import Uijkl
from collections import OrderedDict
from lib import *
from h5dump import *
from fourier_transform import ft_to_tau_hyb

def to_spin_full_G(G):
    G2 = G.transpose((0,3,1,2))
    ntau_p1 = G2.shape[1]
    norb = G2.shape[2]
    G_spinfull = np.zeros((ntau_p1,norb,2,norb,2),dtype=complex)
    G_spinfull[:,:,0,:,0] = G2[0,:,:,:] #up spin
    G_spinfull[:,:,1,:,1] = G2[1,:,:,:] #down spin
    return G_spinfull.reshape((ntau_p1,2*norb,2*norb))
    
def read_G_nspin2(hf):
    sign = hf['/simulation/results/Sign/mean/value'].value

    G_omega_l = hf['/G_omega_legendre/values/mean'].value
    nsite = hf['/G_omega_legendre/ns'].value
    nf = hf['/G_omega_legendre/nf'].value
    nomega = G_omega_l.shape[0]/(nsite*nsite*nf)
    G_omega_l = G_omega_l.reshape((nf,nsite,nsite,nomega,2))
    G_omega_l2 = G_omega_l[:,:,:,:,0]+1J*G_omega_l[:,:,:,:,1]

    nsite = hf['/G_tau_legendre/ns'].value
    nf = hf['/G_tau_legendre/nf'].value
    ntau = hf['/G_tau_legendre/nt'].value-1
    G_tau_l = hf['/G_tau_legendre/values/mean'].value
    G_tau_l = G_tau_l.reshape((nf,nsite,nsite,ntau+1,2))
    G_tau_l = G_tau_l[:,:,:,:,0]+1J*G_tau_l[:,:,:,:,1]

    return to_spin_full_G(G_omega_l2), to_spin_full_G(G_tau_l)

def read_G_spin_orbit(hf):
    sign = hf['/simulation/results/Sign/mean/value'].value

    G_omega_l = hf['/G_omega_legendre/values/mean'].value
    nsite = hf['/G_omega_legendre/ns'].value
    nf = hf['/G_omega_legendre/nf'].value
    assert nf==1
    nomega = G_omega_l.shape[0]/(nsite*nsite*nf)
    G_omega_l = G_omega_l.reshape((nsite,nsite,nomega,2))
    G_omega_l2 = G_omega_l[:,:,:,0]+1J*G_omega_l[:,:,:,1]

    nsite = hf['/G_tau_legendre/ns'].value
    nf = hf['/G_tau_legendre/nf'].value
    assert nf==1
    ntau = hf['/G_tau_legendre/nt'].value-1
    G_tau_l = hf['/G_tau_legendre/values/mean'].value
    G_tau_l = G_tau_l.reshape((nsite,nsite,ntau+1,2))
    G_tau_l = G_tau_l[:,:,:,0]+1J*G_tau_l[:,:,:,1]

    return G_omega_l2.transpose((2,1,0)), G_tau_l.transpose((2,1,0))

def load_sign(hf):
    return hf['/simulation/results/Sign/mean/value'].value

def load_obs_with_sign(hf,obs):
    sign = hf['/simulation/results/Sign/mean/value'].value
    return (hf['/simulation/results/'+obs+'_Re/mean/value'].value+1J*hf['/simulation/results/'+obs+'_Im/mean/value'].value)/sign

def load_real_obs_with_sign(hf,obs):
    sign = hf['/simulation/results/Sign/mean/value'].value
    return hf['/simulation/results/'+obs+'/mean/value'].value/sign

#Return U**alpha
def unitary_mat_power(Umat, alpha):
    assert Umat.shape[0]==Umat.shape[1]
    N = Umat.shape[0]
    evals,Vmat = np.linalg.eig(Umat)
    Gamma = np.zeros((N,N),dtype=complex)
    for i in xrange(N):
        Gamma[i,i] = evals[i]**alpha
    return np.dot(np.dot(Vmat,Gamma),Vmat.conjugate().transpose())

#def projection(self_ene,evecs,norb):
    #self_ene_eigen = np.zeros_like(self_ene)
    #for ib in range(norb):
        #for ib2 in range(norb):
            #for iorb in range(norb):
                #for iorb2 in range(norb):
                    #self_ene_eigen[:,ib,ib2] += self_ene[:,iorb,iorb2]*np.conj(evecs[iorb,ib])*evecs[iorb2,ib2]
    #return self_ene_eigen

def symmetrize_G_tau(app_parms, G_tau):
    ntau = G_tau.shape[0]-1 
    nflavor_sbl = G_tau.shape[1]
    assert G_tau.shape[1]==G_tau.shape[2]

    G_tau_new = np.zeros_like(G_tau)
    if 'SYMM_MAT' in app_parms:
        nsymm = app_parms['SYMM_MAT'].shape[0]
        assert app_parms['SYMM_MAT'].shape[1]==nflavor_sbl
        assert app_parms['SYMM_MAT'].shape[2]==nflavor_sbl
        print "Symmetrizing G_tau..."

        G_tau_symm = np.zeros((nsymm+1,ntau+1,nflavor_sbl,nflavor_sbl),dtype=complex)

        G_tau_symm[0,:,:,:] = 1.0*G_tau
        for isymm in xrange(nsymm):
            G_tau_symm[isymm+1,:,:,:] = projection(G_tau[:,:,:], app_parms['SYMM_MAT'][isymm,:,:],nflavor_sbl)
        G_tau_new[:,:,:] = np.average(G_tau_symm, axis=0)
    else:
        G_tau_new[:,:,:] = 1.*G_tau

    if 'PM' in app_parms and app_parms['PM'] != 0:
            print "Making G_tau paramagnetic..."
            for iorb in range(nflavor_sbl/2):
                #mz=0
                G_tau_new[:,2*iorb,2*iorb] = 0.5*(G_tau_new[:,2*iorb,2*iorb]+G_tau_new[:,2*iorb+1,2*iorb+1])
                G_tau_new[:,2*iorb+1,2*iorb+1] = 1.0*G_tau_new[:,2*iorb,2*iorb]
                #mx=0 and my=0
                G_tau_new[:,2*iorb,2*iorb+1] = 0.0
                G_tau_new[:,2*iorb+1,2*iorb] = 0.0
    return G_tau_new

def solve_sbl_imp_model_spin_orbit(app_parms, imp_model, fourie_transformer, tau_mesh, invG0_omega, mu, isbl):
    time1 = time.time()

    ntau = len(tau_mesh)-1
    norb = imp_model.get_norb()
    nsbl = imp_model.get_nsbl()
    nflavor = imp_model.get_nflavor()
    nflavor_sbl = nflavor/nsbl
    norb_sbl = norb/nsbl
    beta = app_parms['BETA']

    start = isbl*nflavor_sbl
    end = (isbl+1)*nflavor_sbl

    #### impurity solver ####
    path_input = app_parms['prefix']+'_input_ct_int_sbl'+str(isbl)

    #Generate input files...
    input_f = open(path_input,'w')
    print >>input_f, "{"
    parms=OrderedDict()
    parms['N_TAU'] = app_parms['NMATSUBARA']
    parms['N_MATSUBARA'] = app_parms['NMATSUBARA']
    parms['BETA'] = app_parms['BETA']
    parms['SITES'] = nflavor_sbl
    parms['FLAVORS'] = 1

    #Write U tensor
    if 'ROTMAT_Uijkl' in app_parms:
        assert len(app_parms['ROTMAT_Uijkl'].shape)==2
        assert app_parms['ROTMAT_Uijkl'].shape[0]==app_parms['ROTMAT_Uijkl'].shape[1]==nflavor_sbl
        rotmat = app_parms['ROTMAT_Uijkl']
    else:
        rotmat = np.identity(2*norb_sbl)
    parms['GENERAL_U_MATRIX_FILE'] = path_input+'-Uijkl.txt'
    H0_corr = Uijkl.write_Uijkl(0.5*imp_model.get_Uijkl(), rotmat, parms['GENERAL_U_MATRIX_FILE'], False).reshape(2*norb_sbl,2*norb_sbl)

    #Incorpolate the correction term into the cavity function (coming from auxiliary fields)
    invG0_omega_symm = symmetrize_G_tau(app_parms, invG0_omega)
    G0_corr = np.zeros_like(invG0_omega)
    for im in xrange(ntau):
        G0_corr[im,:,:] = inv(invG0_omega_symm[im,:,:]-H0_corr)
    G0_rot = projection(symmetrize_G_tau(app_parms, G0_corr), rotmat, nflavor_sbl)
    
    #FFT to G0(tau)
    G0_tau_rot = np.zeros((ntau+1,2*norb_sbl,2*norb_sbl),dtype=complex)
    c1 = np.diag([1.0]*nflavor_sbl)
    c2 = np.zeros((nflavor_sbl,nflavor_sbl),dtype=complex)
    c3 = np.zeros((nflavor_sbl,nflavor_sbl),dtype=complex)
    matsubara_freq = np.array([((2*im+1)*np.pi)/beta for im in xrange(ntau)])
    ft_to_tau_hyb(ntau, beta, matsubara_freq, tau_mesh, c1, c2, c3, G0_rot, G0_tau_rot, app_parms["CUTOFF_FOURIE"])

    #Write G0
    def cut_small_value(v, eps=1e-10):
        if np.abs(v)>eps:
            return v
        else:
            return 0.0

    if 'ASSUME_REAL' in app_parms and app_parms['ASSUME_REAL'] != 0:
        raise RuntimeError("ASSUME_REAL is not supported")
    else:
        f = open(path_input+'-G0_TAU.txt','w')
        for iflavor in xrange(nflavor_sbl):
            for jflavor in xrange(nflavor_sbl):
                for itau in xrange(ntau+1):
                    print>>f, 0, iflavor, jflavor, itau, cut_small_value(G0_tau_rot[itau,iflavor,jflavor].real), cut_small_value(G0_tau_rot[itau,iflavor,jflavor].imag)
        f.close()
    
    f = open(path_input+'-G0_OMEGA.txt','w')
    for iflavor in xrange(nflavor_sbl):
        for jflavor in xrange(nflavor_sbl):
            for im in xrange(ntau):
                print>>f, 0, iflavor, jflavor, im, cut_small_value(G0_rot[im,iflavor,jflavor].real), cut_small_value(G0_rot[im,iflavor,jflavor].imag)
    f.close()
    parms['G0_OMEGA'] = path_input+'-G0_OMEGA.txt'
    parms['G0_TAU'] = path_input+'-G0_TAU.txt'

    #Set parameters
    for k,v in app_parms.items():
        m = re.search('^IMP_SLV_(.+)$',k)
        if m!=None:
            print k,v,m.group(0),m.group(1)
            parms[m.group(1)] = v

    #Set random seed
    random.seed()
    parms['SEED'] = random.randint(0,10000)

    #Load/dump config
    if 'LOAD_CONFIG_CT_INT' in app_parms and app_parms['LOAD_CONFIG_CT_INT']:
        parms['PREFIX_LOAD_CONFIG'] = path_input
    if 'DUMP_CONFIG_CT_INT' in app_parms and app_parms['DUMP_CONFIG_CT_INT']:
        parms['PREFIX_DUMP_CONFIG'] = path_input

    #Write parameters
    write_parms(input_f, parms)
    print >>input_f, "}"
    input_f.close()

    if (os.path.exists(path_input+'.task1.in.h5')):
      os.remove(path_input+'.task1.in.h5')
    cmd='parameter2hdf5 -f '+path_input
    print cmd
    os.system(cmd)
    output_f = open('output_'+path_input, 'w')
    cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' --mpi '+path_input+'.task1.in.h5'
    print cmd
    time2 = time.time()
    args = shlex.split(cmd)
    subprocess.call(args, stdout=output_f, stderr=output_f) # Success!
    output_f.close()
    print "Finished CT-INT program"
    time3 = time.time()

    #Load measured observables
    result = {}
    foutput=path_input+'.task1.out.h5'

    print "Opening ", foutput, "..."
    hf = h5py.File('./'+foutput, 'r')

    #<Sign>
    sign = load_sign(hf)
    print "sign=", complex(sign)
    print "abs(sign)=", np.abs(sign)
    
    #Im G(tau)
    G_omega, G_tau = read_G_spin_orbit(hf)

    #<n_i> in the rotated basis
    result["n_rotated"] = np.array([-G_tau[-1,iflavor,iflavor].real for iflavor in xrange(nflavor_sbl)],dtype=float)

    #transform G back to the original basis
    G_omega = projection(G_omega, rotmat.conjugate().transpose(), nflavor_sbl)
    G_tau = projection(G_tau, rotmat.conjugate().transpose(), nflavor_sbl)

    hf.close()

    #Symmetrize Green's function (This operation is linear)
    G_tau = symmetrize_G_tau(app_parms, G_tau)
    G_omega = symmetrize_G_tau(app_parms, G_omega)
    result["Greens_imag_tau"] = G_tau
    result["G_imp"] = G_omega

    #Load all observables
    keys,means,errors = load_observables("./"+foutput)
    obs = {}
    for i in range(len(keys)):
        obs[keys[i]+'_mean'] = means[i]
        obs[keys[i]+'_error'] = errors[i]

    self_ene_sbl = np.zeros((ntau,nflavor_sbl,nflavor_sbl),dtype=complex)
    for im in range(ntau):
        self_ene_sbl[im,:,:]=invG0_omega_symm[im,:,:]-inv(G_omega[im,:,:])
    result["self_ene"] = self_ene_sbl

    time4 = time.time()

    print "Timings of solving an impurity model tot=", time4-time1, " : ", time2-time1, " ", time3-time2, " ", time4-time3

    return result, obs

#hyb_tau: Delta(\tau), 
# Note: when we convert Delta to F, we have to exchange flavor indices in Delta and rotmat.
def solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, invG0_omega, mu, isbl):
    time1 = time.time()

    ntau = len(tau_mesh)-1
    norb = imp_model.get_norb()
    nsbl = imp_model.get_nsbl()
    nflavor = imp_model.get_nflavor()
    nflavor_sbl = nflavor/nsbl
    norb_sbl = norb/nsbl
    beta = app_parms['BETA']

    start = isbl*nflavor_sbl
    end = (isbl+1)*nflavor_sbl

    #### impurity solver ####
    path_input = app_parms['prefix']+'_input_ct_int_sbl'+str(isbl)

    #Generate input files...
    input_f = open(path_input,'w')
    print >>input_f, "{"
    parms=OrderedDict()
    parms['N_TAU'] = app_parms['NMATSUBARA']
    parms['N_MATSUBARA'] = app_parms['NMATSUBARA']
    parms['BETA'] = app_parms['BETA']
    parms['SITES'] = norb_sbl
    parms['FLAVORS'] = 2

    #Write U tensor
    if 'ROTMAT_Uijkl' in app_parms:
        assert len(app_parms['ROTMAT_Uijkl'].shape)==2
        assert app_parms['ROTMAT_Uijkl'].shape[0]==app_parms['ROTMAT_Uijkl'].shape[1]==nflavor_sbl
        rotmat = app_parms['ROTMAT_Uijkl']
        assert np.sum(np.abs(rotmat.reshape((norb_sbl,2,norb_sbl,2))[:,0,:,1]))<1E-10 #should be diagonal in spin space
        assert np.sum(np.abs(rotmat.reshape((norb_sbl,2,norb_sbl,2))[:,1,:,0]))<1E-10 #should be diagonal in spin space
    else:
        rotmat = np.identity(2*norb_sbl)
    parms['GENERAL_U_MATRIX_FILE'] = path_input+'-Uijkl.txt'
    H0_corr = Uijkl.write_Uijkl(0.5*imp_model.get_Uijkl(), rotmat, parms['GENERAL_U_MATRIX_FILE'], True).reshape(2*norb_sbl,2*norb_sbl)

    #Incorpolate the correction term into the cavity function (coming from auxiliary fields)
    invG0_omega_symm = symmetrize_G_tau(app_parms, invG0_omega)
    G0_corr = np.zeros_like(invG0_omega)
    for im in xrange(ntau):
        G0_corr[im,:,:] = inv(invG0_omega_symm[im,:,:]-H0_corr)
    #G0_rot = projection(G0_corr, rotmat, nflavor_sbl)
    G0_rot = projection(symmetrize_G_tau(app_parms, G0_corr), rotmat, nflavor_sbl)
    
    #FFT to G0(tau)
    G0_tau_rot = np.zeros((ntau+1,2*norb_sbl,2*norb_sbl),dtype=complex)
    c1 = np.diag([1.0]*nflavor_sbl)
    c2 = np.zeros((nflavor_sbl,nflavor_sbl),dtype=complex)
    c3 = np.zeros((nflavor_sbl,nflavor_sbl),dtype=complex)
    matsubara_freq = np.array([((2*im+1)*np.pi)/beta for im in xrange(ntau)])
    ft_to_tau_hyb(ntau, beta, matsubara_freq, tau_mesh, c1, c2, c3, G0_rot, G0_tau_rot, app_parms["CUTOFF_FOURIE"])
    G0_tau_rot = G0_tau_rot.reshape((ntau+1,norb_sbl,2,norb_sbl,2))
    G0_rot = G0_rot.reshape((ntau,norb_sbl,2,norb_sbl,2))
    assert np.sum(np.abs(G0_tau_rot[:,:,0,:,1]))<1E-10
    assert np.sum(np.abs(G0_rot[:,:,0,:,1]))<1E-10

    #Write G0
    def cut_small_value(v, eps=1e-10):
        if np.abs(v)>eps:
            return v
        else:
            return 0.0

    if 'ASSUME_REAL' in app_parms and app_parms['ASSUME_REAL'] != 0:
        f = open(path_input+'-G0_TAU.txt','w')
        for iflavor in xrange(2):
            for iorb in xrange(norb_sbl):
                for jorb in xrange(norb_sbl):
                    for itau in xrange(ntau+1):
                        print>>f, iflavor, iorb, jorb, itau, cut_small_value(G0_tau_rot[itau,iorb,iflavor,jorb,iflavor].real), 0.0
        f.close()
    else:
        f = open(path_input+'-G0_TAU.txt','w')
        for iflavor in xrange(2):
            for iorb in xrange(norb_sbl):
                for jorb in xrange(norb_sbl):
                    for itau in xrange(ntau+1):
                        print>>f, iflavor, iorb, jorb, itau, cut_small_value(G0_tau_rot[itau,iorb,iflavor,jorb,iflavor].real), cut_small_value(G0_tau_rot[itau,iorb,iflavor,jorb,iflavor].imag)
        f.close()
    
    f = open(path_input+'-G0_OMEGA.txt','w')
    for iflavor in xrange(2):
        for iorb in xrange(norb_sbl):
            for jorb in xrange(norb_sbl):
                for im in xrange(ntau):
                    print>>f, iflavor, iorb, jorb, im, cut_small_value(G0_rot[im,iorb,iflavor,jorb,iflavor].real), cut_small_value(G0_rot[im,iorb,iflavor,jorb,iflavor].imag)
    f.close()
    parms['G0_OMEGA'] = path_input+'-G0_OMEGA.txt'
    parms['G0_TAU'] = path_input+'-G0_TAU.txt'

    #Set parameters
    for k,v in app_parms.items():
        m = re.search('^IMP_SLV_(.+)$',k)
        if m!=None:
            print k,v,m.group(0),m.group(1)
            parms[m.group(1)] = v

    #Set random seed
    random.seed()
    parms['SEED'] = random.randint(0,10000)

    #Load/dump config
    if app_parms['LOAD_CONFIG_CT_INT']:
        parms['PREFIX_LOAD_CONFIG'] = path_input
    if app_parms['DUMP_CONFIG_CT_INT']:
        parms['PREFIX_DUMP_CONFIG'] = path_input

    #Write parameters
    write_parms(input_f, parms)
    print >>input_f, "}"
    input_f.close()

    if (os.path.exists(path_input+'.task1.in.h5')):
      os.remove(path_input+'.task1.in.h5')
    cmd='parameter2hdf5 -f '+path_input
    print cmd
    os.system(cmd)
    output_f = open('output_'+path_input, 'w')
    cmd=app_parms['CMD_MPI']+' '+str(app_parms['N_MPI_PROCESS'])+' '+str(app_parms['HYB_PATH'])+' --mpi '+path_input+'.task1.in.h5'
    print cmd
    time2 = time.time()
    args = shlex.split(cmd)
    subprocess.call(args, stdout=output_f, stderr=output_f) # Success!
    output_f.close()
    print "Finished CT-INT program"
    time3 = time.time()

    #Load measured observables
    result = {}
    foutput=path_input+'.task1.out.h5'

    print "Opening ", foutput, "..."
    hf = h5py.File('./'+foutput, 'r')

    #<Sign>
    sign = load_sign(hf)
    print "sign=", complex(sign)
    print "abs(sign)=", np.abs(sign)
    
    #Im G(tau)
    G_omega, G_tau = read_G_nspin2(hf)
    if 'ASSUME_REAL' in app_parms and app_parms['ASSUME_REAL'] != 0:
        G_tau = np.array(G_tau.real,dtype=complex)

    #<n_i> in the rotated basis
    result["n_rotated"] = np.array([-G_tau[-1,iflavor,iflavor].real for iflavor in xrange(nflavor_sbl)],dtype=float)

    #transform G back to the original basis
    G_omega = projection(G_omega, rotmat.conjugate().transpose(), nflavor_sbl)
    G_tau = projection(G_tau, rotmat.conjugate().transpose(), nflavor_sbl)

    hf.close()

    #Symmetrize Green's function (This operation is linear)
    G_tau = symmetrize_G_tau(app_parms, G_tau)
    G_omega = symmetrize_G_tau(app_parms, G_omega)
    result["Greens_imag_tau"] = G_tau
    result["G_imp"] = G_omega

    #Load all observables
    keys,means,errors = load_observables("./"+foutput)
    obs = {}
    for i in range(len(keys)):
        obs[keys[i]+'_mean'] = means[i]
        obs[keys[i]+'_error'] = errors[i]

    self_ene_sbl = np.zeros((ntau,nflavor_sbl,nflavor_sbl),dtype=complex)
    for im in range(ntau):
        self_ene_sbl[im,:,:]=invG0_omega_symm[im,:,:]-inv(G_omega[im,:,:])
    result["self_ene"] = self_ene_sbl

    time4 = time.time()

    print "Timings of solving an impurity model tot=", time4-time1, " : ", time2-time1, " ", time3-time2, " ", time4-time3

    return result, obs

#hyb_tau: Delta(\tau), 
# Note: when we convert Delta to F, we have to exchange flavor indices in Delta and rotmat.
def call_ct_int(app_parms, imp_model, fourie_transformer, tau_mesh, invG0, mu):
    ntau = len(tau_mesh)-1
    norb = imp_model.get_norb()
    nsbl = imp_model.get_nsbl()
    nflavor = imp_model.get_nflavor()
    nflavor_sbl = nflavor/nsbl
    norb_sbl = norb/nsbl
    beta = app_parms['BETA']
    cutoff_fourie=app_parms["CUTOFF_FOURIE"]

    matsubara_freq = np.array([((2*im+1)*np.pi)/beta for im in xrange(ntau)])

    single_imp = (not ('MULTI_IMP' in app_parms and app_parms['MULTI_IMP'] != 0))

    if single_imp:
        if app_parms['SPIN_DIAGONAL']!=0:
            result,obs = solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, invG0[0,:,:,:], mu, 0)
        elif app_parms['SPIN_DIAGONAL']==0:
            result,obs = solve_sbl_imp_model_spin_orbit(app_parms, imp_model, fourie_transformer, tau_mesh, invG0[0,:,:,:], mu, 0)
        #Copy sublattice self-energy to unit-cell self-energy 
        self_ene = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        for isbl in range(nsbl):
            start = isbl*nflavor_sbl
            end = (isbl+1)*nflavor_sbl
            self_ene[:,isbl*nflavor_sbl:(isbl+1)*nflavor_sbl, isbl*nflavor_sbl:(isbl+1)*nflavor_sbl] = 1.*result['self_ene'][:,:,:]
        result["self_ene"] = self_ene
        return result, obs
    else:
        #### solving an impurity problem for each site ####
        results_sbl = []
        obs_sbl = []
        for isbl in xrange(nsbl):
            start = isbl*nflavor_sbl
            end = (isbl+1)*nflavor_sbl
            if app_parms['SPIN_DIAGONAL']!=0:
                r,o = solve_sbl_imp_model(app_parms, imp_model, fourie_transformer, tau_mesh, invG0[0,:,:,:], mu, isbl)
            elif app_parms['SPIN_DIAGONAL']==0:
                r,o = solve_sbl_imp_model_spin_orbit(app_parms, imp_model, fourie_transformer, tau_mesh, invG0[0,:,:,:], mu, isbl)
            results_sbl.append(r)
            obs_sbl.append(o)

        result = {}
        obs = {}
    
        #Compute G(tau) and self-energy
        result["n_rotated"] = np.zeros((nflavor,),dtype=float)
        result["Greens_imag_tau"] = np.zeros((ntau+1,nflavor,nflavor),dtype=complex)
        result["G_imp"] = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        result["self_ene"] = np.zeros((ntau,nflavor,nflavor),dtype=complex)
        for isbl in range(nsbl):
            start = isbl*nflavor_sbl
            end = (isbl+1)*nflavor_sbl
    
            #result['n'][start:end] = results_sbl[isbl]['n'][:]
            result['n_rotated'][start:end] = results_sbl[isbl]['n_rotated'][:]
            result["Greens_imag_tau"][:,start:end,start:end] = results_sbl[isbl]['Greens_imag_tau'][:,:,:]
            result["G_imp"][:,start:end,start:end] = results_sbl[isbl]['G_imp'][:,:,:]
            result["self_ene"][:,start:end,start:end] = results_sbl[isbl]['self_ene'][:,:,:]

        #Merge all other data
        for isbl in range(nsbl):
            for k,v in obs_sbl[isbl].items():
                obs[k+"_sbl"+str(isbl)] = v
    
        return result, obs
