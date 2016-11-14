import numpy as np
import h5py
from lib import *

def mpow(mat, n):
    assert n>=1
    if n==1:
        return mat
    else:
        return np.dot(mat,mpow(mat,n-1))

class OrbitalModel:
    def __init__(self, h5f):
        self.Hk_list_ = h5f['/MODEL/Hk'].value
        self.weight_k_ = h5f['/MODEL/Hk_weight'].value
        self.weight_k_ /= np.sum(self.weight_k_)
        self.H_loc_ = h5f['/MODEL/H_loc'].value
        self.norb_ = self.Hk_list_.shape[1]/2
        self.nflavor_ = self.Hk_list_.shape[1]
        self.Nk_ = len(self.weight_k_)
        self.nsbl_ = int(h5f['/MODEL/N_sublattice'].value)
        norb_sbl = self.norb_/self.nsbl_

        #Deprecated (for CT-HYB)
        if '/MODEL/U_orb' in h5f:
            self.U_orb_ = h5f['/MODEL/U_orb'].value
            self.Up_bond_ = h5f['/MODEL/Up_bond'].value
            self.J_bond_ = h5f['/MODEL/J_bond'].value

            assert len(self.Up_bond_)==norb_sbl*(norb_sbl-1)/2
            assert len(self.J_bond_)==norb_sbl*(norb_sbl-1)/2
            assert self.Hk_list_.shape[1]==self.Hk_list_.shape[2]
            assert len(self.weight_k_)==self.Hk_list_.shape[0]
        #Newer version
        elif '/MODEL/Uijkl' in h5f:
            self.Uijkl_ = h5f['/MODEL/Uijkl'].value
            assert len(self.Uijkl_.shape)==4
            assert self.Uijkl_.shape[0]==2*norb_sbl
            assert self.Uijkl_.shape[1]==2*norb_sbl
            assert self.Uijkl_.shape[2]==2*norb_sbl
            assert self.Uijkl_.shape[3]==2*norb_sbl
        else:
            raise RuntimeError("Please set a U tensor!")

        #Compute moments
        self.M1_ = np.zeros((self.nflavor_,self.nflavor_,),dtype=complex)
        self.M2_ = np.zeros((self.nflavor_,self.nflavor_,),dtype=complex)
        self.M3_ = np.zeros((self.nflavor_,self.nflavor_,),dtype=complex)
        self.M4_ = np.zeros((self.nflavor_,self.nflavor_,),dtype=complex)
        for ik in range(self.Nk_):
            Hk = 1.*self.Hk_list_[ik,:,:]
            if not is_hermitian(Hk):
                print "H(k) is not a Hermitian matrix ik=", ik
            self.M1_ += self.weight_k_[ik]*Hk
            self.M2_ += self.weight_k_[ik]*np.dot(Hk,Hk)
            self.M3_ += self.weight_k_[ik]*np.dot(Hk,np.dot(Hk,Hk))
            self.M4_ += self.weight_k_[ik]*mpow(Hk,4)
        sum_w = np.sum(self.weight_k_)
        self.M1_ /= sum_w
        self.M2_ /= sum_w
        self.M3_ /= sum_w
        self.M4_ /= sum_w

        assert is_hermitian(self.M1_)
        assert is_hermitian(self.M2_)
        assert is_hermitian(self.M3_)
        assert is_hermitian(self.M4_)

        print "Max |H_loc|=", np.amax(np.abs(self.H_loc_))
        print "Max |H_loc-<H_k>|=", np.amax(np.abs(self.H_loc_-self.M1_))
        assert np.amax(np.abs(self.H_loc_-self.M1_))<1e-8

        self.print_ek_info()

    #Compute Lattice Green's function
    def calc_Glatt(self,beta,matsubara_freq,self_ene,vmu):
        ndiv_tau = len(matsubara_freq)
        nflavor = self.nflavor_
        G_latt=np.zeros((ndiv_tau,nflavor,nflavor),dtype=complex)
        Nk_reduced = self.Nk_
        for im in range(ndiv_tau):
            G_latt[im,:,:] = 0.0+0.0J
            digmat = (1J*matsubara_freq[im]+vmu)*np.identity(nflavor)
            ksum = 0
            for ik in range(Nk_reduced):
                G_latt[im,:,:] += np.linalg.inv(digmat-self.Hk_list_[ik,:,:]-self_ene[im,:,:])*self.weight_k_[ik]
            G_latt[im,:,:] /= np.sum(self.weight_k_)

        tau_tmp = beta-1E-4*(beta/ndiv_tau)
        ztmp = 0.0
        for im in range(ndiv_tau):
            for iorb in range(nflavor):
                ztmp += (G_latt[im,iorb,iorb]+1J/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
        rtmp = 2.0*ztmp.real/beta-0.5*nflavor
    
        return G_latt,-rtmp

    def calc_Gk_omega0(self,beta,self_ene,vmu,matsubara_freq,Hk_list):
        nflavor = self.nflavor_
        Nk = Hk_list.shape[0]
        Gk=np.zeros((Nk,),dtype=complex)
        Ek=np.zeros((Nk,nflavor),dtype=float)

        #matsubara_freq=np.pi/beta
        digmat = (1J*matsubara_freq+vmu)*np.identity(nflavor)
        for ik in range(Nk):
            evals,evecs = eigh_ordered(Hk_list[ik,:,:]+self_ene[0,:,:]-vmu*np.identity(nflavor))
            Ek[ik,:] = 1.*evals
            Gk[ik] = np.trace(np.linalg.inv(digmat-Hk_list[ik,:,:]-self_ene[0,:,:]))

        return Gk,Ek

    def print_ek_info(self):
        ek = np.zeros((self.Nk_,self.nflavor_),dtype=float)
        for ik in range(self.Nk_):
            evals,evecs = eigh_ordered(self.Hk_list_[ik,:,:])
            ek[ik,:] = evals
        print "ek_mean=", np.mean(ek,0)
        print "max ek=", np.amax(ek)
        print "min ek=", np.amin(ek)

    def get_norb(self):
        return self.norb_

    def get_nsbl(self):
        return self.nsbl_

    def get_nflavor(self):
        return self.nflavor_

    def get_moment(self, n):
        assert n>=1 and n<=4
        if n==1:
            return 1.*self.M1_
        elif n==2:
            return 1.*self.M2_
        elif n==3:
            return 1.*self.M3_
        elif n==4:
            return 1.*self.M4_
        else:
            raise RuntimeError("n is invalid")

    def get_H0(self):
        return self.H_loc_

    # -t c_j^\dagger c_i (we have a minus sign)
    # i,ispin : source orbital/spin
    # j,jspin : target orbital/spin
    def get_H0_trans(self, i, j, ispin, jspin):
        iflavor = 2*i+ispin
        jflavor = 2*j+jspin
        return -self.H_loc_[jflavor,iflavor]

    def get_E_flavor(self, iflavor):
        iorb = iflavor/2
        ispin = iflavor%2
        return -self.get_H0_trans(iorb, iorb, ispin, ispin)

    #Deprecated
    def get_U_orb(self, iorb):
        assert iorb>=0 and iorb<self.norb_
        return self.U_orb_[iorb]

    #Deprecated
    def get_Up_bond(self, ibond):
        assert ibond>=0 and ibond<self.norb_*(self.norb_-1)/2
        return self.Up_bond_[ibond]

    #Deprecated
    def get_J_bond(self, ibond):
        assert ibond>=0 and ibond<self.norb_*(self.norb_-1)/2
        return self.J_bond_[ibond]

    #New version
    def get_Uijkl(self):
        return self.Uijkl_
