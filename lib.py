import numpy as np
from math import pi
import sys
import h5py
import scipy.special
from lib import *

def average_complex(data,axis=0):
    ave = np.average(data,axis=axis)
    std = np.std(data.real,axis=axis)+1J*np.std(data.imag,axis=axis)
    return ave, std

def ToBigMatrix(sm,norb):
    sdim = sm.shape[0]
    bm = np.zeros((sdim*norb,sdim*norb),dtype=complex)
    for i in range(norb):
        offset = sdim*i
        bm[0+offset:sdim+offset,0+offset:sdim+offset] = 1.*sm
    return bm

def projection(self_ene,evecs,norb):
    self_ene_eigen = np.zeros_like(self_ene)
    for ib in range(norb):
        for ib2 in range(norb):
            for iorb in range(norb):
                for iorb2 in range(norb):
                    self_ene_eigen[:,ib,ib2] += self_ene[:,iorb,iorb2]*np.conj(evecs[iorb,ib])*evecs[iorb2,ib2]
    return self_ene_eigen

class NPade:
    def __init__(self,z,u):
        assert len(z)==len(u)
        self.z_ = z
        self.u_ = u
        self.N_ = len(z)

        self.do_fit()

    def do_fit(self):
        self.a_ = np.zeros((self.N_,),dtype=complex)
        self.cache_ = {}

        self.gcache_ = np.zeros((self.N_,self.N_),dtype=complex)
        self.gcache_valid_ = np.zeros((self.N_,self.N_),dtype=int)

        for i in xrange(self.N_):
            self.a_[i] = self.compute_g(i,i)

    def compute_g(self,p,z_i):
        if self.gcache_valid_[p,z_i] == 1:
            return self.gcache_[p,z_i]

        self.gcache_valid_[p,z_i] = 1
        if p==0:
            ztmp = self.u_[z_i]
            self.gcache_[p,z_i] = ztmp
            return ztmp
        else:
            ztmp1 = self.compute_g(p-1,p-1)
            ztmp2 = self.compute_g(p-1,z_i)
            ztmp = (ztmp1-ztmp2)/((self.z_[z_i]-self.z_[p-1])*ztmp2)
            self.gcache_[p,z_i] = ztmp
            return ztmp

    def compute(self, z):
        #assert np.abs(self.compute_A(self.N_-1,z)-self.compute_A_new(self.N_-1,z)) < 1E-5
        #assert np.abs(self.compute_B(self.N_-1,z)-self.compute_B_new(self.N_-1,z)) < 1E-5
        #return self.compute_A(self.N_-1,z)/self.compute_B(self.N_-1,z)
        return self.compute_A_new(self.N_-1,z)/self.compute_B_new(self.N_-1,z)

    def compute_A_new(self, i, z):
        if i==-1:
            return 0.0+0.0J
        elif i==0:
            return self.a_[0]

        #i>1
        A0 = 0.0+0.0J
        A1 = self.a_[0]
        for n in xrange(1,i+1):
            A2 =  A1+(z-self.z_[n-1])*self.a_[n]*A0
            A0 = A1
            A1 = A2
        return A2

    def compute_A(self, i, z):
        if i==-1:
            return 0.0+0.0J
        elif i==0:
            return self.a_[0]
        else:
            return self.compute_A(i-1,z)+(z-self.z_[i-1])*self.a_[i]*self.compute_A(i-2,z)

    def compute_B(self, i, z):
        if i==-1:
            return 1.0+0.0J
        elif i==0:
            return 1.0+0.0J
        else:
            return self.compute_B(i-1,z)+(z-self.z_[i-1])*self.a_[i]*self.compute_B(i-2,z)

    def compute_B_new(self, i, z):
        if i==-1:
            return 1.0+0.0J
        elif i==0:
            return 1.0+0.0J

        B0 = 1.0+0.0J
        B1 = 1.0+0.0J
        for n in xrange(1,i+1):
            B2 =  B1+(z-self.z_[n-1])*self.a_[n]*B0
            B0 = B1
            B1 = B2
        return B2

def gen_xbase():
    evecs = np.zeros((6,6),dtype=complex)
    spinor = np.array([[1,-1J],[1,1J]],dtype=complex)/np.sqrt(2.0)
    elec = np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=complex)
    for iorb in xrange(3):
        for isp in xrange(2):
            for jorb in xrange(3):
                for jsp in xrange(2):
                    evecs[2*jorb+jsp,2*iorb+isp] = spinor[jsp,isp]*elec[jorb,iorb]
    return evecs

def is_hermitian(mat,eps=1e-3):
    assert mat.shape[0]==mat.shape[1]
    n=mat.shape[0]
    maxval = max(np.abs(np.amax(mat)), np.abs(np.amin(mat)))
    for i in xrange(n):
        for j in xrange(n):
            if np.abs(mat[i,j]- mat[j,i].conjugate())>eps*maxval:
                print i,j,mat[i,j], mat[j,i].conjugate()
                return False
    return True

def is_unitary(mat,eps=1e-8):
    assert mat.shape[0]==mat.shape[1]
    n=mat.shape[0]
    diff = np.amin(np.abs(np.dot(mat.conjugate().transpose(), mat)-np.identity(n)))
    return diff<eps

def commute(mat1,mat2,eps=1e-4):
    assert mat1.shape[0]==mat1.shape[1]
    assert mat2.shape[0]==mat2.shape[1]
    assert mat1.shape[0]==mat2.shape[0]
    n=mat1.shape[0]
    maxval1 = max(np.abs(np.amax(mat1)), np.abs(np.amin(mat1)))
    maxval2 = max(np.abs(np.amax(mat2)), np.abs(np.amin(mat2)))
    diff = np.amax(np.abs(np.dot(mat1,mat2)-np.dot(mat2,mat1)))
    #if diff/(maxval1+maxval2)>eps:
    print "error in commute = ", diff/(maxval1+maxval2)
    return diff/(maxval1+maxval2)<eps

def hermitialize(mat):
    assert mat.shape[0]==mat.shape[1]
    n=mat.shape[0]
    mat_r = np.zeros_like(mat)

    for i in xrange(n):
        for j in xrange(n):
            mat_r[i,j] = 0.5*(mat[i,j]+mat[j,i].conjugate())

    return mat_r

def print_mat(f,mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print >> f, mat[i][j],
        print >> f

def mk_map(ndiv_k, ndim):
    nk = ndiv_k**ndim
    r = np.zeros((nk,ndim),dtype=int)
    if ndim==1:
        for ik in range(nk):
            r[ik,0] = ik
    elif ndim==2:
        ik = 0
        for ikx in range(ndiv_k):
            for iky in range(ndiv_k):
                r[ik,0] = ikx
                r[ik,1] = iky
                ik += 1
    elif ndim==3:
        ik = 0
        for ikx in range(ndiv_k):
            for iky in range(ndiv_k):
                for ikz in range(ndiv_k):
                    r[ik,0] = ikx
                    r[ik,1] = iky
                    r[ik,2] = ikz
                    ik += 1
    else:
        raise RuntimeError("Unsupported ndim")
    return r

#u1, d1, u2, d2, etc.
def eigh_ordered_spin(mat):
    N = mat.shape[0]
    M = N/2

    mat_mini = np.array(mat[0:M,0:M])
    for i in xrange(M):
        for j in xrange(M):
            mat_mini[i,j] = mat[2*i, 2*j]

    evals_mini,evecs_mini = eigh_ordered(mat_mini)
    evals = np.zeros((N,),dtype=float)
    evecs = np.zeros_like(mat)
    for ie in xrange(M):
        evals[2*ie] = evals_mini[ie]
        evals[2*ie+1] = evals_mini[ie]
        for iorb in xrange(M):
            evecs[2*iorb,2*ie] = evecs_mini[iorb,ie]
            evecs[2*iorb+1,2*ie+1] = evecs_mini[iorb,ie]
    return evals, evecs

def eigh_ordered(mat):
    n=mat.shape[0]
    evals,evecs=np.linalg.eigh(mat)
    idx=np.argsort(evals)
    evecs2=np.zeros_like(evecs)
    evals2=np.zeros_like(evals)
    for ie in range (n):
        evals2[ie]=evals[idx[ie]]
        evecs2[:,ie]=1.0*evecs[:,idx[ie]]
    return evals2,evecs2

def dist_fd(e,beta):
    return 1./(1.+np.exp(e*beta))

def av_braket(wf1,A,wf2):
    return np.dot(np.dot(wf1.conjugate.transpose(),A),wf2)

def expikr(ik,ir,nk):
    kx=(2*pi/nk)*ik
    return np.exp(1J*kx*ir)

def oneshot_Uwfk2(norb,nk,phi,ek):
    Uwfk=np.zeros((nk,norb,),dtype=complex)
    Uwfr=np.zeros((norb,nk,norb),dtype=complex)
    Trans=np.zeros((norb,nk),dtype=float)
    ZTrans=np.zeros((norb,nk),dtype=complex)
    for ik in range(nk):
        for iwann in range(norb):
            assert abs(phi[ik,iwann,iwann])!=0
            rtmp=abs(phi[ik,iwann,iwann])
            Uwfk[ik,iwann]=phi[ik,iwann,iwann].conjugate()/rtmp

    for ik in range(nk):
        rtmp=0.0
        for iwann in range(norb):
            rtmp+=abs(Uwfk[ik,iwann])**2

    for iwann in range(norb):
        for ir in range(nk):
            for ik in range(nk):
                kx=(2*pi/nk)*ik
                Uwfr[iwann,ir,:]+=phi[ik,iwann,:]*Uwfk[ik,iwann]*np.exp(1J*kx*ir)
            Uwfr[iwann,ir,:]/=nk

    for iwann in range(norb):
        for ir in range(nk):
            for ik in range(nk):
                kx=(2*pi/nk)*ik
                ZTrans[iwann,ir]+=ek[ik,iwann]*(abs(Uwfk[ik,iwann])**2)*np.exp(1J*kx*ir)
            ZTrans[iwann,ir]/=nk

    Trans=ZTrans.real

    Uwfk_full=np.zeros((norb,nk,norb),dtype=complex)

    for ik in range(nk):
        for iwann in range(norb):
            Uwfk_full[iwann,ik,iwann]=Uwfk[ik,iwann]

    return Uwfk_full,Uwfr,Trans

def is_equal(a, b, eps=1e-5):
    if b==0:
        return np.abs(a)<eps
    else:
        return (np.abs(a-b)/np.abs(b) < eps)

def write_parms(f, parms):
    for k,v in parms.iteritems():
        if isinstance(v,str):
            #print >>f, k,"=",'"',v,'";'
            print >>f, '{}="{}"'.format(k,v)
        elif isinstance(v,complex):
            if v.imag != 0:
                ostr = str(k)+" = "+str(v.real)+'+(I*('+str(v.imag)+'));'
                f.write(ostr)
                print >>f, ""
            else:
                ostr = str(k)+" = "+str(v.real)+';'
                f.write(ostr)
                print >>f, ""
        else:
            print >>f, k,"=", v, ';'

def write_parms_to_ini(f, parms):
    for k,v in parms.iteritems():
        if isinstance(v,str):
            print >>f, '{}="{}"'.format(k,v)
        else:
            print >>f, k,"=", v, ''

def write_matrix(fname, matrix):
    f = open(fname,'w')
    N1 = matrix.shape[0]
    N2 = matrix.shape[1]
    for i in range(N1):
        for j in range(N2):
            print >>f, i, j, matrix[i,j].real, matrix[i,j].imag
    f.close()


def generate_U_tensor_Hubbard(n_site, U):
    U_tensor = np.zeros((n_site,2,n_site,2,n_site,2,n_site,2),dtype=complex)
    for i_site in xrange(n_site):
        U_tensor[i_site,0,i_site,1,i_site,1,i_site,0] = U
    return U_tensor.reshape((2*n_site,2*n_site,2*n_site,2*n_site))

def check_projectors(projectors):
    dim = projectors[0].shape[0]
    Umat = np.zeros((dim,dim),dtype=complex)
    offset = 0
    for iprj in xrange(len(projectors)):
        #print offset+projectors[iprj].shape[1]
        Umat[:,offset:offset+projectors[iprj].shape[1]] = 1.*projectors[iprj]
    assert(is_unitary(Umat))

def apply_projectors(projectors,self_ene):
    if len(projectors)==0:
        return

    assert self_ene.shape[1]==self_ene.shape[2]
    ntau = self_ene.shape[0]
    N = self_ene.shape[1]
    Nprj = len(projectors)

    self_ene_prj = np.zeros_like(self_ene)
    for itau in xrange(ntau):
        for iprj in xrange(Nprj):
            Uprj = projectors[iprj][:,:]
            self_ene_prj[itau,:,:] += np.dot(Uprj,np.dot(Uprj.conjugate().transpose(),np.dot(self_ene[itau,:,:],np.dot(Uprj,Uprj.conjugate().transpose()))))

    self_ene[:,:,:] = 1.*self_ene_prj

def apply_projectors_2d(projectors,mat):
    if len(projectors)==0:
        return

    assert mat.shape[0]==mat.shape[1]
    N = mat.shape[0]
    Nprj = len(projectors)

    mat_prj = np.zeros_like(mat)
    for iprj in xrange(Nprj):
        Uprj = projectors[iprj][:,:]
        mat_prj[:,:] += np.dot(Uprj,np.dot(Uprj.conjugate().transpose(),np.dot(mat[:,:],np.dot(Uprj,Uprj.conjugate().transpose()))))

    mat[:,:] = 1.*mat_prj

def diagonalize_with_projetion(mat,projectors):
    if len(projectors)==0:
        return eigh_ordered(mat)

    assert mat.shape[0]==mat.shape[1]
    N = mat.shape[0]
    Nprj = len(projectors)

    mat_prj = np.zeros_like(mat)
    evals = np.zeros((N,),dtype=float)
    evecs = np.zeros((N,N),dtype=complex)
    offset = 0
    for iprj in xrange(Nprj):
        Uprj = projectors[iprj][:,:]
        dim = Uprj.shape[1]
        mat_prj = np.dot(Uprj.conjugate().transpose(),np.dot(mat[:,:],Uprj))
        evals_prj, evecs_prj = eigh_ordered(mat_prj)
        evals[offset:offset+dim] = 1.*evals_prj
        evecs[:,offset:offset+dim] = np.dot(Uprj,evecs_prj)
        offset += dim
    assert(is_unitary(evecs))
    return evals, evecs

def compute_Tnl(n_matsubara, n_legendre):
    Tnl = np.zeros((n_matsubara, n_legendre), dtype=complex)
    for n in xrange(n_matsubara):
        sph_jn = scipy.special.sph_jn(n_legendre, (n+0.5)*np.pi)[0]
        for il in xrange(n_legendre):
            Tnl[n,il] = ((-1)**n) * ((1J)**(il+1)) * np.sqrt(2*il + 1.0) * sph_jn[il]
    return Tnl
