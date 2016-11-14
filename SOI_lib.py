import numpy as np
from lib import *

def ToBigMatrix(sm,norb):
    sdim = sm.shape[0]
    bm = np.zeros((sdim*norb,sdim*norb),dtype=complex)
    for i in range(norb):
        offset = sdim*i
        bm[0+offset:sdim+offset,0+offset:sdim+offset] = 1.*sm
    return bm

def gen_J_basis():
    #xy_u xy_d, yz_u, yz_d, zx_u, zx_d jeff=-1/2 (mz>0)
    phi = np.zeros((6,6),dtype=complex)
    phi[:,0] = np.array([1,  0,  0,  1,   0,  1J], dtype=complex)/np.sqrt(3.0)  #j=1/2, jz= 1/2
    phi[:,1] = np.array([0, -1,  1,  0, -1J,  0], dtype=complex)/np.sqrt(3.0) #j=1/2, jz=-1/2
    phi[:,2] = np.array([0,  0,  1,  0, +1J,  0], dtype=complex)/np.sqrt(2.0) #j=3/2, jz= 3/2 #The plus sign is missing in Kargarian (2011)
    phi[:,3] = np.array([2,  0,  0, -1,   0, -1J], dtype=complex)/np.sqrt(6.0) #j=3/2, jz= 1/2
    phi[:,4] = np.array([0,  2,  1,  0, -1J,  0], dtype=complex)/np.sqrt(6.0) #j=3/2, jz=-1/2
    phi[:,5] = np.array([0,  0,  0,  1,   0, -1J], dtype=complex)/np.sqrt(2.0) #j=3/2, jz=-3/2

    return phi

def gen_LS_t2g():
    lx = np.zeros((3,3),dtype=complex)
    ly = np.zeros((3,3),dtype=complex)
    lz = np.zeros((3,3),dtype=complex)
    lx[0,2] = -1J
    ly[0,1] = +1J
    lz[1,2] = +1J
    for i in xrange(3):
        for j in xrange(3):
            if i<=j:
                continue
            lx[i,j] = lx[j,i].conjugate()
            ly[i,j] = ly[j,i].conjugate()
            lz[i,j] = lz[j,i].conjugate()

    Lx = np.zeros((6,6),dtype=complex)
    Ly = np.zeros((6,6),dtype=complex)
    Lz = np.zeros((6,6),dtype=complex)
    for iflavor in xrange(6):
        iorb = iflavor/2
        isp = iflavor%2
        for iflavor2 in xrange(6):
            iorb2 = iflavor2/2
            isp2 = iflavor2%2

            if isp==isp2:
                Lx[iflavor,iflavor2] = lx[iorb,iorb2]
                Ly[iflavor,iflavor2] = ly[iorb,iorb2]
                Lz[iflavor,iflavor2] = lz[iorb,iorb2]

    sx = np.array([[0, 1], [1, 0]], dtype=complex)/2
    sy = np.array([[0, -1J], [1J, 0]], dtype=complex)/2
    sz = np.array([[1, 0], [0, -1]], dtype=complex)/2

    Sx = np.zeros((6,6),dtype=complex)
    Sy = np.zeros((6,6),dtype=complex)
    Sz = np.zeros((6,6),dtype=complex)
    for iflavor in xrange(6):
        iorb = iflavor/2
        isp = iflavor%2
        for iflavor2 in xrange(6):
            iorb2 = iflavor2/2
            isp2 = iflavor2%2

            if iorb==iorb2:
                Sx[iflavor,iflavor2] = sx[isp,isp2]
                Sy[iflavor,iflavor2] = sy[isp,isp2]
                Sz[iflavor,iflavor2] = sz[isp,isp2]

    tri = np.zeros((3,3),dtype=complex)
    Tri = np.zeros((6,6),dtype=complex)
    for i in xrange(3):
        for j in xrange(3):
            if i==j:
                continue
            tri[i,j] = -0.5
    for iflavor in xrange(6):
        iorb = iflavor/2
        isp = iflavor%2
        for iflavor2 in xrange(6):
            iorb2 = iflavor2/2
            isp2 = iflavor2%2

            if isp==isp2:
                Tri[iflavor,iflavor2] = tri[iorb,iorb2]

    return Lx, Ly, Lz, Sx, Sy, Sz, Tri

def ave(phi, op):
    z = np.dot(phi.conjugate(),np.dot(op,phi))
    if np.abs(z.imag)>1E-8:
        raise RuntimeError("Finite imaginary part!")
    return z.real

def phi_dot(phi1, op, phi2):
    return np.dot(phi1.conjugate(),np.dot(op,phi2))

def gen_j111_basis():
    Jbasis = gen_J_basis()
    Lx,Ly,Lz,Sx,Sy,Sz,Tri = gen_LS_t2g()
    LS = np.dot(Lx,Sx)+np.dot(Ly,Sy)+np.dot(Lz,Sz)
    L111 = (Lx+Ly+Lz)/np.sqrt(3.0)
    S111 = (Sx+Sy+Sz)/np.sqrt(3.0)
    
    Jeffx = Sx-Lx
    Jeffy = Sy-Ly
    Jeffz = Sz-Lz
    Jeff111 = (Jeffx+Jeffy+Jeffz)/np.sqrt(3.0)
    
    evals,evecs = eigh_ordered(1.0*LS+0.0001*Tri+0.000001*Jeff111)
    evecs_sorted = np.zeros_like(evecs)
    evals_sorted = np.zeros_like(evals)
    for ie in xrange(6):
        evecs_sorted[:,5-ie] = evecs[:,ie]
        evals_sorted[5-ie] = evals[ie]

    return evecs_sorted 
