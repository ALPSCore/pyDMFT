import numpy as np
import sys

def mymod(Rx, Nx):
    if Rx>=0:
        return int(Rx)%Nx
    else:
        return (int(Rx)+((-int(Rx))/Nx+2)*Nx)%Nx

def pull_back_R(Ridx, primitive_vector, Nkdiv):
    return np.array([mymod(Ridx[i],Nkdiv) for i in xrange(3)])

def pull_back_R_to_wsc(Ridx, primitive_vector, Nkdiv):
    r = -1.0
    R_pos = Ridx[0]*primitive_vector[0]+Ridx[1]*primitive_vector[1]+Ridx[2]*primitive_vector[2]
    R_pos_r = np.zeros_like(R_pos)
    Ridx_r = 1.*Ridx
    for i in range(-1,2):
        for j in range(-1,2):
                for k in range(-1,2):
                    R_pos_tmp = R_pos+Nkdiv*(i*primitive_vector[0,:]+j*primitive_vector[1,:]+k*primitive_vector[2,:])
                    dist = np.sqrt(np.sum(R_pos_tmp**2))
                    if (r<0.0):
                        r = dist
                        R_pos_r = 1.0*R_pos_tmp
                        Ridx_r = Ridx+Nkdiv*np.array([i,j,k],dtype=int)
                    else:
                        if dist<r:
                            r = dist
                            R_pos_r = 1.0*R_pos_tmp
                            Ridx_r = Ridx+Nkdiv*np.array([i,j,k],dtype=int)
    assert r==np.sqrt(np.sum(R_pos_r**2))
    return Ridx_r

def compute_pos(abs_pos, R, primitive_vector):
    r = abs_pos+primitive_vector[0,:]*R[0]+primitive_vector[1,:]*R[1]+primitive_vector[1,:]*R[1]

def get_distance2(abs_pos1, abs_pos2, primitive_vector, Nkdiv):
    r = -1.0
    r12 = -1.0
    for i in range(-2,3):
        for j in range(-2,3):
                for k in range(-2,3):
                    abs_pos2_tmp = abs_pos2+Nkdiv*(i*primitive_vector[0,:]+j*primitive_vector[1,:]+k*primitive_vector[2,:])
                    dist = np.sqrt(np.sum((abs_pos1-abs_pos2_tmp)**2))
                    if (r<0.0):
                        r = dist
                        r12 = abs_pos1-abs_pos2_tmp
                    else:
                        if dist<r:
                            r12 = abs_pos1-abs_pos2_tmp
                        r = min(dist, r)
    return r, r12

class Wannier:
    def __init__(self, parms):
        self.Nkdiv = parms['QMAS_Nkdiv']
        self.Nk = self.Nkdiv**3
        self.Nwann = parms['QMAS_Nwann']
        self.primitive_vector = parms['QMAS_primitive_vector']
        self.wann_pos = parms['QMAS_wann_pos']
        Nk = self.Nk
        Nkdiv = self.Nkdiv
        Nwann = self.Nwann
        tmpdata = np.loadtxt(parms['QMAS_hamR'])

        assert self.Nk*self.Nwann**2==tmpdata.shape[0]

        self.HamR = np.zeros((Nk,Nwann,Nwann,),dtype=complex)
        for i in range(Nk*Nwann**2):
            self.HamR[int(tmpdata[i,0]-1),int(tmpdata[i,1]-1),int(tmpdata[i,2]-1)] = tmpdata[i,3] + 1J*tmpdata[i,4]
        #In QMAS, the ordering of the Wannier functions can be arbitrary. I always the following ordering:
        # wan1_up, wan2_up, ..., wanN_up, wan1_down, ..., wanN_down
        #Thus, I reorder HamR at this point.
        if not ('no_spinor' in parms and parms['no_spinor']):
            self.HamR = (self.HamR.reshape([Nk,2,Nwann/2,2,Nwann/2])).transpose([0,2,1,4,3])
        self.HamR = self.HamR.reshape([Nk,Nwann,Nwann])

        self.Ridx = np.zeros((Nk,3),dtype=int)
        ik = 0
        for ik3 in range(Nkdiv):
            for ik2 in range(Nkdiv):
                for ik1 in range(Nkdiv):
                        self.Ridx[ik,0] = ik1 
                        self.Ridx[ik,1] = ik2 
                        self.Ridx[ik,2] = ik3 
                        ik += 1

        #compute position
        self.abs_pos_wann = np.zeros((Nk,Nwann,3),dtype=float)
        #parms['QMAS_hamR']
        for iR in range(Nk):
            abs_R = self.Ridx[iR,0]*self.primitive_vector[0]+self.Ridx[iR,1]*self.primitive_vector[1]+self.Ridx[iR,2]*self.primitive_vector[2]
            for iwann in range(Nwann):
                self.abs_pos_wann[iR,iwann,:] = 1.*abs_R
                for x in range(3):
                    self.abs_pos_wann[iR,iwann,:] += self.primitive_vector[x]*self.wann_pos[iwann,x]

        #pull back Ridx to WSC
        self.Ridx_wsc = np.zeros((Nk,3),dtype=int)
        for iR in range(Nk):
            self.Ridx_wsc[iR,:] = pull_back_R_to_wsc(self.Ridx[iR,:], self.primitive_vector, self.Nkdiv)

        #Hermitialize H(R)
        map_R = np.zeros((Nkdiv,Nkdiv,Nkdiv),dtype=int)
        map_inv_R = np.zeros((Nk,),dtype=int)
        for iR in xrange(Nk):
            map_R[self.Ridx[iR,0],self.Ridx[iR,1],self.Ridx[iR,2]] = iR
        for iR in xrange(Nk):
            inv_R_pos = pull_back_R(-self.Ridx[iR,:], self.primitive_vector, self.Nkdiv)
            map_inv_R[iR] = map_R[inv_R_pos[0], inv_R_pos[1], inv_R_pos[2]]
        for iR in xrange(Nk):
            #print "debug1", iR, map_inv_R[iR]
            #print "debug1", iR, self.Ridx[iR,:]
            #print "debug2", self.Ridx[map_inv_R[iR],:]
            inv_R = map_inv_R[iR]
            self.HamR[iR,:,:] = 0.5*(self.HamR[iR,:,:]+self.HamR[inv_R,:,:].conjugate().transpose())

        self.map_R = map_R

    def get_Hk(self, kvec):
        Hk = np.zeros((self.Nwann,self.Nwann,),dtype=complex)
        for iR in range(self.Nk):
            Hk += self.HamR[iR,:,:]*np.exp(2J*np.pi*(self.Ridx_wsc[iR,0]*kvec[0]+self.Ridx_wsc[iR,1]*kvec[1]+self.Ridx_wsc[iR,2]*kvec[2]))
        return Hk

    def to_k(self, HR):
        Hk = np.zeros((self.Nk,self.Nwann,self.Nwann,),dtype=complex)
        kvec = np.zeros((3,),dtype=complex)
        for ik in xrange(self.Nk):
            kvec = self.Ridx[ik,:]/(1.*self.Nkdiv)
            for iR in xrange(self.Nk):
                Hk[ik,:,:] += HR[iR,:,:]*np.exp(2J*np.pi*(self.Ridx[iR,0]*kvec[0]+self.Ridx[iR,1]*kvec[1]+self.Ridx[iR,2]*kvec[2]))
        return Hk

    def to_R(self, Hk):
        HR = np.zeros((self.Nk,self.Nwann,self.Nwann,),dtype=complex)
        kvec = self.Ridx/self.Nkdiv
        for iR in xrange(self.Nk):
            for ik in xrange(self.Nk):
                HR[iR,:,:] += Hk[ik,:,:]*np.exp(-2J*np.pi*(self.Ridx[iR,0]*kvec[ik,0]+self.Ridx[iR,1]*kvec[ik,1]+self.Ridx[iR,2]*kvec[ik,2]))
                #if iR==0:
                    #print iR, ik, np.exp(-2J*np.pi*(self.Ridx[iR,0]*kvec[ik,0]+self.Ridx[iR,1]*kvec[ik,1]+self.Ridx[iR,2]*kvec[ik,2]))
                    #print Hk[ik,0:6,0:6]
        #print "debug ", Hk[0,0:6,0:6]
        #print "debug ", HR[0,0:6,0:6]/self.Nk
        return HR/self.Nk

    def get_HR(self, iR):
        return self.HamR[iR,:,:]

    def get_iR(self, R):
        return self.map_R[R[0], R[1], R[2]]
