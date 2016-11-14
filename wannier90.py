import numpy as np

class Wannier90:
    def __init__(self, parms):
        f = open(parms['Wannier90_hr'])
        f.readline()
        self.Nwann = int(f.readline())
        self.nrpts = int(f.readline())

        num_lines = self.nrpts/15
        if self.nrpts%15 != 0:
            num_lines += 1
        ndgen = []
        for iline in xrange(num_lines):
            ndgen.extend(f.readline().split())
        ndgen = np.array(ndgen, dtype=int)

        self.HamR = np.zeros((self.nrpts, self.Nwann, self.Nwann), dtype=complex)
        self.irvec = np.zeros((self.nrpts, 3), dtype=int)
        for ir in xrange(self.nrpts):
            for i in xrange(self.Nwann):
                for j in xrange(self.Nwann):
                    i1,i2,i3,i4,i5,r1,r2 = f.readline().split()
                    self.HamR[ir, int(i4)-1,int(i5)-1] = (float(r1) + 1J * float(r2))/(1.*ndgen[ir]) 
            self.irvec[ir,0] = i1
            self.irvec[ir,1] = i2
            self.irvec[ir,2] = i3

        self.HamR = self.HamR.reshape((self.nrpts, 2, self.Nwann/2, 2, self.Nwann/2)).transpose((0,2,1,4,3)).reshape((self.nrpts, self.Nwann, self.Nwann))
     
    def get_Hk(self, kvec):
        Hk = np.zeros((self.Nwann,self.Nwann,),dtype=complex)
        for iR in range(self.nrpts):
            Hk += self.HamR[iR,:,:]*np.exp(2J*np.pi*(self.irvec[iR,0]*kvec[0]+self.irvec[iR,1]*kvec[1]+self.irvec[iR,2]*kvec[2]))
        return Hk
