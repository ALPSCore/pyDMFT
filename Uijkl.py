import numpy as np
import sys

up = 0
down = 1

def complex_to_str(z):
    return "(%e,%e)"%(z.real,z.imag)

#We use the same notation as in Gorenov (2009)
#U_{ijkl} c_{i,s}^\dagger c_{j,s'}^\dagger c_{k,s'} c_{l,s}
class NormalOrderedInteraction:
    def __init__(self,coeff,i,j,k,l,s,sp):
        self.coeff = coeff
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.s = s
        self.sp = sp

    # If s != sp, we always take s=up, s'=down
    # If s == sp, we always take i<=j and k<=l 
    def to_unique_convention(self):
        if self.s==self.sp:
            i = -1
            j = -1
            k = -1
            l = -1
            sign = 1.
            if self.i>self.j:
                i,j = self.j, self.i
                sign *= -1
            else:
                i,j = self.i, self.j

            if self.k>self.l:
                k,l = self.l, self.k
                sign *= -1
            else:
                k,l = self.k, self.l
            return NormalOrderedInteraction(sign*self.coeff,i,j,k,l,self.s,self.s)
        else:
            if self.s==up and self.sp==down:
                return NormalOrderedInteraction(self.coeff, self.i, self.j, self.k, self.l, self.s, self.sp)
            else: #swap spins
                return NormalOrderedInteraction(self.coeff, self.j, self.i, self.l, self.k, self.sp, self.s)

    def key_without_coeff(self):
        return (self.i, self.j, self.k, self.l, self.s, self.sp)

    #if the Pauli principle is surely violated, it returns true.
    #Note: even if it returns false, the Pauli principle may be violated. (I am not sure!)
    def surely_Pauli_principle_violated(self):
        return (self.s==self.sp) and ((self.i==self.j) or (self.k==self.l))

    def convert_sto(self):
        if self.s!=self.sp:
            return StaggeredOrderedInteraction(self.coeff,self.i,self.l,self.j,self.k,self.s,self.sp)
        else:
            if self.i==self.j or self.k==self.l:
                return None #Pauli principle
            else:
                if self.j!=self.l:
                    return StaggeredOrderedInteraction(self.coeff,self.i,self.l,self.j,self.k,self.s,self.sp)
                else:
                    return StaggeredOrderedInteraction(-self.coeff,self.i,self.k,self.j,self.l,self.s,self.sp)

    def __repr__(self):
        return "U=%f,(i,j,k,l)=(%i,%i,%i,%i);(s,s')=(%i,%i)"%(self.coeff,self.i, self.j, self.k, self.l, self.s, self.sp)

def unique_normal_ordered_interaction_list(org_list):
    dist = {}
    for item in org_list:
        if item.surely_Pauli_principle_violated():
            continue

        item_tmp = item.to_unique_convention()
        this_key = item_tmp.key_without_coeff()
        if not dist.has_key(this_key):
            dist[this_key] = item_tmp
        else:
            dist[this_key].coeff = dist[this_key].coeff+item_tmp.coeff
    return dist.values()       

#U_{ijkl} c_{i,s}^\dagger c_{j,s} c_{k,s'}^\dagger c_{l,s'}
class StaggeredOrderedInteraction:
    def __init__(self,coeff,i,j,k,l,s,sp):
        self.coeff = coeff
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.s = s
        self.sp = sp

    def swap_c_ops(self):
        assert self.s==self.sp
        assert self.j!=self.k
        assert self.j!=self.l
        assert self.k!=self.l
        self.j, self.l = self.l, self.j
        self.coeff *= -1

    def __repr__(self):
        return "U=%f,(i,j,k,l)=(%i,%i,%i,%i);(s,s')=(%i,%i)"%(self.coeff,self.i, self.j, self.k, self.l, self.s, self.sp)

#(U_{ijkl}/n_af) (c_{i,s}^\dagger c_{j,s}-alpha) (c_{k,s'}^\dagger c_{l,s'}-alpha)
class StaggeredOrderedInteractionAF:
    def __init__(self,coeff,n_site,i,j,k,l,s,sp,n_af,alpha):
        self.coeff = coeff
        self.n_site = n_site
        self.i = i
        self.j = j
        self.k = k
        self.l = l
        self.s = s
        self.sp = sp
        self.n_af = n_af
        self.alpha = alpha
        assert self.n_af==len(alpha)

    #Generate an object from an object of StaggeredOrderedInteraction
    @staticmethod
    def add_AF(so_int, n_site, n_af, alpha):
        return StaggeredOrderedInteractionAF(so_int.coeff, n_site, so_int.i, so_int.j, so_int.k, so_int.l, so_int.s, so_int.sp, n_af, alpha)

    def __repr__(self):
        return "U=%f,(i,j,k,l)=(%i,%i,%i,%i);(s,s')=(%i,%i)"%(self.coeff,self.i, self.j, self.k, self.l, self.s, self.sp)

    def get_H0_correction(self, n_spin):
        H0_corr = np.zeros((self.n_site,n_spin,self.n_site,n_spin),dtype=complex)
        for iaf in xrange(self.n_af):
            H0_corr[self.k, self.sp, self.l, self.sp] += self.alpha[iaf][0]
            H0_corr[self.i, self.s, self.j, self.s] += self.alpha[iaf][1]
        H0_corr *= self.coeff/self.n_af
        return H0_corr

    @staticmethod
    def write_as_text(terms, f):
        n_nonzero = 0
        for t in terms:
            if np.abs(t.coeff)>1E-12:
                n_nonzero += 1

        i = 0
        print >>f, n_nonzero
        for t in terms:
            if np.abs(t.coeff)<=1E-12:
                continue

            print >>f, i, 2, t.n_af, complex_to_str(t.coeff),
            print >>f, t.i, t.j, t.k, t.l, #sites
            print >>f, t.s, t.sp, #spins
            for i_af in xrange(t.n_af):
                for irank in xrange(2):
                    print >>f, complex_to_str(t.alpha[i_af][irank]),
            print >>f, ""
            i += 1

    @staticmethod
    def write_as_text_spin_fold(terms, f):
        n_nonzero = 0
        for t in terms:
            if np.abs(t.coeff)>1E-12:
                n_nonzero += 1

        i = 0
        print >>f, n_nonzero
        for t in terms:
            if np.abs(t.coeff)<=1E-12:
                continue

            assert t.s==t.sp
            assert t.i%2==t.j%2
            assert t.k%2==t.l%2

            print >>f, i, 2, t.n_af, complex_to_str(t.coeff),
            print >>f, t.i/2, t.j/2, t.k/2, t.l/2, #sites
            print >>f, t.i%2, t.k%2, #spins
            for i_af in xrange(t.n_af):
                for irank in xrange(2):
                    print >>f, complex_to_str(t.alpha[i_af][irank]),
            print >>f, ""
            i += 1

#See Shinaoka (2015): negative sign problem paper
def generate_SK(site1, site2, U, Up, JH, JHp):
    int_list = []

    #onsite U
    for site in [site1, site2]:
        int_list.append(NormalOrderedInteraction(U,site,site,site,site,up,down))

    #U_(ijij) in Eq. (D9)
    int_list.append(NormalOrderedInteraction(Up,site1,site2,site2,site1,up,up))
    int_list.append(NormalOrderedInteraction(Up,site1,site2,site2,site1,down,down))
    for i in [site1, site2]:
        for j in [site1, site2]:
            if i==j:
                continue
            for s1 in [up, down]:
                for s2 in [up, down]:
                    int_list.append(NormalOrderedInteraction(0.5*Up,i,j,j,i,s1,s2))

    # Hund's coupling and spin flip (D10)
    for i in [site1, site2]:
        for j in [site1, site2]:
            for s1 in [up, down]:
                for s2 in [up, down]:
                    if i==j:
                        continue
                    int_list.append(NormalOrderedInteraction(0.5*JH,i,j,i,j,s1,s2))

    # pair hopping (D11)
    for i in [site1, site2]:
        for j in [site1, site2]:
            if i==j:
                continue
            int_list.append(NormalOrderedInteraction(JHp,i,i,j,j,up,down))
        
    int_list = unique_normal_ordered_interaction_list(int_list)

    st_int_list = []
    for i in xrange(len(int_list)):
        tmp = int_list[i].convert_sto()
        if tmp != None:
            st_int_list.append(tmp)

    return st_int_list

#U_{ijkl} c_i^\dagger c_j^\dagger c_k c_l = U \sum_i n_{i,up} n_{i,down}
# spin and orbital indices are combined.
def generate_U_tensor_Hubbard(n_site, U):
    U_tensor = np.zeros((n_site,2,n_site,2,n_site,2,n_site,2),dtype=complex)
    for i_site in xrange(n_site):
        U_tensor[i_site,0,i_site,1,i_site,1,i_site,0] = U
    return U_tensor.reshape((2*n_site,2*n_site,2*n_site,2*n_site))

# U_tensor: U(alpha, beta, alpha', beta') c^\dagger_alpha c^\dagger_beta c_alpha' c_beta'
def generate_staggered_interaction(U_tensor, n_site, n_spin, spin_orbit_composite=False):
    if spin_orbit_composite:
        if n_spin!=1:
            raise RuntimeError("n_spin must be 1 if spin_orbit_composite=True")
    no_int_list = []
    for site1 in xrange(n_site):
        for site2 in xrange(n_site):
            for site3 in xrange(n_site):
                for site4 in xrange(n_site):
                    Uval = U_tensor[site1,site2,site3,site4]
                    if np.abs(Uval)<1E-10:
                        continue
                    for s1 in xrange(n_spin):
                        for s2 in xrange(n_spin):
                            no_int_list.append(NormalOrderedInteraction(Uval,site1,site2,site3,site4,s1,s2))

    #remove duplicate terms
    no_int_list = unique_normal_ordered_interaction_list(no_int_list)
    
    st_int_list = []
    for i in xrange(len(no_int_list)):
        tmp = no_int_list[i].convert_sto()
        if tmp != None:
            st_int_list.append(tmp)

    #When using the spin-orbit composite representation, we have to sort spins.
    if spin_orbit_composite:
        for item in st_int_list:
            if item.i%2 != item.j%2:
                item.swap_c_ops()
                assert item.i%2==item.j%2
                assert item.k%2==item.l%2

    return st_int_list

#Rotate a Coulomb tensor according to Eq. (D6).
# U_tensor: U(alpha, beta, alpha', beta') c^\dagger_alpha c^\dagger_beta c_alpha' c_beta'
# V_mat: a unitary matrix for a basis transformation
def rotate_Coulomb_tensor(U_tensor, V_mat):
    n_site = V_mat.shape[0]
    assert np.sum(np.abs(np.dot(V_mat,V_mat.conjugate().transpose())-np.identity(n_site)))<1E-10
    tmp = np.tensordot(U_tensor.transpose(0,1,3,2),V_mat.conjugate(),axes=[0,0])
    tmp = np.tensordot(tmp,V_mat.conjugate(),axes=[0,0])
    tmp = np.tensordot(tmp,V_mat,axes=[0,0])
    tmp = np.tensordot(tmp,V_mat,axes=[0,0])
    return tmp.transpose(0,1,3,2)


# U_tensor: U(alpha, beta, alpha', beta') c^\dagger_alpha c^\dagger_beta c_alpha' c_beta'
def write_Uijkl(U_tensor, V_mat, fname, spin_diag=True, alpha_diag=1E-2, alpha=1E-4, cut=1E-10):
    assert len(U_tensor.shape)==4
    assert U_tensor.shape[0]==U_tensor.shape[1]
    assert U_tensor.shape[0]==U_tensor.shape[2]
    assert U_tensor.shape[0]==U_tensor.shape[3]

    n_spin = 1 #up and down are combined.

    n_site = U_tensor.shape[0]
    U_tensor_rot = rotate_Coulomb_tensor(U_tensor, V_mat)

    st_int_list2 = generate_staggered_interaction(U_tensor_rot, n_site, n_spin, spin_diag)
    st_int_nzero = []
    for tmp in st_int_list2:
        if np.abs(tmp.coeff)>1E-10:
            st_int_nzero.append(tmp)

    st_int_af_list = []
    H0_corr = np.zeros((n_site,1,n_site,1),dtype=complex)
    for tmp in st_int_list2:
        #print "debug ", tmp
        if tmp.i==tmp.j and tmp.k==tmp.l:
            st_int_af_list.append(StaggeredOrderedInteractionAF.add_AF(tmp, n_site, 2, [[1+alpha_diag,-alpha_diag],[-alpha_diag,alpha_diag+1]]))
        else:
            st_int_af_list.append(StaggeredOrderedInteractionAF.add_AF(tmp, n_site, 4, [[alpha,-alpha],[-alpha,alpha],[alpha,alpha],[-alpha,-alpha]]))
        H0_corr += st_int_af_list[-1].get_H0_correction(1)

    f = open(fname, 'w')
    if spin_diag:
        StaggeredOrderedInteractionAF.write_as_text_spin_fold(st_int_af_list, f)
        H0_corr = H0_corr.reshape((n_site/2,2,n_site/2,2))
    else:
        StaggeredOrderedInteractionAF.write_as_text(st_int_af_list, f)
    f.close()
    return H0_corr

#Order of operators: c^\dagger_{iorb1} c^\dagger_{iorb2} c_{iorb3} c_{iorb4}
def generate_U_tensor_SK(n_orb, U, JH):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    for iorb1 in xrange(n_orb):
        for iorb2 in xrange(n_orb):
            for iorb3 in xrange(n_orb):
                for iorb4 in xrange(n_orb):
                    coeff = 0.0
                    if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
                        coeff = U
                    elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
                        coeff = U-2*JH
                    elif iorb1==iorb3 and iorb2==iorb4 and iorb1!=iorb2:
                        coeff = JH
                    elif iorb1==iorb2 and iorb3==iorb4 and iorb1!=iorb3:
                        coeff = JH

                    for isp in xrange(2):
                        for isp2 in xrange(2):
                            U_tensor[iorb1,isp,    iorb2,isp2,    iorb3,isp2,  iorb4,isp] += 0.5*coeff

    return U_tensor.reshape((2*n_orb,2*n_orb,2*n_orb,2*n_orb))

#Order of operators: c^\dagger_{iorb1} c^\dagger_{iorb2} c_{iorb3} c_{iorb4}
def generate_U_tensor_SK2(n_orb, U, Up, JH):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    for iorb1 in xrange(n_orb):
        for iorb2 in xrange(n_orb):
            for iorb3 in xrange(n_orb):
                for iorb4 in xrange(n_orb):
                    coeff = 0.0
                    if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
                        coeff = U
                    elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
                        coeff = Up
                    elif iorb1==iorb3 and iorb2==iorb4 and iorb1!=iorb2:
                        coeff = JH
                    elif iorb1==iorb2 and iorb3==iorb4 and iorb1!=iorb3:
                        coeff = JH

                    for isp in xrange(2):
                        for isp2 in xrange(2):
                            U_tensor[iorb1,isp,    iorb2,isp2,    iorb3,isp2,  iorb4,isp] += 0.5*coeff

    return U_tensor.reshape((2*n_orb,2*n_orb,2*n_orb,2*n_orb))

#Order of operators: c^\dagger_{iorb1} c^\dagger_{iorb2} c_{iorb3} c_{iorb4}
def generate_U_tensor_SK_density(n_orb, U, JH):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    for iorb1 in xrange(n_orb):
        for iorb2 in xrange(n_orb):
            for iorb3 in xrange(n_orb):
                for iorb4 in xrange(n_orb):
                    coeff = 0.0
                    if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
                        coeff = U
                    elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
                        coeff = U-2*JH

                    for isp in xrange(2):
                        for isp2 in xrange(2):
                            U_tensor[iorb1,isp,    iorb2,isp2,    iorb3,isp2,  iorb4,isp] += 0.5*coeff

    return U_tensor.reshape((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
