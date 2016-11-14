import numpy as np
import h5py

def create_dataset_from_dict(f, path, dict):
    dset = []
    for k,v in dict.items():
        if isinstance(v,np.ndarray):
            dset.append(f.create_dataset(path+'/'+k, data=v))
        else:
            dset.append(f.create_dataset(path+'/'+k, data=v))
    return dset

def dump_results_modmft(app_parms, isc, obj, exclude_list=[]):
    nsc = isc+1

    if nsc==1:
        f = h5py.File(app_parms['prefix']+'.out.h5','w')
        create_dataset_from_dict(f,'parameters',app_parms)
    else:
        f = h5py.File(app_parms['prefix']+'.out.h5','a')

    #write all instances in obj (only the last iteration)
    if nsc>1:
        if 'results' in f:
            del f['results']
        if 'NUM_RESULTS' in f:
            del f['NUM_RESULTS']
    for k,v in vars(obj).items():
        if k in exclude_list:
            continue

        if isinstance(v,dict):
            create_dataset_from_dict(f,'results/'+k,v)
            create_dataset_from_dict(f,'results'+str(nsc-1)+'/'+k,v)
        elif isinstance(v,np.ndarray):
            f.create_dataset('results/'+k, data=v)
            f.create_dataset('results'+str(nsc-1)+'/'+k, data=v)
        else:
            f.create_dataset('results/'+k, data=v)
            f.create_dataset('results'+str(nsc-1)+'/'+k, data=v)

    f.create_dataset('NUM_RESULTS', data=nsc)

    f.close()

class DMFTResult:
    def update(self, Hk_mean, Hk_var, Hk_var2, vmu, G_latt, G_imp, G_tau_im, self_ene, hyb_n, hyb_tau):
        self.Hk_mean = np.array(Hk_mean)
        self.Hk_var = np.array(Hk_var)
        self.Hk_var2 = np.array(Hk_var2)
        self.vmu = vmu
        self.G_latt = np.array(G_latt)
        self.hyb_n = np.array(hyb_n)
        self.hyb_tau = np.array(hyb_tau)
        self.G_imp = np.array(G_imp)
        self.G_tau_im = np.array(G_tau_im)
        self.self_ene = np.array(self_ene)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

def load_observables(fname):
    print "Opening ", fname
    f = h5py.File(fname,'r')

    keys = f['/simulation/results'].keys()

    keys_t = []
    means_t = []
    errors_t = []
    for obs in keys:
        try:
            m = f['/simulation/results/'+obs+'/mean/value'].value
            e = f['/simulation/results/'+obs+'/mean/error'].value
            keys_t.append(obs)
            means_t.append(m)
            errors_t.append(e)
        except:
            print "Not loaded ", obs

    return keys_t,means_t,errors_t


#f = h5py.File(app_parms['prefix']+'.out.h5','w')
#f = h5py.File('tmp.out.h5','w')
#
#a = np.zeros((2,2),dtype=complex)
#set = f.create_dataset('result/array', data=a)
#
#
#dict = {
        #"A" : 1,
        #"B" : "str"
        #}
##create_dataset_from_dict('parameters',dict)
#create_dataset_from_dict(f, 'parameters',dict)

#f.close()
