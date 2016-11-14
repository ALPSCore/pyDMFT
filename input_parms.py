import sys
import h5py
import numpy as np
#import pyalps.mpi as mpi     # MPI library
import sys
import re

def read_input_parms(fname):
    parms = {}
    f = h5py.File(fname, 'r')
    for key in list(f["/parameters"]):
       parms[key] = f["/parameters/"+key].value
    return parms

def load_ivk_dos(app_parms):
    ndiv_tau=app_parms["NMATSUBARA"]
    n_ivk_dos=app_parms["N_GRID_IVK_DOS"]
    ivk_ene=np.zeros((ndiv_tau, n_ivk_dos,),dtype=float)
    ivk_dos=np.zeros((ndiv_tau, n_ivk_dos,),dtype=float)

    ZB = -1.0
    if 'GEN_SCR_IVK_DOS' in app_parms and app_parms['GEN_SCR_IVK_DOS']==1:
        print "Generating screened interaction..."
        tmp_data,U_unscr,ZB = load_retarded_UV(app_parms, 'SCR')
        ivk_ene[:,:]=tmp_data[:,0,:]
        ivk_dos[:,:]=tmp_data[:,1,:]
        app_parms['U']=U_unscr
    elif 'GEN_UNSCR_IVK_DOS' in app_parms and app_parms['GEN_UNSCR_IVK_DOS']==1:
        print "Generating UNscreened interaction..."
        tmp_data,U_unscr,ZB = load_retarded_UV(app_parms, 'UNSCR')
        ivk_ene[:,:]=tmp_data[:,0,:]
        ivk_dos[:,:]=tmp_data[:,1,:]
        app_parms['U']=U_unscr
    elif 'GEN_STATIC_IVK_DOS' in app_parms and app_parms['GEN_STATIC_IVK_DOS']==1:
        print "Generating static interaction + band renormalization..."
        tmp_data,U_unscr,ZB = load_retarded_UV(app_parms, 'STATIC')
        ivk_ene[:,:]=tmp_data[:,0,:]
        ivk_dos[:,:]=tmp_data[:,1,:]
        app_parms['U']=U_unscr
    elif int(app_parms['RETARDED_VK'])==1:
        print "Loading retarded_vk..."
        raise RuntimeError("This is not supported any more")
        tmp_data = np.load(app_parms['IVK_DOSFILE'])
        print tmp_data.shape
        if tmp_data.shape[0]!=ndiv_tau or (tmp_data.shape[1]!=2 or tmp_data.shape[2]!= n_ivk_dos):
            print "Error in load_ivk_dos"
            print ndiv_tau, 2, n_ivk_dos
            sys.exit(1)
        for i in range(ndiv_tau):
            ivk_ene[i,:]=tmp_data[i,0,:]
            ivk_dos[i,:]=tmp_data[i,1,:]
            #if mpi.rank==0:
                #print "norm of ivk DOS at i_nu=", i, np.trapz(ivk_dos[i],ivk_ene[i])
    else:
        f=open(app_parms['IVK_DOSFILE'],"r")
        for i in range(n_ivk_dos):
            l = f.readline()
            data = re.split('[\s\t)(,]+', l.strip())
            ivk_ene[:,i]=float(data[0])
            ivk_dos[:,i]=float(data[1])
        f.close()
        #if mpi.rank==0:
            #print "norm of ivk DOS=", np.trapz(ivk_dos[0],ivk_ene[0])

    return ivk_ene, ivk_dos, ZB
