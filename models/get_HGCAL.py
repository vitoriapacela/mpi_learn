import os
import glob
try:
    import h5py
    pass
except:
    print ("hum")
import numpy as np

def get_data(datafile):
    #get data for training
    #print ('Loading Data from .....', datafile)
    f = h5py.File(datafile,'r')
    
    y = np.array(f.get('true_energy'))
    pic3d = np.array(f.get('image'))
    #X = np.sum(np.abs(pic3d), axis=4)
    X = pic3d.astype(np.float32)
    X = np.expand_dims(X, axis=-1)
    y = y.astype(np.float32)
    sum_image = np.squeeze(np.sum(X, axis=(1, 2, 3)))
    print(X.shape)
    print(y.shape)
    print(sum_image.shape)

    f.close()
    return X, y, sum_image

dest = '/data/shared/HGCAL_energy/'

#if 'daint' in os.environ.get('HOST', os.environ.get('HOSTNAME','')):
#    dest='/scratch/snx3000/vlimant/3DGAN/'

for F in glob.glob('/bigdata/shared/HGCAL_energy_images/*.h5'):
    _,d,f = F.rsplit('/', 2)
    #if not 'Ele' in d: continue
    X = None
    sub_split = 1
    if sub_split==1:
        nf = '%s%s' % (dest, f)
        if os.path.isfile( nf) :
            continue
        print ("processing files", F, "into", nf)
        if X is None:
            X, y, sum_image = get_data(F)
        o = h5py.File(nf,'w')
        o['X'] = X
        o.create_group("y")
        o['y']['a'] = np.ones(y.shape) # whether the data is real
        o['y']['b'] = y # true_energy
        o['y']['c'] = sum_image # sum of energies
    else:
        for sub in range(sub_split):
            nf = '%s%s_sub%s'%(dest, f, sub)
            if os.path.isfile( nf) :
                continue
            print ("processing files",F,"into",nf)
            if X is None:
                X, y, sum_image = get_data(F)
                N = X.shape[0]
                splits = [i*N/sub_split for i in range(sub_split)]+[-1]
            o = h5py.File(nf,'w')
            o['X'] = X[splits[sub]:splits[sub+1],...]
            o.create_group("y")
            o['y']['a'] = np.ones(y[splits[sub]:splits[sub+1],...].shape)
            o['y']['b'] = y[splits[sub]:splits[sub+1],...]
            o['y']['c'] = sum_image[splits[sub]:splits[sub+1],...]
            o.close()
            X = None
    o.close()

open('train_3d_energy.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f, glob.glob(dest+'/*.h5')[:-4])))
open('test_3d_energy.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[-4:])))

#open('train_small_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[:-4])))
#open('test_small_3d.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[-4:])))

open('train_7_3d_energy.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[:7])))
open('test_1_3d_energy.list','w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[-1:])))