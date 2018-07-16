import h5py
import glob
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

truth = np.array([])
raw = np.array([])

for fileName in glob.glob('/afs/cern.ch/user/v/vbarinpa/work_dir/convert/experimental_mod/new/PU/*.h5'):
    print(fileName)
    myFile = h5py.File(fileName)
    myraw = myFile.get('image') 
    mytruth = myFile.get('true_energy')
    
    raw = np.concatenate((raw, myraw), axis=0) if raw.size else myraw
    truth = np.concatenate((truth ,mytruth), axis=0) if truth.size else mytruth
    
print("loop over")

fileOut = h5py.File('/afs/cern.ch/user/v/vbarinpa/work_dir/convert/experimental_mod/new/all_PU.h5', 'w')

print('file created')

fileOut.create_dataset('image', data=raw,compression='gzip')
fileOut.create_dataset('true_energy', data=truth,compression='gzip')

print("done!")

fileOut.close()
