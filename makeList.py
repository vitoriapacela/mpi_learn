import glob
dest = '/bigdata/shared/HGCAL_data/new_multi_small/no_pu'
open('train_3d_energy.list', 'w').write( '\n'.join(filter(lambda f:not 'sub' in f, glob.glob(dest+'/*.h5')[:-4])))
open('test_3d_energy.list', 'w').write( '\n'.join(filter(lambda f:not 'sub' in f,glob.glob(dest+'/*.h5')[-4:])))
