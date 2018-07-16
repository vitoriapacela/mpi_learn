import root_numpy
from binner_energy import process
import glob
import h5py
import numpy as np


branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy',
            'rechit_energy', 'rechit_phi', 'rechit_eta','rechit_layer', 'seed_eta', 'seed_phi'
           ]


def convert(f1):
    '''
    Adapted from Shah-Rukh Qasim.
    Converts data to images, only for the electrons in the dataset.
    :param f1: root file name
    :type f1: str
    :return: images and true_energy arrays
    :rtype: two lists of numpy arrays
    '''

    A = root_numpy.root2array(f1, treename='deepntuplizer/tree', branches=branches)

    num_entries = len(A['seed_eta'])
    #print("Number of entries is ", num_entries)

    pics = []
    trues = []

    for i in range(num_entries):
        if (A['isElectron'][i] == 1):
            ## pics:
            # N rechits per event
            rechit_energy = A['rechit_energy'][i]  # N energy
            rechit_eta = A['rechit_eta'][i]
            rechit_phi = A['rechit_phi'][i]
            rechit_layer = A['rechit_layer'][i]

            seed_eta = A['seed_eta'][i]  # scalar eta of the seed
            seed_phi = A['seed_phi'][i]  # scalar phi of the seed

            ## other data:
            true_energy = A['true_energy'][i]

            pic4d = process(rechit_eta, rechit_phi, rechit_layer, rechit_energy, seed_eta, seed_phi)
            reduced = np.sum(np.abs(pic4d), axis=3)

            pics.append(reduced)
            trues.append(true_energy)

    return pics, trues


def toH5(f1, pics, trues):
    '''
    Creates HDF5 file containing image and true_energy arrays for a root file.
    :param f1: root file name
    :type f1: str
    :param pics: array containing the images
    :type pics: numpy array
    :param trues: array containing the ground truth energy values
    :type param: numpy array
    '''
    file_name = f1.split('.')[3] + '.h5'

    h5f = h5py.File(file_name, 'w')

    h5f.create_dataset('image', data=pics)

    h5f.create_dataset('true_energy', data=trues)

    h5f.close()


def main():
    d = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/convertedH_FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/fourth/train_fourth/'

    root_files = glob.glob(d + '*.root')
    root_files.remove('/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/convertedH_FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/fourth/train_fourth/partGun_PDGid11_x160_Pt2.0To100.0_NTUP_199.root')

    for f in root_files:
        pics, trues = convert(f)
        toH5(f, pics, trues)


if __name__ == "__main__":
    main()
