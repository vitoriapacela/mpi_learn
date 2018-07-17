# mpi_learn ![Build Status](https://travis-ci.org/vitoriapacela/mpi_learn.svg?branch=master)
GANs for HGCAL data.
Currently only simple_train implemented, not distributed training with mpi.

For more information about `mpi_learn`, check the [original branch](https://github.com/duanders/mpi_learn).

## Depencencies:
* [`NumPy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`h5py`](http://www.h5py.org/)
* `setGPU`
* `gpustat`
* [`tensorflow-gpu`](https://www.tensorflow.org/)
* [`keras`](https://keras.io/) (v. >= 1.2.0)
* [`root_numpy`](https://github.com/scikit-hep/root_numpy)
* [`mpi4py`](http://mpi4py.readthedocs.io/en/stable/) (v. >= 2.0.0)
* [`OpenMPI`](https://www.open-mpi.org/)


## Use with the HGCAL dataset:
* `git clone https://github.com/vitoriapacela/mpi_learn.git`
* `cd mpi_learn`
* Modify `models/get_HGCAL.py` according to the location of the data.
* `python models/get_HGCAL.py`
* Modify `mpi_learn/train/HGCALModel.py` according to the shape and format of the data.
* `python simple_train`
