#!/bin/bash
# 
# Installer for Devito-fwi environment with MPI
#
# Run: ./install_mpienv.sh
#
# M. Ravasi, 04/06/2024

echo 'Creating Devito-fwi environment with MPI'

# create conda env
conda env create -f environment-mpi.yml
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate devitofwi_mpi
echo 'Created and activated environment:' $(which python)

# install torch cpu only (change for any other version of gpu torch)
conda install pytorch cpuonly -c pytorch

# check packages work as expected
echo 'Checking devito version and running a command...'
python -c 'import devito; print(devito.__version__);'

echo 'Done!'