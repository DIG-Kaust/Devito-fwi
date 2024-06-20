#!/bin/bash
# 
# Installer for Devito-fwi environment with Torch-GPU
#
# Run: ./install_mpienv.sh
#
# M. Ravasi, 19/07/2024

echo 'Creating Devito-fwi environmentw with Torch-GPU'

# create conda env
conda env create -f environment-gpu.yml
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate devitofwi_gpu
echo 'Created and activated environment:' $(which python)

# check packages work as expected
echo 'Checking devito version and running a command...'
python -c 'import devito; print(devito.__version__);'

echo 'Done!'