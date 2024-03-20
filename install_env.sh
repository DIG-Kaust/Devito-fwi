#!/bin/bash
# 
# Installer for Devito-fwi environment
#
# Run: ./install_env.sh
#
# M. Ravasi, 17/03/2024

echo 'Creating Devito-fwi environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate devitofwi
echo 'Created and activated environment:' $(which python)

# check packages work as expected
echo 'Checking devito version and running a command...'
python -c 'import devito; print(devito.__version__);'

echo 'Done!'