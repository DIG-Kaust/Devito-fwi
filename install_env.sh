#!/bin/bash
# 
# Installer for package
# 
# Run: ./install.sh
# 
# M. Ravasi, 24/05/2022

echo 'Creating my_env environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy and torch work as expected
echo 'Checking cupy and torch versions and running a command...'
python -c 'import cupy as cp; print(cp.__version__); import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

