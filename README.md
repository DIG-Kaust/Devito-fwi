![LOGO](https://github.com/DIG-Kaust/Project_Template/blob/master/logo.png)

Reproducible material for **XXX -
Ravasi M., Author M., Author C.** submitted to XXX.


## Project structure
This repository is organized as follows:

* :open_file_folder: **package**: python library containing routines for ....;
* :open_file_folder: **data**: folder containing data (or instructions on how to retrieve the data
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments ...

## Notebooks
The following notebooks are provided:

- :orange_book: ``X1.ipynb``: notebook performing ...;
- :orange_book: ``X2.ipynb``: notebook performing ...


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate my_env
```

Finally, to run tests simply type:
```
pytest
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
