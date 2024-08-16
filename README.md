:vertical_traffic_light: :vertical_traffic_light: This library is under early development.
Expect things to constantly change until version v1.0.0. :vertical_traffic_light: :vertical_traffic_light:

This library contains all the required building blocks to perform Full Waveform Inversion (FWI) with Devito (for modelling). 
Our vision is to provide an abstract template to perform FWI with all sort of pre- and post-processing hooks that users can 
develop and seamlessy include into a stable and robust FWI engine.

By doing so, we also aim integrate this library with PyLops and PyProximal to benefit from their modular handling of linear 
and proximal operators.

At its current stage of development, ``devito-fwi`` can be schematically represented as follows:

![DevitoFWIschematic](https://github.com/DIG-Kaust/Devito-fwi/blob/main/asset/fwistructure.png)

## Project structure
This repository is organized as follows:

* :open_file_folder: **devitofwi**: python library containing routines to perform FWI with Devito modelling engines;
* :open_file_folder: **data**: folder containing sample data to run notebooks;
* :open_file_folder: **notebooks**: set of jupyter notebooks used to showcase different ways to run FWI with ``devitofwi``;
* :open_file_folder: **scripts**: set of scripts notebooks used to showcase different ways to run FWI with ``devitofwi`` in distributed mode;

## Notebooks
The following notebooks are provided:
 
- :orange_book: ``acoustic/Modelling_filtering.ipynb``: notebook comparing modelling with a filtered wavelet and filtering of original data computed with unfiltered wavelet;
- :orange_book: ``acoustic/Gradient_comparison.ipynb``: notebook comparing gradients from different objective functions;
- :orange_book: ``acoustic/Gradient_decoupled.ipynb``: notebook comparing gradient computations with mixed computational graphs including a Torch Module (useful to embed NNs in FWI);
- :orange_book: ``acoustic/Data_visualization.ipynb``: notebook showcasing different visualization routines;
- :orange_book: ``acoustic/Acoustic_RTM.ipynb``: notebook performing acoustic RTM;
- :orange_book: ``acoustic/AcousticVel_L2_1stage.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data;
- :orange_book: ``acoustic/AcousticVel_L2_1stagewrapper.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data using AcousticFWI wrapper;
- :orange_book: ``acoustic/AcousticVel_L2torch_1stage.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data using Torch AD-based objective function;
- :orange_book: ``acoustic/AcousticVel_L2adam_1stage.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data wrapping wave engine in a Torch AD-based graph and using Adam solver;
- :orange_book: ``acoustic/AcousticVel_L2refr_1stage.ipynb``: notebook performing acoustic FWI parametrized in velocity with only refracted waves;
- :orange_book: ``acoustic/AcousticVel_XCorrrefr_1stagewrapper.ipynb``: notebook performing acoustic FWI parametrized in velocity with only refracted waves using XCorr objective function and AcousticFWI wrapper;
- :orange_book: ``acoustic/AcousticVel_L2_Nstages.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data in N frequency stages;
- :orange_book: ``acoustic/AcousticVel_L2_Nstageswrapper.ipynb``: notebook performing acoustic FWI parametrized in velocity with entire data in N frequency stages using AcousticFWI wrapper;
- :orange_book: ``acoustic/AcousticVel_L2refr_Nstages.ipynb``: notebook performing acoustic FWI parametrized in velocity with only refracted waves in N frequency stages.


## Scripts
The following scripts are also provided:
- :orange_book: ``acoustic/Gradient_comparison_mpi.ipynb``: notebook computing gradients distributed across sources;
- :orange_book: ``acoustic/AcousticVel_L2_1stage.ipynb``: notebook performing acoustic FWI parametrized in velocity distributed across sources;


## Getting started :space_invader: :robot:

### Serial version
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
conda activate devitofwi
```

### MPI version
Follow the same procedure described above. However, run 
```
./install_mpienv.sh
```

and remember to always activate the environment by typing:
```
conda activate devitofwi_mpi
```

## To do list :memo:

The following list is intended to define some of the improvements that should be implemented moving forward:

- [ ] Optimize time step in N stages FWI to avoid using a too large time step when working at low-frequency 
      (not chosen from the velocity model and frequency of the observed data  but from that of each stage)
- [ ] Create ``acousticfwi`` wrapper that can implement multi-stage acoustic FWI with time-space masking
      (and can specialize to any alternative case)
- [ ] Improve handling of user-defined source wavelets
- [ ] Create NonLinear operator template class to be used for both propagators and norms implementing basic operations such as
      sum, multiply, etc...
- [ ] Switch velocity units to always be km/s to avoid continously rescaling them inside the FWI codes