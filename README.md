# Single Qubit Gates Beyond the RWA

The manuscript that this repository can be found on arXiv: [arXiv:2503.08238](https://doi.org/10.48550/arXiv.2503.08238).

## Getting started

Clone the repo to get started. This repository has been tested using python 3.11.0. This repo will definitely not work for python versions outside 3.8-3.11 due to the numpy version being pinned to <2.0. It is recommened to create a new virtual environment and install the required python packages there using:

```
pip install -r requirements.txt
```

## Files in this repository

This repository contains all scripts used for the project 'Single Qubit Gates Beyond the RWA'. It contains five interesting scripts:

1. `numerical-simulations.ipynb` contains almost all of the numerical simulations and calculations and the corresponding figures.
2. `main.py` and `exec.sh` are used to run the remaining numerical simulations on the HPC Delft Blue.
3. `data-analysis.ipynb` contains all the scripts that analyse measurement data.
4. `sympy-integrals.py` contains a script to symbolically solve expressions for the ideal pulse parameters.
5. `detuning-validation.ipynb` validates some of the expressions in the first-order Magnus approximation.

Below follows a more in-depth description of what goes on in each file.


### numerical-simulations.ipynb
One of the most relevant things actually happens all the way at the bottom of the file. Here, you can load datasets that you want and recreate figures from the paper. Here, we describe what goes on in the script chronologically:

* We start by initializing the script: loading modules and initializing the parameters.
* The first "simulation" consists of simulating the wavefunctions and potential of the fluxonium, which is figure 1 of the paper.
* We then perform all the simulations for the zeroth order Magnus approximation, corresponding to figures 2 and 3 of the paper.
* We then perform the simulations for the first order Magnus approximation, which is figure 4. These simulations can take quite some time, depending on how many cores you have available for multiprocessing.
* Then, we simulate how the fluxonium can be modelled as a TLS using time-dependent modulation of the drive frequency. This corresponds to figure 6 and the figure in appendix C.
* Next are the simulations for figure 5. We optimize the drive strength as a function of $\lambda$ and $\Delta$ and make heatmaps of the resulting error.
* Finally, we perform all the simulations for the experiments. First, we use the time evolution operators generated in `main.py` to numerically simulate the phase error pseudo-identity circuits for varying parameters. Secondly, we simulate the error budgets.
* The figures for the paper are plotted at the end of the script. Existing numerical data can be loaded all the way at the end of the script.


### main.py and exec.sh
`main.py` consists of a small script that simulates time evolution operators for a specified parameter range. Since these simulations can take very long, we run them on the HPC of the TU Delft. `exec.sh` is a small script that submits a job to the queue on this HPC.


### data-analysis.ipynb
All the experimental data is analyzed in this file, and the final experimental figures are produced. Specifically, this file contains the following:

* Initialization of the script.
* Calculation of the residual ZZ rate.
* Calculation of the average T1 and T2E times.
* Analysis of all RB and PRB experiments.
* Plotting of error budgets for figure 12.
* Plotting of experimental and numerical heatmaps for figure 11.
* Plotting of figure 7.
* Plotting of figure 10.


### sympy-integrals.py
A big part of this work is being able to analytically calculate pulse parameters. However, the expressions we obtain are complicated series-expressions. To actually evaluate these expressions we use sympy to compute the series-expressions up to a specified order. Especially for the first-order terms, this is computationally very expensive. Therefore, we generate all these expressions once in this file and store them in a binary file using pickle so they can be used on demand in all the other scripts. Since it is very bad practice to *unpickle* data that you don't trust, these binary files are not provided and should be generated by the user. They can also be provided on request.


### detuning-validation.ipynb
In appendix C of the paper, general forms are provided for the double integrals in the first-order Magnus approximation. A question that the authors had as well was: "Are these expressions correct?" That's why this file exists: it calculates these integrals numerically and using the expressions in appendix C to verify that they are correct.


## Numerical and experimental data
The numerical and experimental data is not included in this repository, but is publicly available through [https://doi.org/10.4121/4e5a545d-436b-4d59-86d2-61506ccaf3cf](https://doi.org/10.4121/4e5a545d-436b-4d59-86d2-61506ccaf3cf). Below is a table linking the numerical datasets to specific figures in the paper:

| Name | Dataset | Figure |
|------|---------|--------|
|figure_data/20241119-140354/|Zeroth-order Magnus expansion optimization|Figures 2 and 3|
|figure_data/20250117-081322/|Optimization for two-level effective Hamiltonian|Figure 6 and 8|
|figure_data/20250130-175226/|Lambda vs Delta heatmap|Figure 5|
|figure_data/20250203-080400/|Optimization for first-order Magnus expansion and full time evolution|Figure 4|
|extended_data/20241201-152543/|Drive strength optimization for experimental parameters|Used for heatmap simulations|
|extended_data/20241202-002816/|Unitary time evolutions generated on the HPC|Used to simulate phase-error heatmap experiments|
|extended_data/20250123-194712/|Error budgets for protocol 2 and 3|Figures 7 and 12|
|extended_data/20250124-131533/|Simulated phase-error heatmaps|Figures 7 and 11|
|extended_data/20250124-171107/|Error budget for protocol 1|Figures 7 and 12|
|extended_data/20250126-133937/|Error budget for protocol 4|Figures 7 and 12|


## Author & Contact
Martijn Zwanenburg \
m.f.s.zwanenburg@tudelft.nl
