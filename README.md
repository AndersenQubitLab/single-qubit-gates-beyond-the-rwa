# Single Qubit Gates Beyond the RWA
This repository contains all scripts used for the project 'Single Qubit Gates Beyond the RWA'. It contains three interesting scripts:

1. `paper-figures.ipynb` contains almost all of the numerical simulations and calculations and the corresponding figures.
2. `main.py` and `exec.sh` are used to run the remaining numerical simulations on the HPC Delft Blue.
3. `data-analysis.ipynb` contains all the scripts that analyse measurement data.

Below follows a more in-depth description of what goes on in each file.


# paper-figures.ipynb
One of the most relevant things actually happens all the way at the bottom of the file. Here, you can load datasets that you want, and recreate figures from the paper. Here, we describe what goes on in the script chronologically:

* We start by initializing the script: loading modules and initializing the parameters.
* The first "simulation" consists of simulating the wavefunctions and potential of the fluxonium, which is figure 1 of the paper.
* We then perform all the simulations for the zeroth order Magnus approximation, corresponding to figures 2 and 3 of the paper.
* We then perform the simulations for the first order Magnus approximation, which is figure 4. This simulations can take quite some time, depending on how many cores you have available for multiprocessing.
* Then, we simulate how the Fluxonium can be modelled as a TLS using phase ramping. This corresponds to figure 5 and a figure in the appendix.
* Finally, we perform the simulations for the experiments. Specifically, here we optimize the drive strength for varying gate durations. We also analyse the simulation data for the heatmaps. This data is not generated in this script, but is generated on the HPC using `main.py` and `exec.sh`.


# Author & Contact
Martijn Zwanenburg
m.f.s.zwanenburg@tudelft.nl