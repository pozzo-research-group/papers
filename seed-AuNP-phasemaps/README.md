## Differentiable Self-driving labs : Case studies on seed-mediated AuNP growth
This directory contains code for reproducing phasemap and retrosynthesis analysis presented in the case studies of differentiable self-driving labs paper.

Files and folders are organized as follows:

- `00_synthesis.ipynb` : Python code in a jupyter notebook format to run a pipetting protocol to synthesize AuNP in a well plate.

- `01_run.py` : Python code to run active learning campaign to fit composite model.

- `02_read_comps.ipynb` : Python code to read compositional data from an activephasemap run 

- `03_read_uvvis.ipynb` : Python code to read UV-Vis data from excel files saved from the spectrophotometer.

- `utils.py` : Some utility files to visualize UV-Vis spectra over the compositional space. 

- `retrosynthesis.py` : Python code to perform retrosynthesis of a target based on the trained differentiable model.

- `data`: Composition and Spectra data saved in `.npy` format to be used by the `activephasemap.simulators.UVVisExperiemnt` class.

- `opentrons` : Volumes of samples to be synthesized by OT2 pipetting-robot using the `00_synthesis.ipynb`.

- `plotting` : Contains the plotting data and Jupyter notebook for figure visualization.
    - `data` : plotting data of grid spectra, acquisiton function values, gradient values, mie scattering fits, and retrosynthesis runs.
    - `00-make_plots_data_for_paper.py` : Python code to create the `data` folder contents (`accuracies.pkl`, `acqf_data_*.npz`, `gradient_daya.npz`, `grid_data_*.npz`).
    - `01-make_paper_plots.ipynb` : Python code reproduce figures from the manuscript.
    - `02-plot_retrosynthesis.ipynb` : Python code plot the retrosynthesis figures from manuscript.
    - `03-plot_mie_fits.ipynb` : Python code to reproduce the Mie scattering fit plot in the supplementary document.

- `pretraining` : Contains Python code perform hyper-parameter tuning of NeuralProcess (NP) model and pre-training.
    - `ray_tune_np.py` : Python code to perform hyper-parameter tuning of NP model using `ray-tune` results are saved in `tune` folder and the best configuration parameters are stored in `best_config.json`.
    - `pretrain_bestconfig.py` : Python code to pre-train NP model based on the best configuration from ray tune. Plots to visualize model perform during the training are stored in `bestconfig` folder.
    - `helpers.py` : Plotting code to visualize the pre-training NP model.
    - `uvvis_data_npy` : UV-Vis spectroscopy data of [silver nanoparticles](https://github.com/pozzo-research-group/papers/blob/activephasemap-preprint/Silver%20Nanoplates/README.md) used for pre-training the NP model.
    - `*.sh` files : Shell scripts to submit multi-core CPU/GPU Python programs on a Slurm scheduler (UW Hyak specific but can be modified to be run with any slurm cluster). 

-  `mie`: Folder containing the code to fit UV-Vis spectra based on Mie scattering.
    - `mie.py` : PyTorch functions to compute extinction of a sphere and nanorod and fit a spectrum using stochastic gradient descent.
    - `fit.py` : PyTorch code to run fitting of experimentally collected or model generated data.