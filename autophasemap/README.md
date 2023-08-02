## Automatic Structure Phase Map (autophasemap) generation

This repository host the supporting code for the following paper.
"Metric geometry tools for automatic structure phase map generation" by Kiran Vaddi, Karen Li, and Lilo D Pozzo.

```bibtex
@article{autophasemap, 
	place={Cambridge}, 
 	title={Metric geometry tools for automatic structure phase map generation}, 
 	DOI={10.26434/chemrxiv-2022-3p4gx}, 
 	journal={ChemRxiv}, 
 	publisher={Cambridge Open Engage}, 
 	author={Vaddi, Kiran and Li, Karen and Pozzo, Lilo D}, year={2022}} 
 	This content is a preprint and has not been peer-reviewed.
```

<img src="./graphical_abstract.png" alt="Simple example of autophasemap with Gaussians"/>

Please see the INSTALL.md file for detailed instructions.

A simple example using synthetic Gaussian functions as an example can be found in [this notebok](expts/Gaussians/gaussian_peaks.ipynb). This example can be run on your local machine once the `autophasemap` package is installed using similar instructions as above.

## Guide to access data and notebooks for case studies in the paper

In the paper, there are three case studies (two with SAXS and one with XRD). The data for XRD has been kindly provided to us by [Dr. Aaron Gilad Kusne](https://www.nist.gov/people/aaron-gilad-kusne).
We have generated the experimental data ourselves via a combination of Federal grants acknowledged in the original paper.
We provide Python and Slurm scripts used to produce the results in `/expts/` folder and the visualization scripts with the converged data in `/postprocess/`.

1. [SAXS case study of pluronic with varying temperature](postprocess/P123_Temp/)
	- This data is generated for a pluronic system (PEO-PPO-PEO block copolymers) with temperature (0-90\deg C) and the weight fraction of the pluronic in aqueous solution.
	- This folder contains the guidelines to access and re-produce the plots (Figures 3, 4, 5).
	- Python scripts to reproduce the results on a cluster computer are provided in `/expts/OMIECS/PPBT_0_P123_Y_Temp.py` and the respective slurm batch script in `/expts/slurm_FePdGa.sh`
	- Jupyter notebook to code used to manually annotations can be accessed at `/postprocess/P123_Temp/manual/Indexing.ipynb` and the code to reproduce the phase diagram in Figure 4 of the manuscript in `/postprocess/P123_Temp/manual/manual_annotation.ipynb`
	
2. [XRD case study on a benchmark system](postprocess/FeGaPd)
	- This dataset is from the following [paper](https://pubs.aip.org/aip/rsi/article/80/10/103902/354187), kindly provided to us by [Dr. Aaron Gilad Kusne](https://www.nist.gov/people/aaron-gilad-kusne)
	- This folder contains the guidelines to access and re-produce the plots (Figure 6).
	- Python scripts to reproduce the results on a cluster computer are provided in `/expts/FeGaPd/FeGaPd_autophasemap.py` and the respective slurm batch script in `/expts/FeGaPd/slurm_blends.sh`	

	
3. [SAXS case study of polymer blends of pluronic and conjugated polymers](postprocess/WSCP_P123_NOpH)
	- This dataset is collected for a blend of P123 pluronic block-copolymer and a conjugated homopolymer poly(3-[potassium-4-butanoate]thiophene) (PPBT) co-dissolved in aqueous solutions. The design space consisted of two-dimensional weight fractions of the two polymers used. 
	- This folder contains the guidelines to access and re-produce the plots (Figure 7).
	- Python scripts to reproduce the results on a cluster computer are provided in `/expts/OMIECS/run_WSCP_p123_NOpH.py` and the respective slurm batch script in `/expts/slurm_blends.sh`

All the notebooks and Python scripts are annotated with comments to describe the nature of computation and usage. Feel free to open an issue if any of the files do not work as expected or if any content is unclear.

## Disclaimer

In our experiments with SAXS and XRD data obtained from typical high-throughput systems (data on the order of 100-1000), we observed that existing code to obtain the phase map is extremely slow, so we used parallel computing to speed up the process. 
We observed runtime on the average of 30-45 minutes with 16 cores working in parallel. We anticipate releasing an updated code that is more efficient to be run on laptop computers soon.   









