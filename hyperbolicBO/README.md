# On learning material synthesis hierarchies using shape matching on function spaces
This repository supplements the following papers
> On learning material synthesis hierarchies using shape matching on function spaces Kiran Vaddi, Huat T Chiang and Lilo D Pozzo

We recommend installing the package in a editable format using the `install.sh` file in a Linux command line or using the instructions with-in the `install.sh` in a python command line.

## The three case studies are in the following files

1. [Learning hierarchies from Gaussian functions](gaussians.py)
2. [Learning hierarchies from numerical simulated spectra of nanorods](nanorods.py)

To run the case studies use the following steps as a guideline:
1. Activate the python virtual environment using `source env/bin/activate`
2. Run the case studies using the template : `python case_study_file.py`

## Known issues
* sometimes, you may get the dreaded 'NotPSDError' from botorch/GPyTorch. We typically fix this by adjusting the noise prior on the likelihood in `gp_regression.py` file in the borth under MIN_INFERRED_NOISE_LEVEL = 5e-3.
see this [link](https://github.com/pytorch/botorch/issues/179#issuecomment-756462566) for more details.
