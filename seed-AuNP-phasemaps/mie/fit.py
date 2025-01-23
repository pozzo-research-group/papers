import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.double)

import numpy as np
import matplotlib.pyplot as plt

import os, pdb, sys, shutil
sys.path.append('./')
from mie import *

from activephasemap.simulators import UVVisExperiment
from scipy.signal import find_peaks

TESTING = False
TEST_SAMPLE_ID = 30
FIT_EXPT = False 

if not TESTING:
    SAVE_DIR = "./results/"
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

DATA_DIR = "../"
ITERATION = 14
grid_data = np.load("../plotting/data/grid_data_10_%d.npz"%ITERATION)
grid_spectra = grid_data["spectra"]

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]
expt = UVVisExperiment(design_space_bounds, "../data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
wavelengths = torch.from_numpy(expt.wl)

if FIT_EXPT:
    comps = expt.comps
    spectra = expt.spectra_normalized
else:
    comps = grid_data["comps"]
    spectra = grid_spectra[:,:,0]

def normalize(x):
    return (x-min(x))/(1e-3+ max(x) - min(x))

def featurize(x,y):
    "Use peak locations to determine morphology"
    peaks, _ = find_peaks(y, prominence=0.01, width=0.3)
    if len(peaks)==0:
        shape = 0
    elif (x[peaks]>600).any():
        shape = 2
    elif not (x[peaks]>600).any():
        shape = 1
        
    return shape

def gaussian_filter(input_tensor, sigma=1.0, truncate=4.0):
    """
    Apply a Gaussian filter to a 1D PyTorch tensor.

    Parameters:
    - input_tensor: torch.Tensor, 1D input tensor.
    - sigma: float, standard deviation for the Gaussian kernel.
    - truncate: float, truncate the kernel at this many standard deviations.

    Returns:
    - torch.Tensor, filtered tensor.
    """
    # Create the Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize the kernel

    # Reshape kernel for 1D convolution
    kernel = kernel.view(1, 1, -1)

    # Add batch and channel dimensions to the input tensor
    input_tensor = input_tensor.view(1, 1, -1)

    # Pad the input tensor
    pad_width = (radius, radius)
    input_tensor = F.pad(input_tensor, pad_width, mode='reflect')

    # Perform the convolution
    filtered_tensor = F.conv1d(input_tensor, kernel).squeeze()

    return filtered_tensor

def simulate_spectra(x):
    s_sphere = torch.zeros_like(wavelengths)
    s_nanorod = torch.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        s_sphere[i] = sphere_extinction(wl, x[0])
        s_nanorod[i] = nanorod_extinction(wl, *x[1:-1])

    s_nanorod = gaussian_filter(s_nanorod)
    s_sphere = gaussian_filter(s_sphere)
    s_query = x[-1]*normalize(s_sphere)+(1-x[-1])*normalize(s_nanorod)
               
    return s_query


parameters_bounds = [(1.0, 10), # dieletric constant for sphereical medium
                       (1.1, 5.0), # nanorod aspect ratio (mu)
                       (0.001, 0.4), # nanorod aspect ratio (sigma)
                       (1.0, 10.0), # dieletric constant for nanorod medium
                       (0.0, 1.0), # mixed model weights
                       ]


for sample_id in range(comps.shape[0]):
    if TESTING:
        sample_id = TEST_SAMPLE_ID
    target_spectra = torch.from_numpy(spectra[sample_id,:])
    feats = featurize(expt.wl, target_spectra)

    def objective(x):
        s_query = simulate_spectra(x)
        error = (s_query-normalize(target_spectra))**2
        return error.sum()

    if feats in [0]:
        print("Skipping %d at composition : "%sample_id, comps[sample_id,:])
        continue

    print("Fitting %d at composition : "%sample_id, comps[sample_id,:], " with features %d"%feats)
    fit_kwargs = {"n_iterations": 250, "n_restarts": 250, "epsilon": 0.1, "lr":0.01}
    best_X, best_error = fit_mie_scattering(objective, parameters_bounds, **fit_kwargs)

    spectra_optimized = simulate_spectra(best_X)

    if not TESTING:
        np.savez(SAVE_DIR+"res_%d.npz"%sample_id,
                feats = feats,
                best_X = best_X.numpy(), 
                best_error = best_error,
                target = normalize(target_spectra.detach().numpy()),
                optimized = spectra_optimized.detach().numpy()
                )

    fig, ax = plt.subplots()
    ax.plot(wavelengths, 
            normalize(target_spectra.detach().numpy()), 
            color="k", 
            label="Target"
            )
    ax.plot(wavelengths, 
            spectra_optimized.detach().numpy(), 
            color="k", 
            ls="--",
            label="Optimized"
            )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction Coefficient")
    ax.set_title("Error : %.2f"%best_error)
    ax.legend()
    if TESTING:
        plt.savefig("fit.png")
        plt.close()
        break
    else:
        plt.savefig(SAVE_DIR + "fit_%d.png"%sample_id)
        plt.close()