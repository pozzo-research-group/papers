import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd 
import pdb, argparse, json, glob, pickle, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from activephasemap.models.np import NeuralProcess
from activephasemap.simulators import UVVisExperiment
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *
from apdist.distances import AmplitudePhaseDistance as dist
from apdist.geometry import SquareRootSlopeFramework as SRSF
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(
                    prog='Train emulator of gold nanoparticle synthesis',
                    description='Perform a single iteration of active learning of Models 1 and 2',
                    epilog='...')
parser.add_argument('iteration', metavar='i', type=int, help='iterations number for the campaign')
args = parser.parse_args()
ITERATION = args.iteration # specify the current itereation number

DATA_DIR = "./output/"
with open('/mmfs1/home/kiranvad/cheme-kiranvad/activephasemap/activephasemap/pretrained/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]
TOTAL_ITERATIONS  = len(glob.glob(DATA_DIR+"comp_model_*.json"))

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))

# Load trained composition to latent model for p(z|c)
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]

# Create the experiment class to load all teh data
expt = UVVisExperiment(design_space_bounds, "./data/")
expt.read_iter_data(ITERATION)
expt.generate(use_spline=True)
bounds_np = expt.bounds.cpu().numpy()

""" 1. Create grid data """
def sample_grid(n_grid_spacing):
    grid_comps = get_twod_grid(n_grid_spacing, bounds_np)
    grid_spectra = np.zeros((grid_comps.shape[0], len(expt.t), 2))
    with torch.no_grad():
        for i, ci in enumerate(grid_comps):
            mu, sigma = from_comp_to_spectrum(expt.t, ci, comp_model, np_model)
            mu_ = mu.cpu().squeeze().numpy()
            sigma_ = sigma.cpu().squeeze().numpy()
            grid_spectra[i, :, 0] = mu_ 
            grid_spectra[i, :, 1] = sigma_

    return grid_comps, grid_spectra

grid_comps, grid_spectra = sample_grid(10)
np.savez("./paper/grid_data_10_%d.npz"%ITERATION, comps=grid_comps, spectra=grid_spectra)
del grid_comps, grid_spectra

if ITERATION==TOTAL_ITERATIONS:
    grid_comps, grid_spectra = sample_grid(20)
    np.savez("./paper/grid_data_20.npz", comps=grid_comps, spectra=grid_spectra)
    del grid_comps, grid_spectra

    grid_comps, grid_spectra = sample_grid(30)
    np.savez("./paper/grid_data_30.npz", comps=grid_comps, spectra=grid_spectra)
    del grid_comps, grid_spectra

""" 2. Create acqusition function data """

acqf = XGBUncertainity(expt, expt.bounds, np_model, comp_model)
C_grid = get_twod_grid(15, bounds_np)
with torch.no_grad():
    acq_values = acqf(torch.tensor(C_grid).reshape(len(C_grid),1,2).to(device)).squeeze().cpu().numpy()

np.savez("./paper/acqf_data_%d.npz"%ITERATION, comps=C_grid, values=acq_values)
del acq_values

""" 3. Create data for train and test errors """
def load_models_from_iteration(i):
    expt = UVVisExperiment(design_space_bounds, "./data/")
    expt.read_iter_data(i)
    expt.generate(use_spline=True)

    # Load trained NP model for p(y|z)
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(i), map_location=device, weights_only=True))

    # Load trained composition to latent model for p(z|c)
    comp_model = XGBoost(xgb_model_args)
    comp_model.load(DATA_DIR+"comp_model_%d.json"%i)

    return expt, comp_model, np_model

def min_max_normalize(x):
    min_x = x.min(dim=1).values 
    max_x = x.max(dim=1).values
    x_norm = (x - min_x[:,None])/((max_x-min_x)[:,None])
    
    return x_norm

def smoothen_and_normalize(y):
    y_hat = gaussian_filter(y,  sigma=1.0)
    y_hat_norm =  (y_hat-min(y_hat))/(max(y_hat)-min(y_hat))

    return y_hat_norm

def weighted_amplitude_phase(x, y_ref, y_query, **kwargs):
    srsf = SRSF(x)
    q_ref = srsf.to_srsf(smoothen_and_normalize(y_ref))
    q_query = srsf.to_srsf(smoothen_and_normalize(y_query))
    gamma = srsf.get_gamma(q_ref, q_query, **kwargs)

    delta = q_ref-q_query
    if delta.sum() == 0:
        dist = 0
    else:
        gam_dev = np.gradient(gamma, srsf.time)
        q_gamma = np.interp(gamma, srsf.time, q_query)
        y_amplitude = (q_ref - (q_gamma * np.sqrt(gam_dev))) ** 2

        amplitude = np.sqrt(np.trapz(y_amplitude, srsf.time))

        p_gamma = np.sqrt(gam_dev)*y_ref # we define p(\gamma) = \sqrt{\dot{\gamma(t)}} * f(t)
        p_identity = np.ones_like(gam_dev)*y_ref
        y_phase =  (p_gamma - p_identity) ** 2

        phase = np.sqrt(np.trapz(y_phase, srsf.time))

    return amplitude, phase

@torch.no_grad()
def get_accuracy(comps, domain, spectra, comp_model, np_model):
    loss = []
    optim_kwargs = {"optim":"DP", "grid_dim":10}
    for i in range(comps.shape[0]):
        mu_i, _ = from_comp_to_spectrum(domain, comps[i,:], comp_model, np_model)
        mu_i_np = mu_i.cpu().squeeze().numpy()
        amplitude, phase = weighted_amplitude_phase(domain, spectra[i,:], mu_i_np)
        loss.append(0.5*(amplitude+phase))

    return np.asarray(loss)

def get_accuraciy_plot_data():
    accuracies = {}
    for i in range(1,TOTAL_ITERATIONS+1):
        expt, comp_model, np_model = load_models_from_iteration(i)
        train_accuracy = get_accuracy(expt.comps.astype(np.double), 
                                      expt.t, 
                                      expt.spectra_normalized, 
                                      comp_model, 
                                      np_model
                                    )
        if not i==TOTAL_ITERATIONS:
            next_comps = np.load("./data/comps_%d.npy"%(i)).astype(np.double)
            next_spectra = np.load("./data/spectra_%d.npy"%(i))
            wav = np.load("./data/wav.npy")
            next_time = (wav-min(wav))/(max(wav)-min(wav))

            test_accuracy =  get_accuracy(next_comps, 
                                          next_time, 
                                          next_spectra, 
                                          comp_model, 
                                          np_model
                                          )
            print("Iteration %d : Train error : %2.4f \t Test error : %2.4f"%(i, train_accuracy.mean(), test_accuracy.mean()))
        else:
            test_accuracy = np.nan
            print("Iteration %d : Train error : %2.4f"%(i, train_accuracy.mean()))

        accuracies[i] = {"train": train_accuracy, "test": test_accuracy}
        

    with open("./paper/accuracies.pkl", 'wb') as handle:
        pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return 

if ITERATION==TOTAL_ITERATIONS:
    get_accuraciy_plot_data()
else:
    print("Total number of iterations %d is higher than current iteration run %d"%(TOTAL_ITERATIONS, ITERATION))