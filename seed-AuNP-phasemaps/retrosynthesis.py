import os, shutil,json, glob
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from botorch.utils.sampling import draw_sobol_samples
torch.set_default_dtype(torch.double)

from activephasemap.models.np import NeuralProcess
from activephasemap.models.xgb import XGBoost
from activephasemap.utils import *
from funcshape.functions import Function, SRSF
import optimum_reparamN2 as orN2

"""
Retrosynthesis Optimization for Spectral Matching

This script performs retrosynthesis optimization to match a target spectrum 
using a combination of Neural Process (NP) models and XGBoost models. The optimization 
process involves creating a simulator, calculating spectral differences, and updating 
compositional parameters to minimize the difference between simulated and target spectra.

Workflow
--------
1. Load pre-trained Neural Process and XGBoost models.
2. Set up the target spectrum and the simulator.
3. Define loss functions for spectral matching:
   - Mean Squared Error (MSE)
   - Amplitude-Phase (AP) loss
4. Optimize the compositional parameters using LBFGS.
5. Save results and generate plots for visualization.

Global Parameters
-----------------
TRAINING_ITERATIONS : int
    Total number of optimization iterations for each restart.
NUM_RESTARTS : int
    Number of optimization restarts using Sobol sampling.
LEARNING_RATE : float
    Learning rate for the optimizer.
TARGET_SHAPE_ID : int
    Index of the target shape (0 for "sphere", 1 for "nanorod").
TARGET_SHAPES : list of str
    List of target shapes.
SAVE_DIR : str
    Directory to save results and plots.
DATA_DIR : str
    Directory containing pre-trained model files.
DESIGN_SPACE_DIM : int
    Dimensionality of the compositional design space.

Functions
---------
min_max_normalize(x)
    Normalize an input array to the range [0, 1] using min-max normalization.
amplitude_phase_distance(t_np, f1, f2, **kwargs)
    Compute amplitude and phase differences between two spectra.
mse_loss(y_pred)
    Calculate the mean squared error loss between simulated and target spectra.
ap_loss(y_pred, is_training=True)
    Calculate the amplitude-phase loss between simulated and target spectra.
closure()
    LBFGS optimizer closure function to compute gradients and update parameters.

Classes
-------
Simulator
    Wrapper for the Neural Process and XGBoost models to simulate spectra 
    given compositional parameters.

Usage
-----
Run the script to optimize compositions for matching the target spectrum. 
Results and plots will be saved to the specified `SAVE_DIR`.

"""

TRAINING_ITERATIONS = 100 # total iterations for each optimization
NUM_RESTARTS = 4 # number of optimization from random restarts
LEARNING_RATE = 1e-1
TARGET_SHAPE_ID = 1 # chose from [0 - "sphere", 1 - "nanorod"]

TARGET_SHAPES = ["sphere", "nanorod"]
print("Retrosynthesizing %s"%TARGET_SHAPES[TARGET_SHAPE_ID])
SAVE_DIR = "./plotting/data/retrosynthesis/%s/"%TARGET_SHAPES[TARGET_SHAPE_ID]
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)

DATA_DIR = './output/'
ITERATION = len(glob.glob(DATA_DIR+"comp_model_*.json"))
with open('./pretraining/best_config.json') as f:
    best_np_config = json.load(f)
N_LATENT = best_np_config["z_dim"]

# Load trained NP model for p(y|z)
np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
np_model.load_state_dict(torch.load(DATA_DIR+'np_model_%d.pt'%(ITERATION), map_location=device, weights_only=True))
np_model.train(False)

# Load trained composition to latent model for p(z|c)
xgb_model_args = {"objective": "reg:squarederror",
                  "max_depth": 3,
                  "eta": 0.1,
                  "eval_metric": "rmse"
                  }
comp_model = XGBoost(xgb_model_args)
comp_model.load(DATA_DIR+"comp_model_%d.json"%ITERATION)

design_space_bounds = [(0.0, 35.0), (0.0, 35.0)]
bounds = torch.tensor(design_space_bounds).transpose(-1, -2).to(device)
DESIGN_SPACE_DIM = len(design_space_bounds)

# Create a target spectrum
TARGETS_DIR = "./plotting/data/retrosynthesis/"
if TARGET_SHAPE_ID==0:
    target = np.load(TARGETS_DIR+"target_sphere.npz")
else:
    target = np.load(TARGETS_DIR+"target_nanorod.npz")
wav = target["x"]
n_domain = len(wav)
t_np = (wav-min(wav))/(max(wav)-min(wav))
xt = torch.from_numpy(t_np).to(device)
yt = torch.from_numpy(target["y"]).to(device)

def min_max_normalize(x):
    """
    Normalize an input array to the range [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape `(n_samples, n_features)`.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values scaled to the range [0, 1].

    Examples
    --------
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> min_max_normalize(x)
    tensor([[0.0000, 0.5000, 1.0000],
            [0.0000, 0.5000, 1.0000]], dtype=torch.float64)
    """
    min_x = x.min(dim=1).values 
    max_x = x.max(dim=1).values
    x_norm = (x - min_x[:,None])/((max_x-min_x)[:,None])
    
    return x_norm

class Simulator(torch.nn.Module):
    """
    Wrapper for Neural Process and XGBoost models to simulate spectra.

    Simulates spectra given compositional parameters by sampling latent variables 
    and evaluating the Neural Process model.

    Parameters
    ----------
    xt : torch.Tensor
        Normalized wavelengths (time) of shape `(n_wavelengths, )`.
    c2z : XGBoost
        Composition-to-latent model.
    z2y : NeuralProcess
        Latent-to-spectrum model.
    nz : int, optional
        Number of latent variable samples (default is 128).

    Methods
    -------
    forward(x)
        Simulate spectra for given compositional parameters.

    Examples
    --------
    >>> xt = torch.linspace(0, 1, 100)
    >>> simulator = Simulator(xt, comp_model, np_model)
    >>> compositions = torch.rand(10, 1, 2)
    >>> spectra = simulator(compositions)
    """
    def __init__(self, xt, c2z, z2y, nz=128):
        super().__init__()
        self.c_to_z = c2z 
        self.z_to_y = z2y
        self.t = xt
        self.nz = nz

    def forward(self, x):
        """
        Simulate spectra for given compositional parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input compositional parameters of shape `(n_samples, n_batches, n_features)`.

        Returns
        -------
        torch.Tensor
            Simulated spectra with mean and standard deviation, shape `(n_samples, n_batches, n_wavelengths, 2)`.

        Examples
        --------
        >>> compositions = torch.rand(10, 1, 2)
        >>> spectra = simulator.forward(compositions)
        >>> print(spectra.shape)
        torch.Size([10, 1, 100, 2])
        """
        z_mu, z_std = self.c_to_z.predict(x)
        nr, nb, dz = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, dz)
        time = self.t.repeat(self.nz*nr*nb, 1, 1).to(device)
        time = torch.swapaxes(time, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(time, z)

        mu = y_samples.view(self.nz, nr, nb, len(self.t), 1).mean(dim=0).squeeze()
        sigma = y_samples.view(self.nz, nr, nb, len(self.t), 1).std(dim=0).squeeze()

        return torch.stack((mu, sigma), dim=-1)


def amplitude_phase_distance(t_np, f1, f2, **kwargs):
    """
    Compute amplitude and phase differences between two spectra.

    Parameters
    ----------
    t_np : numpy.ndarray
        Normalized time (wavelength) vector.
    f1 : torch.Tensor
        First spectrum (reference), shape `(n_wavelengths, )`.
    f2 : torch.Tensor
        Second spectrum (target), shape `(n_wavelengths, )`.
    **kwargs : dict, optional
        Additional parameters:
        - `lambda` (float): Regularization parameter for phase reparameterization (default is 0.0).
        - `grid_dim` (int): Grid dimension for reparameterization (default is 7).

    Returns
    -------
    float
        Amplitude difference.
    float
        Phase difference.

    Examples
    --------
    >>> t_np = np.linspace(0, 1, 100)
    >>> f1 = torch.sin(torch.linspace(0, 2 * np.pi, 100))
    >>> f2 = torch.cos(torch.linspace(0, 2 * np.pi, 100))
    >>> amplitude, phase = amplitude_phase_distance(t_np, f1, f2)
    >>> print(amplitude, phase)
    """
    t_tensor = torch.tensor(t_np, dtype=f1.dtype, device=f2.device)
    f1_ = Function(t_tensor, f1.reshape(-1,1))
    f2_ = Function(t_tensor, f2.reshape(-1,1))
    q1, q2 = SRSF(f1_), SRSF(f2_)

    delta = q1.qx-q2.qx
    if (delta==0).all():
        amplitude, phase = 0.0, 0.0
    else:
        q1_np = q1.qx.clone().detach().cpu().squeeze().numpy()
        q2_np = q2.qx.clone().detach().cpu().squeeze().numpy()
        
        gamma = orN2.coptimum_reparam(np.ascontiguousarray(q1_np), 
                                      t_np,
                                      np.ascontiguousarray(q2_np), 
                                      kwargs.get("lambda", 0.0),
                                      kwargs.get("grid_dim", 7)
                                    )
        gamma = (t_np[-1] - t_np[0]) * gamma + t_np[0]
    gamma_tensor = torch.from_numpy(gamma).to(device)
    gamma_tensor = gamma_tensor.clone().detach().requires_grad_(True)
    warping = Function(t_tensor.squeeze(), gamma_tensor.reshape(-1,1))

    # Compute amplitude
    gam_dev = torch.abs(warping.derivative(warping.x))
    q_gamma = q2(warping.fx)
    y_amplitude = (q1.qx.squeeze() - (q_gamma.squeeze() * torch.sqrt(gam_dev).squeeze())) ** 2
    amplitude = torch.sqrt(torch.trapezoid(y_amplitude, q1.x))

    # Compute phase
    # we define p(\gamma) = \sqrt{\dot{\gamma(t)}} * f(t)
    p_gamma = torch.sqrt(gam_dev)*(f1_.fx)
    p_identity = torch.ones_like(gam_dev)*(f1_.fx)
    y_phase =  (p_gamma - p_identity) ** 2
    phase = torch.sqrt(torch.trapezoid(y_phase.squeeze(), t_tensor))

    return amplitude, phase

def mse_loss(y_pred):
    """
    Compute the mean squared error (MSE) loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted spectra of shape `(n_samples, n_wavelengths)`.

    Returns
    -------
    float
        Total MSE loss.
    torch.Tensor
        Per-sample MSE losses, shape `(n_samples, )`.

    Examples
    --------
    >>> y_pred = torch.rand(10, 100)
    >>> total_loss, per_sample_loss = mse_loss(y_pred)
    >>> print(total_loss, per_sample_loss.shape)
    """
    num_points, _ = y_pred.shape
    target = yt.repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(y_pred)

    loss = ((target_-mu_)**2).sum(dim=1)

    return loss.sum(), torch.tensor(loss, dtype=y_pred.dtype, device=y_pred.device)   

def ap_loss(y_pred, is_training=True):
    """
    Compute the amplitude-phase loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted spectra of shape `(n_samples, n_wavelengths)`.
    is_training : bool, optional
        Whether the loss is used during training (default is True).

    Returns
    -------
    float
        Total amplitude-phase loss.
    torch.Tensor
        Per-sample amplitude-phase losses, shape `(n_samples, )`.

    Examples
    --------
    >>> y_pred = torch.rand(10, 100)
    >>> total_loss, per_sample_loss = ap_loss(y_pred)
    >>> print(total_loss, per_sample_loss.shape)
    """
    alpha = 0.5
    num_points, _ = y_pred.shape
    target = yt.repeat(num_points, 1)
    target_ = min_max_normalize(target)
    mu_ = min_max_normalize(y_pred)
    loss = 0.0
    loss_values = []
    for i in range(num_points):
        amplitude, phase = amplitude_phase_distance(t_np, mu_[i,:], target_[i,:])
        dist = (1-alpha)*amplitude + (alpha)*phase
        if is_training:
            dist.backward(retain_graph=True)
        loss += dist 
        loss_values.append(dist.item())
    
    return loss, torch.tensor(loss_values, dtype=y_pred.dtype, device=y_pred.device)

def closure():
    global loss_values
    global loss
    global spectra

    lbfgs.zero_grad()
    spectra = sim(X)
    loss, loss_values = loss_fn(spectra[...,0])

    return loss

sim = Simulator(xt, comp_model, np_model).to(device)

# Initialize using random Sobol sequence sampling
X = draw_sobol_samples(bounds=bounds, n=NUM_RESTARTS, q=1).to(device)
X.requires_grad_(True)

lbfgs = torch.optim.LBFGS([X],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")

X_traj, loss_traj, spectra_traj = [], [], []
loss_fn = ap_loss

# run a basic optimization loop
counter = 0
for i in range(TRAINING_ITERATIONS):
    lbfgs.step(closure)
    # clamp values to the feasible set
    for j, (lb, ub) in enumerate(zip(*bounds)):
        X.data[..., j].clamp_(lb, ub) 
    # store the optimization trajectory
    # clone and detaching is importat to not meddle with the autograd
    X_traj.append(X.clone().detach())
    loss_traj.append(loss_values.clone().detach())
    spectra_traj.append(spectra.clone().detach())

    current_losses = loss_values.clone().detach()
    if i==0:
        best_losses = loss_values.clone().detach()
    elif torch.linalg.norm(best_losses-current_losses)<1e-2:
        counter += 1
    else:
        best_losses = loss_values.clone().detach()
        counter = 0
    
    if counter>(0.1*TRAINING_ITERATIONS):
        print("The error has not improved over 10% " + \
              "of total iterations\nEarly stopping at %d/%d"%(i, TRAINING_ITERATIONS)
        )
        break 
    
    if (i + 1) % 1 == 0:
        print(f"Iteration {i+1:>3}/{TRAINING_ITERATIONS:>3} - Loss: {loss.item():>4.3f}; dX: {X.grad.mean():>.2e}")

# Compute loss function on a grid for plotting
with torch.no_grad():
    grid_comps = get_twod_grid(15, bounds=bounds.cpu().numpy())
    grid_spectra = sim(torch.from_numpy(grid_comps).view(grid_comps.shape[0], 1, 2).to(device))
    _, grid_loss = loss_fn(grid_spectra[...,0], is_training=False)
    print(grid_loss.shape)

with torch.no_grad():
    spectra_optim = sim(X_traj[-1])
    _, loss_optim = loss_fn(spectra_optim[...,0], is_training=False)
    print("Optimized composition : ", X_traj[-1].squeeze()[torch.argmin(loss_optim)])
    X_traj = torch.stack(X_traj, dim=1).squeeze()
    for i in range(NUM_RESTARTS):
        fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
        fig.subplots_adjust(wspace=0.5)
        mu = spectra_optim[i,:,0].cpu().squeeze().numpy()
        sigma = spectra_optim[i,:,1].cpu().squeeze().numpy()
        axs[0].plot(target["x"], target["y"], label="Target", color="k", ls='--')
        ax2 = axs[0].twinx()
        ax2.plot(target["x"], mu, label="Best Estimate", color="k")
        ax2.fill_between(target["x"], mu-sigma, mu+sigma,  
                        color='grey', alpha=0.5, label="Uncertainity"
                        )
        axs[0].set_title("Loss : %.2f"%loss_optim[i].item())

        traj = X_traj.cpu().numpy()[i,:,:]
        axs[1].tricontourf(grid_comps[:,0], 
                           grid_comps[:,1], 
                           grid_loss.detach().cpu().numpy(), 
                           cmap="binary",
                           )
        axs[1].plot(traj[:,0], traj[:,1],
                    lw=2,c='tab:red', label="Trajectory"
                    )
        axs[1].scatter(traj[0,0], traj[0,1],
                       s=100,c='tab:red',marker='.',
                       zorder=10,lw=2, label="Initial"
                       )
        axs[1].scatter(traj[-1,0], traj[-1,1],
                       s=100,c='tab:red',marker='+',
                       zorder=10,lw=2, label="Final"
                       )
        # axs[1].set_xlim(*design_space_bounds[0])
        # axs[1].set_ylim(*design_space_bounds[1])
        axs[1].legend()
        plt.savefig(SAVE_DIR+"comparision_%d.png"%i)
        plt.close()

# create result object and save
optim_result = {"X_traj" : X_traj,
                "spectra_traj" : torch.stack(spectra_traj, dim=1).squeeze(),
                "loss" : torch.stack(loss_traj, dim=1).squeeze(),
                "spectra" : spectra_optim,
                "target_y" : yt,
                "target_x" : xt,
                "grid_loss" : grid_loss,
                "grid_comps" : grid_comps
                }
torch.save(optim_result, SAVE_DIR+"optim_traj.pkl")