import torch
import numpy as np
torch.set_default_dtype(torch.double)
from botorch.utils.sampling import draw_sobol_samples
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline 
import time, datetime

def gold_dielectric_function(x):
    """
    Compute the complex dielectric function of gold at given wavelengths.

    The dielectric function is calculated using interpolated real (`n_real`) 
    and imaginary (`n_imag`) refractive indices.

    Parameters
    ----------
    x : torch.Tensor
        Wavelengths in nanometers, shape `(n_samples, )`.

    Returns
    -------
    torch.Tensor
        Real part of the dielectric function, shape `(n_samples, )`.
    torch.Tensor
        Imaginary part of the dielectric function, shape `(n_samples, )`.

    Notes
    -----
    The refractive index data is based on experimental values for gold. 
    Interpolation is performed using natural cubic splines.

    Examples
    --------
    >>> wavelengths = torch.tensor([500.0, 600.0, 700.0])
    >>> e1, e2 = gold_dielectric_function(wavelengths)
    >>> print(e1, e2)
    """

    wl = 1239.19/torch.tensor([0.1,0.2,0.3,0.4,0.5,0.5450000,0.5910000,0.6360000,0.64,0.77,0.89,
                           1.02,1.14,1.26,1.39,1.51,1.64,1.76,1.88,2.01,2.13,2.26,2.38,2.50,
                           2.63,2.75,2.88,3.00,3.12,3.25,3.37,3.50,3.62,3.74,3.87,3.99,4.12,
                           4.24,4.36,4.49,4.61,4.74,4.86,4.98,5.11,5.23,5.36,5.48,5.60])
    n_real = torch.tensor([25.17233,7.60352,3.53258,2.02586,1.299091,1.097350,0.9394755,
                       0.8141369,0.92,0.56,0.43,0.35,0.27,0.22,0.17,0.16,0.14,0.13,0.14,
                       0.21,0.29,0.43,0.62,1.04,1.31,1.38,1.45,1.46,1.47,1.46,1.48,1.50,
                       1.48,1.48,1.54,1.53,1.53,1.49,1.47,1.43,1.38,1.35,1.33,1.33,
                       1.32,1.32,1.30,1.31,1.30])
    n_imag = torch.tensor([77.92804,43.34848,29.52751,22.25181,17.77038,16.24777,
                       14.94747,13.82771,13.78,11.21,9.519,8.145,7.15,6.35,5.66,
                       5.08,4.542,4.103,3.697,3.272,2.863,2.455,2.081,1.833,1.849,
                       1.914,1.948,1.958,1.952,1.933,1.895,1.866,1.871,1.883,1.898,
                       1.893,1.889,1.878,1.869,1.847,1.803,1.749,1.688,1.631,1.577,
                       1.536,1.497,1.460,1.427])
    n_r = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.flip(wl, [0]), torch.flip(n_real, [0]).reshape(-1,1)))
    n_i = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.flip(wl, [0]), torch.flip(n_imag, [0]).reshape(-1,1)))
    eps = (n_r.evaluate(x) + 1j*n_i.evaluate(x))**2
   
    return eps.real, eps.imag

def LogNormal(mean, std):
    """
    Create a log-normal distribution based on mean and standard deviation.

    Parameters
    ----------
    mean : float
        Mean of the log-normal distribution.
    std : float
        Standard deviation of the log-normal distribution.

    Returns
    -------
    torch.distributions.LogNormal
        Log-normal distribution object.

    Examples
    --------
    >>> dist = LogNormal(mean=1.5, std=0.5)
    >>> samples = dist.sample((10,))
    >>> print(samples)
    """

    # Compute parameters of the underlying Normal distribution
    mu = torch.log(mean**2 / torch.sqrt(std**2 + mean**2))
    sigma = torch.sqrt(torch.log(1 + (std**2 / mean**2)))

    dist = torch.distributions.LogNormal(mu, sigma)

    return dist

def sphere_extinction(wavelength, em):
    """
    Compute the extinction coefficient for a gold nanosphere in a medium.

    Parameters
    ----------
    wavelength : torch.Tensor
        Wavelengths in nanometers, shape `(n_samples, )`.
    em : float
        Dielectric constant of the surrounding medium.

    Returns
    -------
    torch.Tensor
        Extinction coefficient at the given wavelengths, shape `(n_samples, )`.

    Examples
    --------
    >>> wavelengths = torch.tensor([500.0, 600.0, 700.0])
    >>> gamma = sphere_extinction(wavelengths, em=1.33)
    >>> print(gamma)
    """

    e1, e2 = gold_dielectric_function(wavelength)

    factor = (18 * np.pi * (em**1.5))/wavelength
    scale = e2/((e1 + 2*em)**2 + e2**2)

    gamma = scale*factor

    return gamma

def nanorod_extinction(wavelength, aspect_ratio_mu, aspect_ratio_sigma, em):
    """
    Compute the extinction coefficient for a gold nanorod in a medium.

    Parameters
    ----------
    wavelength : torch.Tensor
        Wavelengths in nanometers, shape `(n_samples, )`.
    aspect_ratio_mu : float
        Mean aspect ratio of the nanorods.
    aspect_ratio_sigma : float
        Standard deviation of the aspect ratio.
    em : float
        Dielectric constant of the surrounding medium.

    Returns
    -------
    torch.Tensor
        Extinction coefficient at the given wavelengths, shape `(n_samples, )`.

    Examples
    --------
    >>> wavelengths = torch.tensor([500.0, 600.0, 700.0])
    >>> ext = nanorod_extinction(wavelengths, aspect_ratio_mu=3.0, aspect_ratio_sigma=0.5, em=1.33)
    >>> print(ext)
    """

    aspect_ratio = LogNormal(aspect_ratio_mu, aspect_ratio_sigma*aspect_ratio_mu)
    n_samples = 4096
    ar_samples = aspect_ratio.rsample((n_samples, ))
    flags = ar_samples>1.0

    e1, e2 = gold_dielectric_function(wavelength)
    scale = (2 * np.pi * (em**1.5)) / (3 * wavelength)
    
    e = torch.sqrt(1 - (1 / ar_samples[flags])**2)
    PA = ( (1 - e**2)/ e**2 ) * ( ((1/ (2 * e)) * torch.log((1 + e) / (1 - e))) - 1 )
    PB = 0.5 * (1 - PA) 
    PC = 0.5 * (1 - PA)
    gamma = []
    for P in [PA, PB, PC]:
        gj = (e2/ (P**2) )/( ( e1 + (((1-P)/P)*em) )**2 + e2**2)
        gamma.append(gj)

    ext = scale*(torch.stack(gamma).mean(dim=1).sum())

    return ext      

def fit_mie_scattering(objective, design_space_bounds, **kwargs):
    """
    Fit parameters to minimize an objective function using Mie scattering.

    Parameters
    ----------
    objective : callable
        Objective function to be minimized. It should accept a tensor of parameters 
        and return a scalar loss value.
    design_space_bounds : list of tuple
        Bounds for the parameters, specified as a list of `(min, max)` tuples for each dimension.
    **kwargs : dict, optional
        Additional hyperparameters for the optimization:
        - `n_iterations` (int): Maximum number of iterations (default: 100).
        - `n_restarts` (int): Number of random restarts for optimization (default: 100).
        - `epsilon` (float): Error threshold for early stopping (default: 0.1).
        - `lr` (float): Learning rate for the optimizer (default: 0.01).

    Returns
    -------
    torch.Tensor
        Optimized parameters, shape `(n_dimensions, )`.
    float
        Best objective function value (error) achieved.

    Notes
    -----
    - Sobol sampling is used for initialization of restarts.
    - Optimization is performed using Adam with projected gradient descent.

    Examples
    --------
    >>> def objective(params):
    ...     return torch.sum(params**2)
    >>> bounds = [(0, 1), (0, 2)]
    >>> best_params, best_error = fit_mie_scattering(objective, bounds)
    >>> print(best_params, best_error)
    """

    n_iterations = kwargs.get("n_iterations", 100)
    n_restarts = kwargs.get("n_restarts", 100)
    epsilon = kwargs.get("epsilon", 0.1)
    lr = kwargs.get("lr", 0.01)
    start = time.time()

    bounds = torch.tensor(design_space_bounds).transpose(-1, -2)
    samples = draw_sobol_samples(bounds=bounds, n=n_restarts, q=1).view(n_restarts, len(design_space_bounds))

    pruning_errors = []
    for i in range(n_restarts):
        X = samples[i,...].clone().detach()
        loss = objective(X)
        pruning_errors.append(loss)

    pruning_errors = torch.tensor(pruning_errors)
    best_starting_id = torch.argmin(pruning_errors)

    X = samples[best_starting_id,...].clone().detach()
    X.requires_grad_(True)
    optimizer = torch.optim.Adam([X], lr=lr)
    
    best_error = np.inf
    for j in range(n_iterations):
        optimizer.zero_grad()
        loss = objective(X)
        loss.backward()
        optimizer.step()

        # clamp values to the feasible set
        for k, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., k].clamp_(lb, ub) 

        if (100*j/n_iterations)%10==0:
            print(f"Iteration {j+1:>3}/{n_iterations:>3} - Loss: {loss.item():>4.3f}; dX: {X.grad.squeeze()}")

        if loss.item()<epsilon:
            best_error = loss.item()
            best_X = X.clone().detach()
            print("Error threshold of %.2f reached."%best_error)
            break
        elif loss.item()<best_error:
            best_error = loss.item()
            best_X = X.clone().detach()

    end = time.time()
    time_str =  str(datetime.timedelta(seconds=end-start)) 

    print("Best parameters : ", best_X.squeeze(), 
          "\nBest Error : ", best_error,
          "\nTotal time : ", time_str
          )

    return best_X, best_error

    