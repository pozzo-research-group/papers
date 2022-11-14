import numpy as np
from scipy import linalg
import pdb
import optimum_reparamN2 as orN2
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import cumtrapz
from collections import namedtuple
from pygsp import graphs
import time, os, traceback, shutil, warnings


class DiffusionMaps:
	def __init__(self, domain, k=8):
		""" Diffusion maps on graph structure data.

		Parameters:
		===========
		domain : numpy array of shape (n_samples, dim)
			Cartesian coordinates of the design space for graph construction
		k : int
			Number of nearest neighbors for k-NN graph

		Attributes:
		===========
		get_asymptotic_function  : Compute asymptote of a function by approximating the Laplace operator of the graph diffusion

		"""
		
		self.G = graphs.NNGraph(domain, NNtype='knn', k=8, 
		                   center=False, rescale=False
		                  )
		self.G.compute_fourier_basis()
    
	def get_asymptotic_function(self, s, num_freq=30):
		""" Compute asymptotic approximation of a function.
		
		Parameters:
		===========
		s : numpy array of shape (n_domain, )
			Function to be diffused
			
		num_freq : int
			Number of eigen values to truncate Laplace operator
				
		Returns:
		========
		s_norm : numpy array of shape (n_domain, )
			[0,1] scaled input function 
			
		s_hat : numpy array of shape (n_domain, )
			Eigen value approximationof s_norm 
			
		s_hat : numpy array of shape (n_domain, )

		"""
		
		s_norm = s/max(s)
		s_hat = self.G.gft(s_norm)
		s_hat_n = np.zeros(s_hat.shape)
		s_hat_n[:num_freq] = s_hat[:num_freq]
		s_tilda = self.G.igft(s_hat_n)
	    
		return s_norm, s_hat, s_tilda