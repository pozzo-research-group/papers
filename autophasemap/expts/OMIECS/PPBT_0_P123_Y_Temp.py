#!/usr/bin/env python
# coding: utf-8

""" Compute phase map given a set of spectra and required number of phases

This file performs elastic kmeans algorithm to learn templates to identify
phase maps given a set of spectra

To run in this file, change the following:
1. Identify a directory where to store data (see SAVE_DIR in #23)
2. Specify number of clusters (see n_clusters in #31)
3. Change the rows and columns of subplots in #32

"""

import numpy as np
import matplotlib.pyplot as plt
import ray
import time, os, traceback, shutil, warnings, pickle
from scipy.signal import savgol_filter
from autophasemap import compute_elastic_kmeans, plot_clusters, compute_BIC

SAVE_DIR = './output/P123_Temp/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)


# Specify variables
N_CLUSTERS = 8
PLOT_ROWS , PLOT_COLS = 2, 4
MAX_ITER = 200
VERBOSE = 3

if not "ip_head" in os.environ:
    ray.init()
else:
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"])
         
num_nodes = len(ray.nodes())
print('Total number of nodes are {}'.format(num_nodes))
print('Avaiable CPUS : ', ray.available_resources()['CPU'])

class DataSet:
    def __init__(self, C, q, Iq, N, n_domain=200):
        """ Dataset object for autophasemap with SAXS.

        Parameters:
        ===========
        C : numpy array of shape (N, dim)
            Cartesian coordinates of the design space
        q : numpy array of shape (n_domain, )
            q-vector sampling 
        Iq : numpy array of shape (N, n_domain)
            Intensity as a function of q
        N : int
            Number of samples 
        n_domain : int
            Number of discrete samples of the domain q      
        """
        self.n_domain = n_domain
        self.t = np.linspace(0,1, num=self.n_domain)
        self.N = N
        self.Iq = Iq
        self.C = C
        self.q = q
        
    def generate(self, process=None):
        """ Generate the data into an acceptable form by autophasemap.

        Parameters:
        ===========
        process : string
            "normalize" : Normalizes each function to have L2 norm=1
            "smoothen" : Normalize and smoothen each function
            None : Use the data as is
     
        """
        
        if process=="normalize":
            self.F = [self.Iq[i]/self.l2norm(self.Iq[i]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.Iq[i])) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i] for i in range(self.N)]
            
        return
        
    def l2norm(self, f):
        """ Compute L2 norm of a function.

        Parameters:
        ===========
        f : numpy array of shape (n_domain, )
            Discrete evaluation of function at the domain q
        
        Return:
        ======
        norm : float
            L2 norm of the function       
        
        """
        norm = np.sqrt(np.trapz(f**2, data.t))
        
        return norm
    
    def _smoothen(self, f):
        f_hat = savgol_filter(f, 8, 3)
        
        return f_hat
    

saxs = np.load('./OMIECS/PPBT_0_P123_Y_Temp.npz')

q = saxs['q']
n_domain = len(saxs['q'])

temp_flags = saxs['c'][:,1]<85
C = saxs['c'][temp_flags,:]
Iq = saxs['Iq'][temp_flags,:]

N = C.shape[0]

data = DataSet(C, q, Iq, N, n_domain=n_domain)
data.generate(process="smoothen")
print('Total number of samples %d'%data.N)

out = compute_elastic_kmeans(data, N_CLUSTERS, 
    max_iter=MAX_ITER, 
    verbose=VERBOSE, 
    smoothen=True
    )
BIC = compute_BIC(data, out.fik_gam, out.qik_gam, out.delta_n)

print('BIC with %d template functions : %2.4f'%(N_CLUSTERS, BIC))

# plot phase map and corresponding spectra
fig, axs = plt.subplots(PLOT_ROWS,PLOT_COLS, figsize=(4*PLOT_COLS, 4*PLOT_ROWS))
axs = axs.flatten()    
fig.subplots_adjust(wspace=0.4, hspace=0.4)
plot_clusters(axs, data, out, use_aligned=True)
plt.savefig(SAVE_DIR+'/phase_map_smooth%d.pdf'%N_CLUSTERS, dpi=600)

with open(SAVE_DIR+'/data_smooth%d.pkl'%N_CLUSTERS, 'wb') as handle:
    pickle.dump(out._asdict(), handle, 
        protocol=pickle.HIGHEST_PROTOCOL
        )
