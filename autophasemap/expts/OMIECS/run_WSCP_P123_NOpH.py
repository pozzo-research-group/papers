#!/usr/bin/env python
# coding: utf-8

""" Compute phase map of polymer blend data

Change log:
-----------

10/18 12:02 AM : Initial run setup with 4 clusters

10/18 3:45 AM : run with 8 clusters

10/19 12:27 AM : 8 seems too many and 6 might be the right one?

10/19 10:48 PM : Something with the installation is broke
                


"""

import numpy as np
import matplotlib.pyplot as plt
import ray
import os, shutil, pickle
from scipy.signal import savgol_filter
from autophasemap import compute_elastic_kmeans, plot_clusters, compute_BIC

SAVE_DIR = './output/WSCP_P123_NOpH/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)


# Specify variables
N_CLUSTERS = 5
PLOT_ROWS , PLOT_COLS = 2, 3
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
        self.n_domain = n_domain
        self.t = np.linspace(0,1, num=self.n_domain)
        self.N = N
        self.Iq = Iq
        self.C = C
        self.q = q
        
    def generate(self, process=None):
        if process=="normalize":
            self.F = [self.Iq[i]/self.l2norm(self.Iq[i]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.Iq[i])) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i] for i in range(self.N)]
            
        return
        
    def l2norm(self, f):
        norm = np.sqrt(np.trapz(f**2, data.t))
        
        return norm
    
    def _smoothen(self, f):
        f_hat = savgol_filter(f, 8, 3)
        
        return f_hat

saxs = np.load('./OMIECS/blends_WSCP_P123_NOpH.npz')

q = saxs['q']
n_domain = len(saxs['q'])

C = saxs['C']
Iq = saxs['Iq']

N = C.shape[0]

data = DataSet(C, q, Iq, N, n_domain=n_domain)
data.generate(process="smoothen")
print('Number of functions : ', data.N)
print('Domain sampling of each function : ', n_domain)

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
plot_clusters(axs, data, out, mode="input")
plt.savefig(SAVE_DIR+'/phase_map_smooth%d.pdf'%N_CLUSTERS, dpi=600)

with open(SAVE_DIR+'/data_smooth%d.pkl'%N_CLUSTERS, 'wb') as handle:
    pickle.dump(out._asdict(), handle, 
        protocol=pickle.HIGHEST_PROTOCOL
        )
