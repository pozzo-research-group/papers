#!/usr/bin/env python
# coding: utf-8

""" Compute phase map given a set of spectra and required number of phases

Change log:

10/11 9:40 PM : Fix smoothening after normalization, fix python issue on Hyak
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import linalg
import pdb
import optimum_reparamN2 as orN2
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter
import scipy.io as sio
import time, os, warnings
from pygsp import graphs
from collections import namedtuple
import ray
import time, os, traceback, shutil, warnings, pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

color_blindf = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", 
"#a96b59", "#e76300", "#b9ac70", "#92dadd"]
bkg_spectra_color = "#717581"

from autophasemap import compute_elastic_kmeans, plot_clusters, compute_BIC

SAVE_DIR = './output/FeGaPd/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)


# Specify variables
N_CLUSTERS = 5
MAX_ITER = 200
VERBOSE = 3

# load data from the matlab file
FGP = sio.loadmat('./FeGaPd/FeGaPd_full_data_200817a.mat')

C = FGP['C'] # composition
XRD = FGP['X'][:,631:1181] # X-ray diffraction intensities
T = FGP['T'][:,631:1181].squeeze() # 2theta for the XRD intensities
N, n_domain = XRD.shape

# setup ray cluster
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
        
    def generate(self, normalize=True):
        if normalize:
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.Iq[i])) for i in range(self.N)]
        else:
            self.F = [self.Iq[i,:] for i in range(self.N)]
            
        return
    
    def _smoothen(self, f):
        f_hat = savgol_filter(f, 51, 3)
        
        return f_hat
    
    def l2norm(self, f):
        norm = np.sqrt(np.trapz(f**2, data.t))
        
        return norm

data = DataSet(C, T, XRD, N, n_domain=n_domain)
data.generate(normalize=True)
print('Total number of samples %d'%data.N)

converged = False
restarts = 0
while not converged:
    if restarts>10:
        raise RuntimeError('Clustering with %d is not converging'%N_CLUSTERS)
    try:
        out = compute_elastic_kmeans(data, N_CLUSTERS, 
            max_iter=MAX_ITER, 
            verbose=VERBOSE, 
            smoothen=True
            )
        BIC = compute_BIC(data, out.fik_gam, out.qik_gam, out.delta_n)
        print('BIC with %d template functions : %2.4f'%(N_CLUSTERS, BIC))
        converged = True
    except:
        traceback.print_exc()
        restarts +=1
        continue

ray.shutdown()

# plot phase map and corresponding spectra
def cosd(deg):
    # cosine with argument in degrees
    return np.cos(deg * np.pi/180)

def sind(deg):
    # sine with argument in degrees
    return np.sin(deg * np.pi/180)

def tern2cart(T):
    # convert ternary data to cartesian coordinates
    sT = np.sum(T,axis = 1)
    T = 100 * T / np.tile(sT[:,None],(1,3))

    C = np.zeros((T.shape[0],2))
    C[:,1] = T[:,1]*sind(60)/100
    C[:,0] = T[:,0]/100 + C[:,1]*sind(30)/sind(60)
    return C

XYc = tern2cart(C[:,[1,2,0]])

def plot_clusters(axs, data, labels, use_aligned=True):
    """ Plot phase map and corresponding spectra.

    axs  : axis handles
    data : DatSet object
    out : Output from the cluster run

    use_aligned : whether to use aligned functions or not
    """
    bounds = [[0, 0.5], 
          [0, 0.5]
         ]
    n_templates = len(np.unique(labels))
    
    for k in range(n_templates):
        Mk = np.argwhere(labels==k).squeeze()
        for cs in Mk:
            if use_aligned:
                spectra = out.fik_gam[cs,k,:]
            else:
                spectra = data.F[cs]

            axs[k].plot(data.q, spectra, 
                        color=bkg_spectra_color,
                        alpha = 0.5
                       )            
        axs[k].plot(data.q, out.templates[k], color=color_blindf[k], lw=3.0)
        axs[k].set_ylim([np.asarray(data.F).min(), np.asarray(data.F).max()])
        
        axins = inset_axes(axs[k], width=1.3, height=0.9)
        axins.scatter(XYc[labels==k,0], XYc[labels==k,1], 
                      color = color_blindf[k],
                      s=10
                     )
        axins.scatter(XYc[labels!=k,0], XYc[labels!=k,1], 
                      color = bkg_spectra_color,
                      alpha = 0.1,
                      s=10
                     )        
        axins.set_xlim(bounds[0])
        axins.set_ylim(bounds[1])
        axins.axis('off')
        
    return
    
fig, axs = plt.subplots(3,2, figsize=(4*2, 4*3))
axs = axs.flatten()    
fig.subplots_adjust(wspace=0.3, hspace=0.4)
plot_clusters(axs, data, out.delta_n, use_aligned=False)
plt.savefig(SAVE_DIR+'/phase_map.pdf', dpi=600)

with open(SAVE_DIR+'/data.pkl', 'wb') as handle:
    pickle.dump(out._asdict(), handle, 
        protocol=pickle.HIGHEST_PROTOCOL
        )
