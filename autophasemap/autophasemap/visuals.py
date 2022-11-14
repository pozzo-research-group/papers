import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

color_blindf = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", 
"#a96b59", "#e76300", "#b9ac70", "#92dadd"]
bkg_spectra_color = "#717581"

def plot_clusters(axs, data, out, mode="aligned"):
	""" Plot phase map and corresponding spectra.
	
	Parameters:
	===========
		axs : axis handles
		data : DatSet object
		out : Output from the cluster run

		mode : Possible modes to plot the data
			"aligned" - plots the aligned functions from data.F (default)
			"input" - plots data.F
			"expt" - plots the experimental data in data.Iq
	"""
	if isinstance(out, dict):
		from types import SimpleNamespace
		out = SimpleNamespace(**out)		
	
	n_clusters = len(np.unique(out.delta_n))
	
	for k in range(n_clusters):
		Mk = np.argwhere(out.delta_n==k).squeeze()
		for cs in Mk:
		    if mode=="aligned":
		        spectra = out.fik_gam[cs,k,:]
		    elif mode=="input":
		        spectra = data.F[cs]
		    elif mode=="expt":
		    	spectra = data.Iq[cs,:]
		        
		    axs[k].loglog(data.q, spectra, 
		                color='grey'
		               )            
		axs[k].loglog(data.q, out.templates[k], 
			color=color_blindf[k], 
			lw=3.0)
		axs[k].set_ylim([np.asarray(data.F).min(), np.asarray(data.F).max()])
		axs[k].set_xlabel(r'$q$')
		axs[k].set_ylabel(r'$I_{q}$')

		axins = axs[k].inset_axes([0.7, 0.7, 0.4, 0.4])
		axins.patch.set_alpha(0.1)
		axins.scatter(data.C[out.delta_n==k,0], 
		           data.C[out.delta_n==k,1],
		           color = color_blindf[k],
		           s = 10
		          )
		axins.scatter(data.C[out.delta_n!=k,0], 
		           data.C[out.delta_n!=k,1],
		           color = bkg_spectra_color,
		           alpha = 0.1,
		           s = 10
		          )
		
		axins.tick_params(
		    axis='both', 
		    which='both',     
		    labelbottom=False,
		    labelleft=False,  
		    )
	return
	
def plot_phasemap(composition, labels):
	""" Plot phase map.
	
	Parameters:
	===========
		composition : numpy array of shape (n_samples, dim)
			Composition typically returned from data.C
			
		labels : numpy array of shape (n_samples, )
			Phase map labels obtained from autophasemap

	"""
	cmap = plt.get_cmap("tab10")
	fig, ax = plt.subplots()
	for k in np.unique(labels):
	    Mk = labels==k
	    ax.scatter(composition[Mk,0], 
	               composition[Mk,1],
	               color = color_blindf[k], 
	               alpha=1.0
	              )
	    
	return fig, ax