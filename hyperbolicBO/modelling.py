import numpy as np
import matplotlib.pyplot as plt

from pyGDM2 import (core, propagators, fields, 
                    materials, linear, structures, 
                    tools, visu)

import pdb

import warnings
warnings.filterwarnings("ignore")

import time
from scipy.interpolate import splev, splrep


class EmulatorSingleParticle:
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def get_uvvis(self, radius, length, theta=0.0, 
        n_samples=100, scale_factor=0.25, num_dipoles=2e3):
        """Compute UV-Vis spectra
        Note that to speed up the simulation, this function spline interpolates the spectra
        """
        
        t0 = time.time()
        if self.verbose:
            print('Radius : %.2f length : %.2f theta : %.2f'%(radius, length, theta))

        field_generator = fields.planewave
        wavelengths = np.linspace(400, 900, 20)

        # assign a polarization to the source
        kwargs = dict(theta = [theta])
        efield = fields.efield(field_generator, 
                       wavelengths=wavelengths, kwargs=kwargs)
        
        if np.isclose(radius,length, atol=1e-2):
            scale_factor = ((3*num_dipoles)/(4*np.pi))**(1/3)
            step = radius/scale_factor
            geometry = structures.sphere(step, R=radius/step, mesh='cube')
        else:
            scale_factor = ((radius*num_dipoles)/(length*np.pi))**(1/3)
            step = radius/scale_factor
            geometry = structures.nanorod(step, R=radius/step, L=length/step, mesh='cube')
            
        if self.verbose:
            print('Using a step size of %.2f given sf : %.2f and dipoles %d'%(step, scale_factor, num_dipoles))
            print('Number of dipoles: ', len(geometry))
        material = materials.gold()

        # substrate and environment
        n1, n2 = 1.33, 1.33
        dyads = propagators.DyadsQuasistatic123(n1=n1, n2=n2)

        struct = structures.struct(step, geometry, material)
        sim = core.simulation(struct, efield, dyads)
        
        t0 = time.time()
        E = core.scatter(sim, method='lu', verbose=False)
        if self.verbose:
            print('Simulation took %.2f'%(time.time()-t0))
            
        field_kwargs = tools.get_possible_field_params_spectra(sim)[0]
        
        t0 = time.time()
        wl, spec = tools.calculate_spectrum(sim, field_kwargs, linear.extinct)
        if self.verbose:
            print('calculate_spectrum took %.2f'%(time.time()-t0))
            
        a_ext, a_sca, a_abs = spec.T
        a_geo = tools.get_geometric_cross_section(sim)
        
        ext_eff = a_ext/a_geo 
        spl = splrep(wl, ext_eff)
        wl_spl = np.linspace(400, 900, n_samples)
        ext_eff_spl = splev(wl_spl, spl)
        
        return wl_spl , ext_eff_spl 
   
            
        
        
        
        
        
        
        