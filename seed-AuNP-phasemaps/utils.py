import numpy as np

def _inset_spectra(c, t, ft, ax, **kwargs):
    """Add spectra at a location on an existing axis

    This function add spectra at a compositional location
    on a 2D axis. 

    Parameters
    ----------
    c : np.ndarray
        composition in 2D axis, shape (2, )
    t : np.ndarray
        domain sampling of the spectra, shape (n_samples, )
    ft : np.ndarray
        intensity values of the spectra evaluated at t, shape (n_samples, )
    ax : pyplot axis object
        axis to add the spectra into
    """
    loc_ax = ax.transLimits.transform(c)
    ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
    ins_ax.plot(t, 
                ft, 
                color = kwargs.get("color", "tab:blue"),
                alpha = kwargs.get("alpha", 1.0),
                )
    limits = kwargs.get("limits", None)
    if limits is not None:
        ins_ax.set_ylim(*limits)
    ins_ax.axis('off')
    
    return 

class MinMaxScaler:

    def __init__(self, min, max):
        """Min-Max Scaler class
        
        Provides machinary to scale vectors based on a specified
        minimum and maximum values.

        Parameters
        ----------
        min : float
            Minimum value desired
        max : float
            Maximum value desired
        """
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        """Scale a vector by specified min-max

        Parameters
        ----------
        x : np.ndarray
            array to be scaled by min-max (n_samples, )

        Returns
        -------
        np.ndarray
            min-max scaled array (n_samples, )
        """
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        """Inverse scale a vector by a specified min-max

        Parameters
        ----------
        xt : np.ndarray
            array to be inverse-scaled

        Returns
        -------
        np.ndarray
            unscaled vector
        """
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.2f'%scaler.inverse(x) 

def plot_phasemap(bounds, ax, c, s, scale_axis=True, **kwargs):
    """_summary_

    Parameters
    ----------
    bounds : np.ndarray
        bounds on the 2D design space, on column for each dimension (2,2)
    ax : matplotlib.pyplot.axis
        axis object to plot on
    c : np.ndarray
        Composition array (n_samples, 2)
    s : np.ndarray
        Spectra (n_samples, n_domain)
    scale_axis : bool, optional
        whether to scale the axis to original coordinates, by default True
    """
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    if scale_axis:
        ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
        ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    t = np.linspace(0,1, s.shape[1])
    for i, (ci, si) in enumerate(zip(c, s)):
        norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
        _inset_spectra(norm_ci,t, si, ax, **kwargs)

    return 
