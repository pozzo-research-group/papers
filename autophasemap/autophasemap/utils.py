import abc 
from scipy.signal import savgol_filter
import time, datetime 
from collections import namedtuple
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from .clustering import assign_clusters
import numpy as np 

class BaseDataSet(abc.ABC):
    def __init__(self, n_domain):
        self.n_domain = n_domain 

    @abc.abstractmethod
    def generate(self):
        """Generate the dataset.

        Add the functional data into a list self.F.
        """

    def _smoothen(self, f, **kwargs):
        """Compute a Savitsky-Golay filtering of the data

        Inputs:
        =======
            f : function values at the discrete sample along the domain
        
        returns:
        ========
            f_hat : Smoothened function values
        """
        f_hat = savgol_filter(f, **kwargs)
        
        return f_hat
    
    def l2norm(self, t, f):
        """Compute L2-norm of functions
        
        Inputs:
        =======
            t : numpy array of shape (n_domain, )
                the discrete sample along the domain
            f : numpy array of shape (n_domain, )
                function values at the discrete sample along the domain
        
        returns:
        ========
            norm : float 
                Norm of the function
        """
        norm = np.sqrt(np.trapz(f**2, t))
        
        return norm
    
def compute_euclidean_kmeans(data, n_clusters, max_iter=100, verbose=1, smoothen=True):
    """Compute Euclidean kmeans result for phasemapping. 
    
    This function computes a Euclidean k-means based approximation of the template functions.
    
    Parameters:
    ===========
    data : Data class object 
        (see examples)
    n_clusters : int
        Number of template functions
    max_iter : int, default 100
        Maximum number of iterations to perform
    verbose : int, [1,2,3]
        Flag to print output 
    smoothen : Boolean (default, True)
        Boolean variable to use Diffusion based assignment
        
    Returns:
    ========
    res : namedtuple
        templates : Learned template functions as a list of numpy arrays
        gam_ik : Warping functions 
            numpy array of shape (n_samples, n_templates)
        qik_gam : SRSF functions of the original data. 
            Numpy array of shape (n_samples, n_templates)
        fik_gam : Template-algined original functional data. 
            Numpy array of shape (n_samples, n_templates, n_domain)
        delta_n : Array of labels assigning each data point to a template
            List of integer labels of shape (n_sample, )
        d_amplitude : Amplitude distance of each data point to a template
            Numpy array of shape (n_sample, n_templates)
        dist : Diffused version of `d_amplitude'
            Numpy array of shape (n_sample, n_templates)     
        error: Final error at the end of the execution (float)

    Notes:
    ======
    Some of the output varaibles are redundant but are kept to keep the consistency
    with `compute_elastic_kmeans` function
                                                                                                
    """
    res = namedtuple("res", "templates gam_ik qik_gam fik_gam delta_n dist d_amplitude error")
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, max_iter=max_iter, verbose=verbose)
    n_samples = len(data.F)
    X = np.asarray(data.F)
    kmeans.fit(X)
    templates = kmeans.cluster_centers_ 
    gam_ik = np.ones((n_samples, n_clusters))
    qik_gam = np.ones((n_samples, n_clusters)) 
    fik_gam = np.tile(X.reshape(len(data.F), 1, data.n_domain), [1,n_clusters,1])
    d_amplitude = cdist(X, templates) 
    dist, delta_n = assign_clusters(data, d_amplitude, smoothen=smoothen) 
    error = kmeans.inertia_  

    end = time.time()
    time_str =  str(datetime.timedelta(seconds=end-start))   
    print('Error : %2.4f and took %s'%(error, time_str))
        
    return res(templates, gam_ik, qik_gam, fik_gam, delta_n, dist, d_amplitude, error)