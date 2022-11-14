import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pdb
import optimum_reparamN2 as orN2
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import cumtrapz
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import svd
from pygsp import graphs
import time, os, traceback, shutil, warnings, sys, datetime
from collections import namedtuple
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

from .geometry import SquareRootSlopeFramework, WarpingManifold
from .diffusion import DiffusionMaps

import ray

@ray.remote
def compute_cluster_distance(i, data, eta):
    SRSF = SquareRootSlopeFramework(data.t)
    ip_address = ray._private.services.get_node_ip_address()
    n_clusters = len(eta)
    di = np.zeros(n_clusters)
    gam_i = np.zeros((n_clusters, data.n_domain))
    qi_gam = np.zeros((n_clusters, data.n_domain))
    fi_gam = np.zeros((n_clusters, data.n_domain))
    qi = SRSF.to_srsf(data.F[i])
    for k in range(n_clusters):
        _gam = SRSF.get_gamma(eta[k], qi)
        _fik_gam = SRSF.warp_f_gamma(data.F[i], _gam)
        _qik_gam = SRSF.to_srsf(_fik_gam)
        di[k] = np.sqrt(np.trapz((eta[k] - _qik_gam)**2, data.t))
        
        gam_i[k,...] = _gam
        qi_gam[k,...] = _qik_gam
        fi_gam[k,...] = _fik_gam

    return i, di, gam_i, qi_gam, fi_gam

def process_step2a(results, data, n_clusters):
    d_amplitude = np.zeros((data.N, n_clusters))
    gam_ik = np.zeros((data.N,n_clusters, data.n_domain))
    qik_gam = np.zeros((data.N,n_clusters, data.n_domain))
    fik_gam = np.zeros((data.N,n_clusters, data.n_domain))

    for r in results:
        d_amplitude[r[0],...] = r[1]
        gam_ik[r[0],...] = r[2]
        qik_gam[r[0],...] = r[3]
        fik_gam[r[0],...] = r[4]
        
    return d_amplitude, gam_ik, qik_gam, fik_gam

@ray.remote
def center_to_template(i, data, template, gam_inv):
    SRSF = SquareRootSlopeFramework(data.t)
    center = SRSF.warp_q_gamma(template, gam_inv)
    qi = SRSF.to_srsf(data.F[i])
    _gam = SRSF.get_gamma(center, qi)
    _fik_gam = SRSF.warp_f_gamma(data.F[i], _gam)
    _qik_gam = SRSF.to_srsf(_fik_gam)
    
    return i, _gam, _fik_gam, _qik_gam


def get_Mk(data, delta_n, k):
    Mk = np.argwhere(delta_n==k)
    _len, _ = Mk.shape
    # if a cluster doesn't have any points, assign a random one
    if _len == 0:
        warnings.warn("No points in the cluster...")
        Mk = np.random.choice(np.arange(data.N), 2)
    else:
        Mk = Mk.reshape(
            _len,
        )
        
    return Mk

def assign_clusters(data, d_amplitude, smoothen=True):
    N, p = d_amplitude.shape
    
    if smoothen:
        diffmap = DiffusionMaps(data.C)
        
        dist = np.zeros((N, p))
        for i in range(p):
            s = d_amplitude[:,i]
            s_norm, s_hat, s_tilda = diffmap.get_asymptotic_function(s)
            dist[:,i] = s_tilda
    else:
        dist = d_amplitude

    labels = labels_constrained(dist, 5, N)
    
    return dist, labels
    
def compute_elastic_kmeans(data, n_clusters, max_iter=100, verbose=1, smoothen=True):
    """Compute elastic kmeans 
    
    This function computes a elastic k-means based approximation of the template functions.
    
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
        delta_n : Array of labels assigning each data point to a template
            List of integer labels of shape (n_sample, )
        d_amplitude : Amplitude distance of each data point to a template
            Numpy array of shape (n_sample, n_templates)
        dist : Diffused versionof `d_amplitude'
            Numpy array of shape (n_sample, n_templates)     
        error: Final error at the end of the execution (float)
                                                                                                
    """
    DATA_RAY = ray.put(data)
    
    SRSF = SquareRootSlopeFramework(data.t)
    manifold = WarpingManifold(data.t)
    
    res = namedtuple("res", "templates gam_ik qik_gam fik_gam delta_n dist d_amplitude error")
    start = time.time()
    Q = [SRSF.to_srsf(fi) for fi in data.F]
    random_sample = np.random.choice(np.arange(data.N), n_clusters)
    eta = [Q[r] for r in random_sample]

    for n in range(max_iter):
        # step 2a
        eta_ray = ray.put(eta)

        results_ids = [compute_cluster_distance.remote(i, DATA_RAY, eta_ray) for i in range(data.N)]
        results = ray.get(results_ids)

        d_amplitude, gam_ik, qik_gam, fik_gam = process_step2a(results, data, n_clusters)
        
        #step 2b
        dist, delta_n = assign_clusters(data, d_amplitude, smoothen=smoothen)

        #step 2c
        for k in range(n_clusters):
            Mk = get_Mk(data, delta_n, k)
            gam_inv = manifold.center(gam_ik[Mk,k,:].T)
            results_ids = [center_to_template.remote(i, DATA_RAY, eta[k], gam_inv) for i in Mk]
            results = ray.get(results_ids)
            for r in results:
                qik_gam[r[0],k,...] = r[3]
                fik_gam[r[0],k,...] = r[2]
                gam_ik[r[0],k,...] = r[1]

        # step 2d
        eta_new = []
        templates = []
        for k in range(n_clusters):
            Mk = get_Mk(data, delta_n, k)
            mean_fk = fik_gam[Mk,k,:].mean(axis=0)
            templates.append(mean_fk)
            eta_new.append(SRSF.to_srsf(mean_fk))

        error = 0
        for k in range(n_clusters):
            error += np.linalg.norm(eta_new[k]-eta[k])/np.linalg.norm(eta[k])

        eta = eta_new
        error = error/n_clusters
        
        if error<1e-3:
            if verbose>1:
                print('Error threshold reached...')
            break
            
        if (100*n/max_iter)%10==0:
            if verbose>2:
                end = time.time()
                time_str =  str(datetime.timedelta(seconds=end-start)) 
                print('(%s)\tIteration : %d\tError : %2.4f'%(time_str, n, error))
            
        del eta_ray 
    
    end = time.time()
    time_str =  str(datetime.timedelta(seconds=end-start))   
    print('Total iterations %d\tError : %2.4f and took %s'%(n, error, time_str))
        
    return res(templates, gam_ik, qik_gam, fik_gam, delta_n, dist, d_amplitude, error)
    
    
def amplitude_fpca(time, Q, F, n_components):
    
    mididx = int(np.round(time.shape[0] / 2))
    mq_new = Q.mean(axis=0)

    m_new = np.sign(F[:, mididx]) * np.sqrt(np.abs(F[:, mididx]))
    mqn = np.append(mq_new, m_new.mean())
    
    Qhat = np.hstack((Q, m_new.reshape(-1,1)))
    N, T = Qhat.shape
    
    # Compute PCA
    cov = np.cov(Qhat, rowvar=False)
    try:
        U, s, V = svd(cov)
    except:
        traceback.print_exc()
    
    coeff = np.zeros((N, n_components))
    for i in range(N):
        for j in range(n_components):
            coeff[i, j] = np.dot((Qhat[i,:] - mqn), U[:, j])

    eigs = s[:n_components]

    return coeff, eigs


def compute_BIC(data, fik_gam, qik_gam, delta_n):
    """Compute Bayesian information criteria.
    
    Parameters:
    ===========
        data : DatSet object 
            see GaussianClusters for an example.
        fik_gam : numpy array of shape (n_samples, n_clusters, n_domain)
            Aligned functions.
        qik_gam : numpy array of shape (n_samples, n_clusters, n_domain)
            Aligned SRSFs of functional data.
        delta_n : array-like, shape=[n_samples,]
            Cluster memberships.   
            
    Returns:
    =======
        BIC : float
            Computed BIC of the clustering.
                     
    """
    n_components = 20
    K = len(np.unique(delta_n))
    vfpca = []

    explained_variance = np.zeros((K,n_components)) 
    for k in range(K):
        Mk = delta_n == k
        qk = qik_gam[Mk, k, ...]
        fk = fik_gam[Mk, k, ...]

        coeff, eigs = amplitude_fpca(data.t, qk, fk, n_components)
        vfpca.append(coeff)

        explained_variance[k,:] = np.cumsum(eigs) / sum(eigs)
        
    # BIC
    explained_variance = explained_variance.mean(axis=0)
    d = np.where(explained_variance >= 0.99)[0][0]

    mu = np.zeros((d,K))
    C = np.zeros((d,d,K))
    sizes = np.zeros(K)

    ell = 0.0
    for k in range(K):
        tmp = vfpca[k]
        N1 = tmp.shape[0]
        sizes[k] = N1/data.N
        C = tmp[:,:d]
        mu = np.mean(C,axis=0)
        sigma = np.cov(C.transpose())
        ell += sum(np.log(N1/data.N) + 
            mvn.logpdf(C, mu,sigma,allow_singular=True))

    BIC = -2*ell + np.log(data.N)*((2*d+1)*K-1)

    return BIC
    

def labels_constrained(D, size_min, size_max):
    n_X, n_C = D.shape
    edges, costs, capacities, supplies = minimum_cost_flow_problem_graph(D, size_min, size_max)
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    return labels

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out

def minimum_cost_flow_problem_graph(D, size_min, size_max):
    n_X, n_C = D.shape
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Edges
    edges_X_C_dummy = cartesian([X_ix, C_dummy_ix])  # All X's connect to all C dummy nodes (C')
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # Each C' connects to a corresponding C (centroid)
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # All C connect to artificial node

    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs
    costs_X_C_dummy = D.reshape(D.size)
    costs = np.concatenate([costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))])

    # Capacities - can set for max-k
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # The total supply and therefore wont restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C)
    ])

    # Sources and sinks
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)  # Demand node
    supplies_art = -1 * (n_X - n_C * size_min)  # Demand node
    supplies = np.concatenate([
        supplies_X,
        np.zeros(n_C),  # C_dummies
        supplies_C,
        [supplies_art]
    ])

    # All arrays must be of int dtype for `SimpleMinCostFlow`
    edges = edges.astype('int32')
    costs = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give extra precision
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies


def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    for count, supply in enumerate(supplies):
        min_cost_flow.set_node_supply(count, supply)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_M = np.array([min_cost_flow.flow(i) for i in range(n_X * n_C)]).reshape(n_X, n_C).astype('int32')

    labels = labels_M.argmax(axis=1)
    
    return labels
