import sasmodels
import sys
#sys.path.append('/home/huatthart/sasview/src')
sys.path.append('../sasview/src')
import numpy as np
from sasmodels.data import load_data
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#from geomstats.geometry.functions import SRVF
from scipy.spatial import distance
from bumps.names import Parameter
from bumps.fitters import fit
import sasmodels
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
import bumps
#uncomment and edit line below to add path to the sasview source code
#sys.path.append("/path/to/sasview/src")
#import sas
import os
import pickle
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
from scipy.spatial import distance
from matplotlib import pyplot as plt, colors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor


def combine_all(*args):
    '''Creates the main_array in which the first column is the q values, the second is the I values, the third is either 1, 2, or 3
       which determines if the data comes from ESAXS, SAXS, or MAXS.'''
    if len(args) == 3:
        ESAXS_q = args[0].x.reshape(-1,1)
        ESAXS_I = args[0].y.reshape(-1,1)
        ESAXS_dI = args[0].dy.reshape(-1,1)
        SAXS_q =  args[1].x.reshape(-1,1)
        SAXS_I =  args[1].y.reshape(-1,1)
        SAXS_dI = args[1].dy.reshape(-1,1)
        MAXS_q =  args[2].x.reshape(-1,1)
        MAXS_I =  args[2].y.reshape(-1,1)
        MAXS_dI = args[2].dy.reshape(-1,1)
        ESAXS_ident = np.array([1]*len(ESAXS_I)).reshape(-1,1)
        SAXS_ident = np.array([2]*len(SAXS_I)).reshape(-1,1)
        MAXS_ident = np.array([3]*len(MAXS_I)).reshape(-1,1)
        ESAXS = np.hstack((ESAXS_q, ESAXS_I, ESAXS_dI, ESAXS_ident))
        SAXS = np.hstack((SAXS_q, SAXS_I, SAXS_dI, SAXS_ident))
        MAXS = np.hstack((MAXS_q, MAXS_I, MAXS_dI, MAXS_ident))
        #ESAXS = cut_noise(ESAXS)
        #SAXS = cut_noise(SAXS)
        MAXS = cut_noise(MAXS)
        main_array = np.vstack((ESAXS, SAXS, MAXS))
        #main_array = outlier_detection(main_array)
    elif len(args) == 2:
        ESAXS_q = args[0].x.reshape(-1,1)
        ESAXS_I = args[0].y.reshape(-1,1)
        ESAXS_dI = args[0].dy.reshape(-1,1)
        SAXS_q =  args[1].x.reshape(-1,1)
        SAXS_I =  args[1].y.reshape(-1,1)
        SAXS_dI = args[1].dy.reshape(-1,1)
        ESAXS_ident = np.array([1]*len(ESAXS_I)).reshape(-1,1)
        SAXS_ident = np.array([2]*len(SAXS_I)).reshape(-1,1)
        ESAXS = np.hstack((ESAXS_q, ESAXS_I, ESAXS_dI, ESAXS_ident))
        SAXS = np.hstack((SAXS_q, SAXS_I, SAXS_dI, SAXS_ident))
        ESAXS = cut_noise(ESAXS)
        #SAXS = cut_noise(SAXS)
        main_array = np.vstack((ESAXS, SAXS))
        #main_array = outlier_detection(main_array)
    else:
        raise('enter 2 or 3 data sets')

    main_array = main_array[np.argsort(main_array[:, 0])]
    del_row = np.where(main_array[:,1] < 0)
    main_array = np.delete(main_array, del_row, axis=0)
    main_array = np.hstack((np.log10(main_array[:,0]).reshape(-1,1),np.log10(main_array[:,1]).reshape(-1,1), main_array[:,2].reshape(-1,1), main_array[:,3].reshape(-1,1)))
    main_array = main_array[np.argsort(main_array[:, 0])]
    return main_array

def cut(main_array, cut_loc):
    '''This function cuts the main_array depending on the values specified by cut_loc'''
    #main_array = np.delete(main_array, np.where(main_array[0:15,2] == 2)[0], axis=0)
    data_type = 1
    for row in range(main_array.shape[0]):
        if row == 0:
            all_data = main_array[row, :].reshape(1,-1)
        else:
            if len(cut_loc) == 2:
                if main_array[row, 3] == data_type:
                    all_data = np.vstack((all_data, main_array[row, :].reshape(1,-1)))
                if row == cut_loc[0]:
                    data_type = data_type + 1
                elif row == cut_loc[1]:
                    data_type = data_type + 1
            elif len(cut_loc) == 1:
                if main_array[row, 3] == data_type:
                    all_data = np.vstack((all_data, main_array[row, :].reshape(1,-1)))
                if row == cut_loc[0]:
                    data_type = data_type + 1
    score = np.sum(all_data[:,-1])
    return all_data, score

def outlier_detection(X):
    #X is an array with columns: q, I, Dq, Ident 
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty = False)
    y_pred = clf.fit_predict(X[:,0:2])
    X_scores = clf.negative_outlier_factor_
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    mean_r = np.mean(radius)
    del_row = np.where(radius > 1.5*mean_r)[0]
    X = np.delete(X, del_row, 0)
    return X


def score_func_1(arr):
    '''Fits a polynomial to the saxs curves and returns the score of the fit. This function fits the polynomial 
       for two datasets such as ESAXS and SAXS'''
    region_1 = arr[np.where(arr[:,3] == 1)[0], :]
    region_2 = arr[np.where(arr[:,3] == 2)[0], :]
    region_1_2 = np.vstack((region_1, region_2))
    

    # code to perform regression on outliers 
    #poly_regr = PolynomialFeatures(degree = 4) # our polynomial model is of order
    #X_poly = poly_regr.fit_transform(region_1_2[:,0].reshape(-1,1)) # transforms the features to the polynomial form
    #lin_reg_2 = LinearRegression() # creates a linear regression object
    #lin_reg_2.fit(X_poly, region_1_2[:,1].reshape(-1,1))
    #regress = lin_reg_2.predict(poly_regr.fit_transform(region_1_2[:,0].reshape(-1,1)))
    #plt.scatter(region_1_2[:,0], region_1_2[:,1])
    #plt.plot(region_1_2[:,0], regress)
    #score_1 = lin_reg_2.score(X_poly, region_1_2[:,1].reshape(-1,1))
    
    # fit the model for outlier detection 
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.3, novelty = False)
    y_pred = clf.fit_predict(region_1_2)
    X_scores = clf.negative_outlier_factor_
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    mean_r = np.mean(radius)
    score_1 = len(np.where(radius > 2*mean_r)[0])
    return -score_1

def score_func_2(arr):
    '''Fits a polynomial to the saxs curves and returns the score of the fit. This function fits the polynomial 
       for the ESAXS-SAXS and SAXS-MAXS and takes the average of the two fits'''
    region_1 = arr[np.where(arr[:,3] == 1)[0], :]
    region_2 = arr[np.where(arr[:,3] == 2)[0], :]
    region_3 = arr[np.where(arr[:,3] == 3)[0], :]
    region_1_2 = np.vstack((region_1, region_2))
    region_2_3 = np.vstack((region_2, region_3))
    poly_regr = PolynomialFeatures(degree = 4) # our polynomial model is of order
    X_poly = poly_regr.fit_transform(region_1_2[:,0].reshape(-1,1)) # transforms the features to the polynomial form
    lin_reg_2 = LinearRegression() # creates a linear regression object
    lin_reg_2.fit(X_poly, region_1_2[:,1].reshape(-1,1))
    regress = lin_reg_2.predict(poly_regr.fit_transform(region_1_2[:,0].reshape(-1,1)))
    #plt.scatter(region_1_2[:,0], region_1_2[:,1])
    #plt.plot(region_1_2[:,0], regress)
    score_1 = lin_reg_2.score(X_poly, region_1_2[:,1].reshape(-1,1))

    poly_regr = PolynomialFeatures(degree = 4) # our polynomial model is of order
    X_poly = poly_regr.fit_transform(region_2_3[:,0].reshape(-1,1)) # transforms the features to the polynomial form
    lin_reg_2 = LinearRegression() # creates a linear regression object
    lin_reg_2.fit(X_poly, region_2_3[:,1].reshape(-1,1))
    regress = lin_reg_2.predict(poly_regr.fit_transform(region_2_3[:,0].reshape(-1,1)))
    #plt.scatter(region_2_3[:,0], region_2_3[:,1], s = 8)
    #plt.plot(region_2_3[:,0], regress)
    score_2 = lin_reg_2.score(X_poly, region_2_3[:,1].reshape(-1,1))
    return np.mean([score_1, score_2])

def cut_noise(ESAXS):
    '''This functions cuts off the noise only in the beginning of the curve based on the average distance between the points.
       This is used to remove beam contributions to the data due to bad subtraction with X-scat.'''
    ESAXS_log = np.clip(ESAXS, 0.000001, 1000)
    ESAXS_log = np.log10(ESAXS_log)
    distances = []
    for i in range(ESAXS_log.shape[0]-1):
        distances.append(distance.euclidean([ESAXS_log[i+1,0], ESAXS_log[i+1,1]], [ESAXS_log[i,0], ESAXS_log[i,0]]))
    avg_dist_I = np.mean(distances)
    
    for i in range(ESAXS_log.shape[0]):
        if distance.euclidean([ESAXS_log[i+1,0], ESAXS_log[i+1,1]], [ESAXS_log[i,0], ESAXS_log[i,0]]) < avg_dist_I*2.0:
            break
    ESAXS = ESAXS[i:,:]
    return ESAXS

def perform_merge(**kwargs):
    '''This function merges 3 sets of data together based on two values specified in cut_locations, and plots the merged 
       and unmerged data.
       The keyword arguments for kwargs are: Data_1, Data_2, Data_3 (optional), cut_locations, idir, save.'''
    if len(kwargs.keys()) == 6:
        ESAXS = kwargs['Data_1']
        SAXS = kwargs['Data_2']
        MAXS = kwargs['Data_3']
        idir = kwargs['idir']
        arr = combine_all(ESAXS, SAXS, MAXS)
    elif len(kwargs.keys()) == 5:
        ESAXS = kwargs['Data_1']
        SAXS = kwargs['Data_2']
        arr = combine_all(ESAXS, SAXS)
        idir = kwargs['idir']
    cut_locations = kwargs['cut_locations']
    arr, _ = cut(arr, cut_locations)

    fig, ax = plt.subplots(figsize=(15,7), ncols = 2, nrows = 1)
    cm = plt.get_cmap('Set1')
    norm = colors.Normalize(vmin=1, vmax=3)
    ax[1].scatter(10**(arr[:,0]), 10**(arr[:,1]), color=cm(norm(arr[:,3])))
    #ax[1]t.scatter(np.log10(MAXS.x), np.log10(MAXS.y), s = 2, color = 'red')
    ax[1].set_xlabel('Q (${\AA}^{-1}$)')
    ax[1].set_ylabel('Intensity (a.u.)')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_title('Merged Data')
    #sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    #fig.colorbar(sm)
    q = 10**(arr[:,0])
    I = 10**(arr[:,1])
    dI = arr[:,2]
    
    #plt.scatter(10**(arr[:,0]), 10**(arr[:,1]))
    ax[0].scatter(ESAXS.x, ESAXS.y, color = 'red')
    ax[0].scatter(SAXS.x, SAXS.y, color = 'orange')
    if len(kwargs.keys()) == 6:
        ax[0].scatter(MAXS.x, MAXS.y, color = 'grey')
    ax[0].set_xlabel('Q (${\AA}^{-1}$)')
    ax[0].set_ylabel('Intensity (a.u.)')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title('All Data')
    merged_data = np.hstack((q.reshape(-1,1), I.reshape(-1,1), dI.reshape(-1,1)))
    #if os.path.exists('../Merged_data/220812/' + ESAXS.filename[21:-7]):
    #    os.remove('../Merged_data/220812/' + ESAXS.filename[21:-7])
    if kwargs['save'] == True:
        np.savetxt(idir + '/Data/' + ESAXS.filename[21:-7] + 'merged.txt', merged_data)
        plt.savefig(idir + '/Images/' + ESAXS.filename[21:-7] + '_image.png')
    return merged_data
    
def perform_search(**kwargs):
    '''This function performs a search to fund where the best locations to merge the data is. It does a grid search
       specified in x1 and x2 to find where the highest score (polynomial fit score) is located. Keyword arguments are:
       Data_1, Data_2, Data_3 (optional), idir, save'''
    if len(kwargs.keys()) == 5:
        ESAXS = kwargs['Data_1']
        SAXS = kwargs['Data_2']
        MAXS = kwargs['Data_3']
        arr = combine_all(ESAXS, SAXS, MAXS)
        x1 = np.round(np.linspace(10,200,10)).astype(int)
        x2 = np.round(np.linspace(10,700,10)).astype(int)
        xx,yy = np.meshgrid(x1,x2)
        lst_1 = []
        lst_2 = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                lst_1.append(xx[i,j])
                lst_2.append(yy[i,j])
        arr_1 = np.array(lst_1).reshape(-1,1)
        arr_2 = np.array(lst_2).reshape(-1,1)
        x = np.hstack((arr_1, arr_2))
        del_row = []
        for i in range(x.shape[0]):
            if x[i,1] < x[i,0]:
                del_row.append(i)
        x = np.delete(x, del_row, 0)
    elif len(kwargs.keys()) == 4:
        ESAXS = kwargs['Data_1']
        SAXS = kwargs['Data_2']
        arr = combine_all(ESAXS, SAXS)
        x = np.round(np.linspace(100,500,20)).astype(int)
        x = x.reshape(-1,1)

    score_lst = []
    for i in range(len(x)):      
        if len(kwargs.keys()) == 5:   
            arr = combine_all(ESAXS, SAXS, MAXS)
            cut_locations = [x[i,0],x[i,1]]
            arr, _ = cut(arr, cut_locations)
            score = score_func_2(arr)
            score_lst.append(score)
        elif len(kwargs.keys()) == 4:
            arr = combine_all(ESAXS, SAXS)
            cut_locations = [x[i]]
            arr, _ = cut(arr, cut_locations)
            score = score_func_1(arr)
            score_lst.append(score)

    scores = np.array(score_lst).reshape(-1,1)*-1
    cut_and_score = np.hstack((x,scores))
    cut_and_score = cut_and_score[np.argsort(cut_and_score[:, -1])]
    #fig, ax = plt.subplots(figsize=(10,7))
    #cm = plt.get_cmap('coolwarm')
    #norm = colors.Normalize(vmin=np.min(score_lst), vmax=np.max(score_lst))
    #plt.scatter(x[:,0], x[:,1], color=cm(norm(score_lst)), s = 100)
    #sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    #fig.colorbar(sm)
    #if kwargs['save'] == True:
    #    plt.savefig(idir + '/Search/'  + ESAXS.filename[21:-7] + 'search.png')
    
    return cut_and_score[0,:-1]

def automatic_merge(*args, **kwargs):
    '''Automatically merge datasets. The inputs are: Data_1, Data_2, Data_3 (optional), idir (the directory where the data should be saved)
       and save = True/False.'''
    if len(args) == 4:
        ESAXS = args[0]
        SAXS = args[1]
        MAXS = args[2]
        idir = args[3]
        isExist = os.path.exists(idir)
        if isExist == False:
            os.mkdir(idir)
            os.mkdir(idir + '/Search')
            os.mkdir(idir + '/Data')
            os.mkdir(idir + '/Images')
        cut_locations = perform_search(Data_1 = ESAXS, Data_2 = SAXS, Data_3 = MAXS, idir = idir, save = kwargs['save'])
        merged_data = perform_merge(Data_1 = ESAXS, Data_2 = SAXS, Data_3 = MAXS, cut_locations = cut_locations, idir = idir, save = kwargs['save'])
    elif len(args) == 3:
        ESAXS = args[0]
        SAXS = args[1]
        idir = args[2]
        isExist = os.path.exists(idir)
        if isExist == False:
            os.mkdir(idir)
            os.mkdir(idir + '/Search')
            os.mkdir(idir + '/Data')
            os.mkdir(idir + '/Images')
        cut_locations = perform_search(Data_1 = ESAXS, Data_2 = SAXS, idir = idir, save = kwargs['save'])
        print(cut_locations)
        merged_data = perform_merge(Data_1 = ESAXS, Data_2 = SAXS, cut_locations = cut_locations, idir = idir, save = kwargs['save'])
    return merged_data
     