import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class sample_design_space():
    def __init__(self):
        return 

    def fit_model_classifier(self, x, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.5)

        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                test_size=0.3,
                                                random_state=23)
        self.x = x
        self.gpc = GaussianProcessClassifier(kernel=kernel)
        # Fit GP with the traning data to calculate the accuracy score 
        self.gpc.fit(X_train, y_train.ravel())
        y_pred = self.gpc.predict(X_test)
        self.accuracy_score = accuracy_score(y_pred, y_test)
        #Fit GP will all the data
        self.gpc.fit(x, y.ravel())
        self.R_score = self.gpc.score(x, y.ravel())

    def fit_model_regressor(self, x, y):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.5)

        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                test_size=0.3,
                                                random_state=23)
        self.x = x
        self.gpc = GaussianProcessRegressor(kernel=kernel)
        # Fit GP with the traning data to calculate the accuracy score 
        self.gpc.fit(X_train, y_train.ravel())
        y_pred = self.gpc.predict(X_test)
        #self.accuracy_score = accuracy_score(y_pred, y_test)
        #Fit GP will all the data
        self.gpc.fit(x, y.ravel())
        self.R_score = self.gpc.score(x, y.ravel())

    def extract_probabilities(self):
        size = 10j #If this changes, also change self.normalizer to the same number and remove "j"
        self.normalizer = 10
        grid = np.mgrid[np.min(self.x[:,0]):np.max(self.x[:,0]):size, np.min(self.x[:,1]):np.max(self.x[:,1]):size, 
                                  np.min(self.x[:,2]):np.max(self.x[:,2]):size,np.min(self.x[:,3]):np.max(self.x[:,3]):size,
                                  np.min(self.x[:,4]):np.max(self.x[:,4]):size]

        grid_0 = grid[0]
        grid = np.stack([grid[0], grid[1], grid[2], grid[3], grid[4]], axis=-1)
        pt_test_gpc = self.gpc.predict_proba(grid.reshape(-1, 5))
        self.pt_test_gpc = pt_test_gpc[:, 1].reshape(*grid_0.shape)


    def select_area_by_probability(self, p = 0.7):
        locations = np.where(self.pt_test_gpc > 0.7)
        self.design_space = np.hstack((locations[0].reshape(-1,1), locations[1].reshape(-1,1), locations[2].reshape(-1,1), 
                       locations[3].reshape(-1,1), locations[4].reshape(-1,1)))/self.normalizer


    def mean_distance(self):
        '''Determines the mean distance of consecutive points in the design space'''
        dist = []
        for i in range(self.design_space.shape[0]-1):
            d = distance.euclidean([self.design_space[i,0],
                                    self.design_space[i,1],
                                    self.design_space[i,2], 
                                    self.design_space[i,3],
                                    self.design_space[i,4]],
                                    [self.design_space[i+1,0], 
                                    self.design_space[i+1,1],
                                    self.design_space[i+1,2],
                                    self.design_space[i+1,3],
                                    self.design_space[i+1,4]])
            dist.append(d)
        self.mean_dist = np.mean(dist)

    def check_design_space(self, sample):
        '''Checks if a sample is in the design space. Returns 0 if it is not and 1 if it is'''
        dist = []
        for i in range(self.design_space.shape[0]):
            d = distance.euclidean(sample, self.design_space[i,:])
            dist.append(d)
        min_dist = np.min(dist)
        if min_dist < self.mean_dist: #Sample is in design space
            in_design_space = 1
        else: #Sample is not in design space
            in_design_space = 0
        return in_design_space


    def fit_model(self, x, y, p):
        self.fit_model_classifier(x, y)
        self.extract_probabilities()
        self.select_area_by_probability(p=p)
        self.mean_distance()

    def fit_model_R(self, x, y):
        self.fit_model_regressor(x, y)
        
