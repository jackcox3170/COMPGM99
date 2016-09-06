# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 12:14:23 2016

@author: Jack
"""

#### Incremental PCA

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA, IncrementalPCA
from pymf import pca


#%% Define pca learner - standard PCA
def pca_learner(X, n_components = 20):
    # time runs horizontally in X
    pymf_pca = pca.PCA(X, num_bases = n_components)
    pymf_pca.factorize()
    pca_basis = pymf_pca.W
    pca_weights = pymf_pca.H
    
    return pca_basis, pca_weights


#%% Define function to learn weights from fixed basis
def weights_from_fixed_basis(X, pca_basis):
    # time runs horizontally in X
    n_components = np.shape(pca_basis)[1]
    pymf_pca = pca.PCA(X, num_bases = n_components, compW=False)
    pymf_pca.W = pca_basis
    pymf_pca.factorize()
    pca_weights = pymf_pca.H
    
    return pca_weights


#%% Code to execute if file is run
if __name__ == "__main__":

    startTime = datetime.now()
    print startTime

    # Load in training data
    print "Loading training data"
    #directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
    os.chdir(directory)
    X = np.load('HF_train_2.npy') # do not transpose so time runs in the horizontal direction
    #y = pd.Series(np.load('appliance_labels.npy'))
    #y_names = pd.Series.unique(y)

    #run standard pca
    print datetime.now()
    print "Learning PCA basis"
    pca_basis, pca_weights = pca_learner(X)
    print datetime.now()
    #np.save('pca_basis', pca_basis)
    #np.save('pca_weights', pca_weights.transpose()) # save so time runs vertically down

#    #run weights learner from fixed basis (House 2 labelled data)
#    #load in data
#    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy/train3/no_background/downsampled'
#    os.chdir(directory)
#    X = np.load('HF_train_3.npy') #do not transpose so time runs in the horizontal direction
#
#    pca_weights = weights_from_fixed_basis(X, pca_basis)
#    np.save('pca_weights', pca_weights.transpose())
    
    
#    #run weights learner from fixed basis (House 3 test data)
#    for i in range(1,5):
#        print datetime.now()
#        print "Learning weights for HF_lean_%s..." % i
#        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy_test/lean_%s/no_background_global' % i
#        os.chdir(directory)
#        file_name = 'HF_lean_%s.npy' % i
#        X = np.load(file_name).transpose() #transpose so time runs in the horizontal direction
#    
#        pca_weights = weights_from_fixed_basis(X, pca_basis)
#        np.save('pca_weights', pca_weights.transpose()) # save so time runs vertically down

    #run weights learner from fixed basis (House 3 test data)
    for i in range(1,5):
        print datetime.now()
        print "Learning weights for HF_lean_%s..." % i
        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test/lean_%s' % i
        os.chdir(directory)
        file_name = 'HF_lean_%s.npy' % i
        X = np.load(file_name).transpose() #transpose so time runs in the horizontal direction
    
        pca_weights = weights_from_fixed_basis(X, pca_basis)
        #np.save('pca_weights', pca_weights.transpose()) # save so time runs vertically down


    print datetime.now() - startTime
