# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 16:39:58 2016

@author: Jack
"""

### Removes background from HF files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


#%% Define background remover
def HF_background_remover(HF_original, HF_with_background):
    # time runs vertically down in both HF arrays (both numpy)
    return (HF_original - HF_with_background.min(axis=0)).clip(min=0)
    

#%% Define non appliance data remover (uses random sampling)
def non_appliance_data_remover(X, y, time_ticks, num_to_remove = 50000):
    # X, y, time_ticks all dataframes with time running vertically down
    notX = X[y == 'no appliance'].sample(50000)
    X = X[~X.index.isin(notX.index)]
    y = y[~y.index.isin(notX.index)]
    time_ticks = time_ticks[~time_ticks.index.isin(notX.index)]

    return X, y, time_ticks
    
#%% Define ignore data remover

def ignore_data_remover(X, y):
    # X, y, time_ticks all dataframes with time running vertically down
    newX = X[y != 'ignore']    
    newy = y[y != 'ignore']
    
    return newX, newy

#%% Code to execute if file is run
if __name__ == "__main__":
    
    # 1. remove background
    background_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test'
    os.chdir(background_directory)    
    #HF_with_background = np.load('HF_train_2.npy').transpose() # so time runs vertically down
    for i in range(1,5):
        print "Computing lean file number %s..." % i
        #load in HF data
        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test/lean_%s' % i
        os.chdir(directory)
        file_name = 'HF_lean_%s.npy' % i
        HF_raw = np.load(file_name)
        
        #remove background (HF_no_background has time running vertically)
        #HF_no_background = HF_background_remover(HF_raw, HF_with_background)
        HF_no_background = HF_background_remover(HF_raw, HF_raw)
        
        #save new file
        np.save('no_background/' + file_name, HF_no_background)


#    # 2. downsample
#
#    #get data
#    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy/train4'
#    os.chdir(source_directory)
#    
#    #X = pd.DataFrame(np.load('HF_train_2.npy').transpose())
#    #X = pd.DataFrame(np.load('pymf_weights_aggregated.npy'))
#    X = pd.DataFrame(np.load('HF_train_4.npy').transpose())    
#    y = pd.Series(np.load('appliance_labels.npy'))
#
#    #record time-ticks as well
#    day_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Tagged_Training_06_15_1339743601'
#    os.chdir(day_directory)
#    #extract tagged event information
#    time_ticks = pd.read_csv(day_directory + '/TimeTicksHF.csv', header = None)
#
#    #downsample this data
#    X, y, time_ticks = non_appliance_data_remover(X, y, time_ticks)
#
#    #save this data in an appropriate directory
#    os.chdir(source_directory + '/no_background/downsampled')
#    np.save('HF_train_4.npy', np.transpose(X))
#    np.save('appliance_labels.npy', y)
#    np.save('time_ticks.npy', time_ticks)
#
#



#%% Remove ignore for classifier
#%% Spectrogram

directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
os.chdir(directory)    

X = pd.DataFrame(np.load('HF_train_2.npy').transpose())
y = pd.Series(np.load('appliance_labels.npy'))

newX, newy = ignore_data_remover(X,y)

np.save('specX_no_ignore', newX)
np.save('y_no_ignore', newy)

#%% PCA
X = pd.DataFrame(np.load('pca_weights.npy'))
newX, newy = ignore_data_remover(X,y)
np.save('pcaX_no_ignore', newX)




#%% NMF
directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
os.chdir(directory)    

X = pd.DataFrame(np.load('nmf_weights_no_background.npy'))
y = pd.Series(np.load('appliance_labels.npy'))

newX, newy = ignore_data_remover(X,y)

np.save('nmfX_no_ignore', newX)
np.save('y_no_ignore', newy)

#%% NMF (no background)

directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
os.chdir(directory)    

X = pd.DataFrame(np.load('nmf_weights.npy'))
y = pd.Series(np.load('appliance_labels.npy'))

newX, newy = ignore_data_remover(X,y)

np.save('nmfX_no_ignore', newX)
np.save('y_no_ignore', newy)



