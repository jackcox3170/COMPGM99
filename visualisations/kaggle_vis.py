# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 18:29:01 2016

@author: Jack
"""

### Data visualisation file (rip off kaggle visualisation pieces)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import cm




# Make plot with vertical (default) colorbar

#following can draw weights and FTFT over time
#def weights_drawer(data, title):
#    fig, ax = plt.subplots()
#    cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
#    ax.set_title(title)    
#    # Add colorbar, make sure to specify tick locations to match desired ticklabels
#    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
#    #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
#    plt.show()

# may want to minus the mean from all the data so that differences are exaggerated

#%%
def weights_drawer(data, title, cbar = False):
    fig, ax = plt.subplots(figsize = (5,5))
    cax = ax.imshow(data.transpose(), interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title(title)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Basis vector')
    plt.yticks(range(4,20,5), range(5,21,5))
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if cbar:    
        cbar = fig.colorbar(cax, ticks=[np.min(data), np.max(data)], orientation='horizontal')
        cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar
    plt.show()


#%%
def spec_drawer(data, title, cbar = False):
    plt.figure(figsize = (5,2))
    fig, ax = plt.subplots()
    cax = ax.imshow(np.fliplr(data), interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title(title)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Time Step (s)')
    plt.xticks([0,1023,2047,3071,4095], ['0kHz','250kHz','500kHz','750kHz','1000kHz'])
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if cbar:
        cbar = fig.colorbar(cax, ticks=[np.min(data), np.max(data)], orientation='horizontal')
        cbar.ax.set_xticklabels(['Low', 'High'])  # horizontal colorbar
    plt.show()



#%% Event detector playing
orig_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Tagged_Training_08_01_1343804401'
tagging_info = pd.read_csv(orig_directory + '/TaggingInfo.csv', header = None)

start_array = np.array(tagging_info.iloc[:,2]) - 10 #allow for recording error
end_array = np.array(tagging_info.iloc[:,3]) + 10 #allow for recording error
num_tagged_events = np.shape(start_array)[0]


#%%
directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
os.chdir(directory)

#time_ticks = pd.DataFrame(np.load('time_ticks.npy'))

spec_weights = np.load('HF_train_2.npy').transpose()
pca_weights = np.load('pca_weights.npy')


directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
os.chdir(directory)

nmf_weights = np.load('nmf_weights_no_background.npy')

#%% Spectrograms
spec_drawer(spec_weights[0:1000,:], 'Log Spectrogram: No Appliance', cbar = True)
spec_drawer(spec_weights[7000:8000,:], 'Log Spectrogram: Washer')
spec_drawer(spec_weights[10500:11500,:], 'Log Spectrogram: Dryer')
spec_drawer(spec_weights[17000:18000,:], 'Log Spectrogram: Dishwasher')



#%% No appliance
weights_drawer(spec_weights[0:1000,:], 'Log Spectrogram: No Appliance')
weights_drawer(nmf_weights[100:150,:], 'NMF Components: No Appliance', cbar=True)
weights_drawer(pca_weights[100:150,:], 'PCA Components: No Appliance', cbar=True)


#%%Washer
weights_drawer(spec_weights[7000:8000,:], 'Log Spectrogram: Washer')
weights_drawer(nmf_weights[7100:7150,:], 'NMF Components: Washer')
weights_drawer(pca_weights[7100:7150,:], 'PCA Components: Washer')

#%%Dryer
weights_drawer(spec_weights[10500:11500,:], 'Log Spectrogram: Dryer')
weights_drawer(nmf_weights[10600:10650,:], 'NMF Components: Dryer')
weights_drawer(pca_weights[10600:10650,:], 'PCA Components: Dryer')


#%%Dishwasher
weights_drawer(spec_weights[17000:18000,:], 'Log Spectrogram: Dishwasher')
weights_drawer(nmf_weights[17100:17150,:], 'NMF Components: Dishwasher', cbar=True)
weights_drawer(pca_weights[17100:17150,:], 'PCA Components: Dishwasher', cbar=True)


#%%Lights
weights_drawer(spec_weights[19670:19750,:], 'Front Lights Spectrogram')
