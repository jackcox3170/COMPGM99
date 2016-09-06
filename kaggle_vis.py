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


#%%
appliance_labels = pd.DataFrame(np.load('appliance_labels.npy'))

#%%
appliance_labels = appliance_labels.loc[appliance_labels[0] != 'no appliance']

#%%
import glob
import create_data_dict as cdf
from scipy import linspace, io
from pylab import *

# Import the data
f_list = glob.glob('data/H*/Tagged_*.bin')
f1 = f_list[0]
taggingInfo, df1, df2 = cdf.create_data_dict(f1)

# Plot L1_Real and L2_Real

# Gets the timestamp range from the tagging data
def get_tagging_max_min(taggingInfo):
    tags = []
    for i in range(len(taggingInfo)):
        tags.append(taggingInfo[i, 2])
        tags.append(taggingInfo[i, 3])
    return max(tags), min(tags)


# Gets the cooresponding index ranges based on the 
# tagging data
def get_time_tick_index(timetick, mx, mn):
    lag_tick = 0
    for i, tick in enumerate(timetick):
        if tick > mx and lag_tick <= mx:
            idx_stop = i
        elif tick > mn and lag_tick <= mn:
            idx_start = i
        lag_tick = tick
    return idx_start, idx_stop    


# Use the index info to get the appropriate subset of data
mx1, mn1 = get_tagging_max_min(taggingInfo)
idx_start1, idx_stop1 = get_time_tick_index(df1['L1_TimeTicks'], mx1, mn1)
subset1 = np.array(xrange(int(round(idx_start1*0.999, -3)),int(round(idx_stop1*1.001, -3))))

mx2, mn2 = get_tagging_max_min(taggingInfo)
idx_start2, idx_stop2 = get_time_tick_index(df2['L2_TimeTicks'], mx2, mn2)
subset2 = np.array(xrange(int(round(idx_start2*0.999, -3)),int(round(idx_stop2*1.001, -3))))

# Plots the stop and start of individual devices
def plot_devices(p, taggingInfo, df):
    top = max(df)
    bottom = min(df)
    for i in range(len(taggingInfo)):
        p.plot([taggingInfo[i,2], taggingInfo[i,2]], [bottom, top],  c='g')
        p.plot([taggingInfo[i,3], taggingInfo[i,3]], [bottom, top], c='r')


# Plot L1
fig1 = figure(1)
ax1 = fig1.add_subplot(411)
ax1.plot(df1['L1_TimeTicks'][subset1], df1['L1_Real'][subset1])
plot_devices(ax1, taggingInfo, df1['L1_Real'][subset1])

ax2 = fig1.add_subplot(412)
ax2.plot(df1['L1_TimeTicks'][subset1], df1['L1_Imag'][subset1], c='r')
plot_devices(ax2, taggingInfo, df1['L1_Imag'][subset1])

ax3 = fig1.add_subplot(413)
ax3.plot(df1['L1_TimeTicks'][subset1], df1['L1_Pf'][subset1], c='g')
plot_devices(ax3, taggingInfo, df1['L1_Pf'][subset1])

# Plot L2
fig2 = figure(2)
ax4 = fig2.add_subplot(411)
ax4.plot(df2['L2_TimeTicks'][subset2], df2['L2_Real'][subset2])
plot_devices(ax4, taggingInfo, df2['L2_Real'][subset2])

ax5 = fig2.add_subplot(412)
ax5.plot(df2['L2_TimeTicks'][subset2], df2['L2_Imag'][subset2], c='r')
plot_devices(ax5, taggingInfo, df2['L2_Imag'][subset2])

ax6 = fig2.add_subplot(413)
ax6.plot(df2['L2_TimeTicks'][subset2], df2['L2_Pf'][subset2], c='g')
plot_devices(ax6, taggingInfo, df2['L2_Pf'][subset2])

show()

#%%
#%% plot of spectrogram - ###NEED TO EDIT

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.random import randn

# Make plot with vertical (default) colorbar
fig, ax = plt.subplots()

data = np.clip(randn(250, 250), -1, 1)

cax = ax.imshow(data, interpolation='nearest', cmap=cm.coolwarm)
ax.set_title('Gaussian noise with vertical colorbar')

# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

# Make plot with horizontal colorbar
fig, ax = plt.subplots()

data = np.clip(randn(250, 250), -1, 1)

cax = ax.imshow(data, interpolation='nearest', cmap=cm.afmhot)
ax.set_title('Gaussian noise with horizontal colorbar')

cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

plt.show()



#%% Pymf constrained NMF
pymf_learner = pymf.NMF(spec_matrix, num_bases=2, niter=1, compH=False)
pymf_learner.initialization()
pymf_learner.H = NMF_new_basis
pymf_learner.factorize()
pymf_weights = pymf_learner.W

#%% add labelled training data to learned weights

# read in tagging information
HF_ticks_list = []
for directory in train_directory_list:
    # read ticks
    HF_ticks = pd.read_csv(directory + '/TimeTicksHF.csv', header = None)
    HF_ticks_list.append(HF_ticks)

# create list of HF ticks
HF_ticks_all = np.concatenate(HF_ticks_list, axis = 0)

#%% combine with learned weights
tagging_info = pd.read_csv(source_directory + '/AllTaggingInfo.csv', header = None)


#%%
weights_ticks = np.concatenate((HF_ticks_all, SC_weights_all), axis = 1)