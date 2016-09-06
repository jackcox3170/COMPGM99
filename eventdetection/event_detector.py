# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:31:26 2016

@author: Jack
"""

#### Event classifier

# import packages and functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import NMF
from sklearn.decomposition import SparseCoder
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier

import nimfa #has many NMF variants
import librosa #has inverse stft function
import pymf

import tables as tb
import io
import glob
import soundfile as sf
from datetime import datetime
from pymf import nmf



def nmf_multiple_parts(directory, initial_Z = 10, final_Z = 40):
    print datetime.now()
    print "NMF in stages..."
    os.chdir(directory)
    pxy_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        pxy_list.extend(filenames)
        break

    # Initial NMF analysis of each day (could do dictionary learning??)
    print "Learning bases for each chunk..."
    NMF_basis_dict = {}
    for pxy_name in pxy_list:
        # read HF spectrogram
        pxy_chunk = np.load(pxy_name).transpose() #transpose so time runs vertically down (each sample is a row)
        
        # NMF analysis
        print datetime.now()
        print "Analysing %s" % pxy_name
        NMF_learner = NMF(n_components = initial_Z) #start with a small # of components here
        NMF_learner.fit(pxy_chunk) #only need to fit as just dictionary is wanted
        NMF_basis = NMF_learner.components_
        #NM_weights_desc = pd.DataFrame(NM_power_weights.transpose()).describe()
        #print NM_weights_desc
        NMF_basis_dict[str(pxy_name)] = NMF_basis
        print "Basis vectors learned for %s" % pxy_name


    # Clustering of the learned bases
    print datetime.now()
    print "Clustering learned bases"
    
    # create dataset of all learned basis (dictionary) components
    NMF_bases_all = np.concatenate(NMF_basis_dict.values(), axis = 0)
    
    # cluster learned components into final_Z clusters
    KMeans_learner = KMeans(n_clusters = final_Z)
    KMeans_learner.fit(NMF_bases_all)
    
    # set decomposition basis to be cluster centres
    Clustered_basis = KMeans_learner.cluster_centers_
    print "Cluster centres found"


    # Learn weights (Y) for decomposition X = BY - Sparse coding with a pre-computed dictionary
    print "Learning weights for new cluster centres"
    SC_weights_dict = {}
    for pxy_name in pxy_list:
        # read HF spectrogram
        pxy_chunk = np.load(pxy_name).transpose() #transpose so time runs vertically down (each sample is a row)
        
        # SC analysis with fixed dictionary
        print datetime.now()
        print "Learning Sparse Coding weights for %s" % pxy_name
        coder = SparseCoder(Clustered_basis, transform_n_nonzero_coefs = final_Z/2) # aim for half the coefficients being non-zero
        SC_weights = coder.fit_transform(pxy_chunk)
#        file_name = 'scweights/%s' % pxy_name
#        np.save(file_name, SC_weights)
        SC_weights_dict[str(pxy_name)] = SC_weights
        print "Sparse Coding weights learned for %s" % pxy_name

    return NMF_basis_dict, Clustered_basis, SC_weights_dict

def pymf_fixed_bases(directory, Clustered_basis, final_Z = 40):
    os.chdir(directory)
    pxy_list = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        pxy_list.extend(filenames)
        break
    
    pymf_weights_dict = {}
    for pxy_name in pxy_list:
        # read HF spectrogram        
        pxy_chunk = np.load(pxy_name).transpose() #transpose so time runs vertically down (each sample is a row)
        
        # pymf analysis with fixed dictionary
        print datetime.now()
        print "Learning pymf weights for %s" % pxy_name
        pymf_learner = nmf.NMF(pxy_chunk, num_bases=final_Z, niter=1, compH=False)
        pymf_learner.H = Clustered_basis
        pymf_learner.factorize()
        pymf_weights = pymf_learner.W

        pymf_weights_dict[str(pxy_name)] = pymf_weights
        print "pymf weights learned for %s" % pxy_name

    return pymf_weights_dict
    

def event_detector(day_array, tagging_info, time_ticks, num_big_events = 1000, error_room = 10):

    # extract tagged event information    
    start_array = np.array(tagging_info.iloc[:,2]) - error_room #allow for recording error
    end_array = np.array(tagging_info.iloc[:,3]) + error_room #allow for recording error
    num_tagged_events = np.shape(start_array)[0]
    
    #calculate distance between consecutive feature vecs
    difference = np.diff(day_array, axis = 0)
    difference_mag = np.linalg.norm(difference, axis = 1)

    #plot these distances
    plt.figure()
    plt.plot(difference_mag)
    plt.title("Event detector")

    #choose biggest X events
    ind = np.argpartition(difference_mag, -num_big_events)[-num_big_events:]
    big_events_list = [time_ticks.iloc[i] for i in ind]
        
    #count the number of the biggest events that are in the event windows
    count = 0
    match_list = []
    for event in big_events_list:
        big_event_vec = np.repeat(event, num_tagged_events)
        check = ((big_event_vec > start_array) & (big_event_vec < end_array))
        match_list.append(check)
        count += check.any() # the dryer!!
    
    match_list_sum = np.sum(match_list, axis = 0)
    
    print count #2066/ 4000 biggest events are in the event windows (low because of unlablled data?)
    print match_list_sum
    print np.count_nonzero(match_list_sum) #only getting 35/131 events (only looked at 1/3 data sets)
    
    #obtain list of the appliances found and those not
    selector = match_list_sum > np.repeat(0, num_tagged_events)
    unfound_selector = 1 - selector
    found_appliances = tagging_info.iloc[:,1] * selector
    unfound_appliances = tagging_info.iloc[:,1] * unfound_selector

    return found_appliances, unfound_appliances
    
def pymf_weight_concat(pymf_weights_dict):
    #initialise pymf_weights as first chunk
    pymf_weights = pymf_weights_dict['chunk_2_0.npy']
    for i in range(1,17):
        chunk_key = 'chunk_2_%s.npy' % i
        pymf_weights = np.concatenate((pymf_weights, pymf_weights_dict[chunk_key]), axis=0)
        
    return pymf_weights

def appliance_label_maker(day_directory, tagging_info, error_room = 20):
    #create pandas dataframe for labelled data
    appliance_labels = pd.read_csv(day_directory + '/TimeTicksHF.csv', header = None)
    appliance_labels.columns = ['time_ticks']
    
    #create appliance label column
    appliance_labels['appliance'] = 'no appliance'
    
    #create labels for each of the tagged events
    for i in tagging_info.index:
        #find out when appliances may have been on and when they were definitely on
        maybe_name = 'event_%s_maybe' % i
        sure_name = 'event_%s_sure'% i
        appliance_labels[maybe_name] = (appliance_labels['time_ticks'] > tagging_info.iloc[i,2] - error_room) & (appliance_labels['time_ticks'] < tagging_info.iloc[i,3] + error_room)
        appliance_labels[sure_name] = (appliance_labels['time_ticks'] > tagging_info.iloc[i,2] + error_room) & (appliance_labels['time_ticks'] < tagging_info.iloc[i,3] - error_room)
        
        # replace appliance labels with the appropriate label
        # for the definite appliance
        appliance_labels.ix[appliance_labels[sure_name], 'appliance'] = tagging_info.iloc[i,1]
        # for the fuzzy edges around events
        appliance_labels.ix[(1 - appliance_labels[sure_name]) & appliance_labels[maybe_name], 'appliance'] = 'ignore' # can learn these - not sure the worth of this though

    return appliance_labels

#tSNE

#%%
if __name__ == "__main__":

    # Run sklearn multiple parts
    startTime = datetime.now()
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/chunks'
    NMF_basis_dict, Clustered_basis, SC_weights_dict = nmf_multiple_parts(source_directory)

    # Run pymf (TIDY!!!)
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/chunks'
    pymf_weights_dict = pymf_fixed_bases(source_directory, Clustered_basis)

    # Save relevant arrays (TIDY!!!)
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy'
    os.chdir(source_directory)
    np.save('pymf_weights_dict', pymf_weights_dict)
    np.save('NMF_bases', NMF_basis_dict)
    np.save('Clustered_basis', Clustered_basis)
    
    # Spectrogram event detector
    day_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Tagged_Training_08_01_1343804401'
    spec_matrix = np.array(pd.read_csv(day_directory + '/HF.csv', header = None).transpose()) #transpose so time runs vertically down (each sample is a row)
    spec_found_appliances, spec_unfound_appliances = event_detector(spec_matrix, day_directory)

    # PYMF event detector
    pymf_weights = pymf_weight_concat(pymf_weights_dict)
    pymf_found_appliances, pymf_unfound_appliances = event_detector(pymf_weights, day_directory)

    print datetime.now() - startTime    

    

#%% ROUGH working
# PYMF event classifier
# Preprocessing into labelled training set
day_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Tagged_Training_06_15_1339743601'
os.chdir(day_directory)
# extract tagged event information    
tagging_info = pd.read_csv(day_directory + '/TaggingInfo.csv', header = None)

#%%
start_array = np.array(tagging_info.iloc[:,2]) + 20 #allow for recording error
end_array = np.array(tagging_info.iloc[:,3]) - 20 #allow for recording error

event_duration_less_edges = end_array - start_array
choice_list = [event_duration_less_edges]
cond_list = [event_duration_less_edges > 0]
print np.select(cond_list, choice_list)

#%% create pandas dataframe for labelled data
appliance_labels = pd.read_csv(day_directory + '/TimeTicksHF.csv', header = None)
appliance_labels.columns = ['time_ticks']

#create appliance label column
appliance_labels['appliance'] = 'no appliance'

#%% Create labels for each of the time stamps

for i in tagging_info.index:
    #find out when appliances may have been on and when they were definitely on
    maybe_name = 'event_%s_maybe' % i
    sure_name = 'event_%s_sure'% i
    appliance_labels[maybe_name] = (appliance_labels['time_ticks'] > tagging_info.iloc[i,2] - 20) & (appliance_labels['time_ticks'] < tagging_info.iloc[i,3] + 20)
    appliance_labels[sure_name] = (appliance_labels['time_ticks'] > tagging_info.iloc[i,2] + 20) & (appliance_labels['time_ticks'] < tagging_info.iloc[i,3] - 20)
    
    # replace appliance labels with the appropriate label
    # for the definite appliance
    appliance_labels.ix[appliance_labels[sure_name], 'appliance'] = tagging_info.iloc[i,1]    
    # for the fuzzy edges around events
    appliance_labels.ix[(1 - appliance_labels[sure_name]) & appliance_labels[maybe_name], 'appliance'] = 'ignore' # can learn these - not sure the worth of this though


#%%
appliance_labels = appliance_label_maker(day_directory, tagging_info)

#%%
source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy/train4'
os.chdir(source_directory)

#%%
np.save('appliance_labels', appliance_labels['appliance'])

#%%
np.save('pymf_weights_aggregated', pymf_weights)

#%%
np.save('pymf_weights_dict', pymf_weights_dict)


#%% Event detector playing
orig_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Tagged_Training_08_01_1343804401'
tagging_info = pd.read_csv(orig_directory + '/TaggingInfo.csv', header = None)

new_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
os.chdir(new_directory)

time_ticks = pd.DataFrame(np.load('time_ticks.npy'))

#%%
spec_array = np.load('HF_train_2.npy').transpose()
spec_found_appliances, spec_unfound_appliances = event_detector(spec_array, tagging_info, time_ticks, num_big_events = 200)

#%%
nmf_array = np.load('nmf_weights_no_background.npy')
nmf_found_appliances, nmf_unfound_appliances = event_detector(nmf_array, tagging_info, time_ticks, num_big_events = 200)


#%% PCA
new_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
os.chdir(new_directory)

time_ticks = pd.DataFrame(np.load('time_ticks.npy'))

#%%
pca_array = np.load('pca_weights.npy')
pca_found_appliances, pca_unfound_appliances = event_detector(pca_array, tagging_info, time_ticks, num_big_events = 200)

#%%
joke_nmf_array = np.load('nmf_weights_downsampled.npy')
joke_found_appliances, joke_unfound_appliances = event_detector(joke_nmf_array, tagging_info, time_ticks, num_big_events = 200)


#%%
nmf_array = np.load('nmf_weights.npy')
nmf_found_appliances, nmf_unfound_appliances = event_detector(nmf_array, tagging_info, time_ticks, num_big_events = 2000, error_room = 20)


#%%
dummy_array = np.random.random((30999,2))
dummy_found_appliances, dummy_unfound_appliances = event_detector(dummy_array, tagging_info, time_ticks, num_big_events = 2000, error_room = 20)


#%%
