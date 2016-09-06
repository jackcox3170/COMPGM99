# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:18:07 2016

@author: Jack
"""

### Utility functions for NMF

import os
import numpy as np
import pandas as pd


#%%
def zfilled_list_concatenator(pxy_list, tpz):
    chunk_weights = np.load('chunks/tpygz/chunk_000.npy') * tpz
    for index in range(1, len(pxy_list)):
        file_name = 'chunks/tpygz/' + str(pxy_list[index])    
        chunk_weights = np.concatenate((chunk_weights, np.load(file_name)*tpz))

    return chunk_weights


#%% dictionary concater
#define weight concatenator
def chunk_weight_concat(chunk_dict):
    #initialise pymf_weights as first chunk
    chunk_weights = chunk_dict['chunk_2_0.npy']
    for i in range(1,len(chunk_dict)):
        chunk_key = 'chunk_2_%s.npy' % i
        chunk_weights = np.concatenate((chunk_weights, chunk_dict[chunk_key]), axis=0)        
    return chunk_weights


def chunk_dict_maker(pxy_list):
    nmf_chunks_dict = {}
    for pxy_chunk in pxy_list:
        file_name = 'chunks/tpygz/' + str(pxy_chunk)
        tpygz_chunk = np.load(file_name)
        tpyz_chunk = tpygz_chunk * tpz
        nmf_chunks_dict[str(pxy_chunk)] = tpyz_chunk

    return nmf_chunks_dict



#%% Code to execute if file run
if __name__ == "__main__":

    for i in range(1,5):
        #create pxy_list
        print "Saving weights for test set number %s" % i
        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test/lean_%s/no_background' % i
        os.chdir(directory)
        pxy_list = []
        for (dirpath, dirnames, filenames) in os.walk(directory + '/chunks'):
            pxy_list.extend(filenames)
            break
        
        # OPTION 1: SAVE IF ZFILLED
        tpz = np.load('cox_tpz.npy')
        chunk_weights = zfilled_list_concatenator(pxy_list, tpz)
        np.save('nmf_weights', chunk_weights)    

#    # Option1
#    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
#    os.chdir(directory)
#
#    pxy_list = []
#    for (dirpath, dirnames, filenames) in os.walk(directory + '/chunks'):
#        pxy_list.extend(filenames)
#        break
#
#    tpz = np.load('cox_tpz.npy')
#    chunk_weights = zfilled_list_concatenator(pxy_list, tpz)
#    np.save('nmf_weights', chunk_weights)    


#    # OPTION 2: SAVE IF NOT ZFILLED
#    nmf_chunks_dict = chunk_dict_maker(pxy_list)
#    np.save('nmf_weights_no_background', chunk_weight_concat(nmf_chunks_dict))


