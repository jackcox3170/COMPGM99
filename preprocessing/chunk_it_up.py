# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:52:55 2016

@author: Jack
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


#%% Define chunking function
def chunk_it_up(array, chunk_width):
    # need to have different directories for each of the lean files now...
    # this chunks the horizontal direction of the array
    Y = np.shape(array)[1]
    if Y % chunk_width == 0:
        chunk_range = Y/chunk_width
    else:
        chunk_range = Y/chunk_width + 1

    for chunk_num in range(chunk_range):
        chunk_filename = 'chunks/chunk_%s' % str(chunk_num).zfill(3)           
        np.save(chunk_filename, chunker(array, chunk_width, chunk_num))

    return None

#%% Define chunker
def chunker(pxy, chunk_width, chunk_num):
    return pxy[:, chunk_width*chunk_num:chunk_width*(chunk_num+1)]


#%% Code to execute
if __name__ == "__main__":
    #read in array to be chunked up
    for i in range(1,5):
        print datetime.now()
        print "Chunking up HF_lean_%s..." % i
        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test/lean_%s/no_background' % i
        os.chdir(directory)
        file_name = 'HF_lean_%s.npy' % i
        array = np.load(file_name).transpose() #transpose so time runs in the horizontal direction
        chunk_width = 500
        chunk_it_up(array, chunk_width)


