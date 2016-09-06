# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:04:47 2016

@author: Jack
"""

#### NMF translated from D. Barber's BRML toolbox in Matlab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re


# Define class for nmf options
class Options:
    def __init__(self, tol = 0.0001, plotprogress = True, maxit = 10, randinit = True, tpxgz = None, tpygz = None, tpz = None):
        self.tol = tol
        self.plotprogress = plotprogress
        self.maxit = maxit
        self.randinit = randinit
        self.tpxgz = tpxgz
        self.tpygz = tpygz
        self.tpz = tpz

# Define logeps
def logeps(x):
    return np.log(x + np.spacing(1)) # np.spacing(1) distance between 1 and nearest f.p.

# Define main routine
def nmf(pxy_directory, Z, tpxgz, opts):
    ### INPUTS
    # pxy_directory: directory with all of the pxy chunks contained (chunked as we want)
    # Z: number of hidden components
    # tpxgz: fixed dictionary
    # opts: Options
    ### OUTPUTS
    # tpxgz : approximation of p(x|z)
    # tpygz : approximation of p(y|z)
    # tpz : approximation of p(z)
    # tpxy : approximation of p(x,y)
    print "Initialising parameters..."
    os.chdir(pxy_directory)
    pxy_list = []
    for (dirpath, dirnames, filenames) in os.walk(pxy_directory):
        pxy_list.extend(filenames)
        break

    X = np.shape(np.load(pxy_list[0]))[0]


    if opts.randinit:
        #initialise tpz
        tpz = np.random.random((Z,1))
        tpz = tpz/ tpz.sum()        

        #no need to initialise tpxgz

        #tpygz_chunks randomly initialised
        tpygz_sum = np.zeros((1,Z))
        for pxy_chunk in pxy_list:
            Y_chunk = np.shape(np.load(pxy_chunk))[1]
            #initialise tpygz
            tpygz_chunk = np.random.random((Y_chunk, Z))
            #record sum of columns        
            tpygz_sum = tpygz_sum + tpygz_chunk.sum(axis = 0)
            file_name = 'tpygz/%s' % str(pxy_chunk)            
            np.save(file_name, tpygz_chunk)

    else:
        tpz = opts.tpz
        #tpygz stored in the tpygz/ directory and are NORMALISED TO START WITH            
        
    # Run EM algorithm
    print "Starting to run EM..."
    L = [0] * opts.maxit
    for emloop in range(opts.maxit):
        print datetime.now()
        print "EM loop number %s" % emloop
        # Likelihood evaluation (chunked)
        for pxy_chunk in pxy_list:
            #load pxy_chunk from disk
            pxy_chunk_array = np.load(pxy_chunk)
            Y_chunk = np.shape(pxy_chunk_array)[1]                        

            #load tpygz_chunk
            file_name = 'tpygz/%s' % str(pxy_chunk)
            #tpygz already normalised if given
            if (emloop == 0) and (not opts.randinit):
                tpygz_sum = 1
            #divide tpygz_chunk by its sum of the z axis
            tpygz_chunk = np.load(file_name)/ tpygz_sum
            np.save(file_name, tpygz_chunk)
            
            #initialise tpxy_chunk
            tpxy_chunk = np.zeros((X,Y_chunk))

            for z in range(Z):
                tpxy_chunk = tpxy_chunk + np.multiply(np.matrix(tpxgz)[:,z] * np.matrix(tpygz_chunk.transpose())[z,:], tpz[z])
            # Record Log "likelihood" for this chunk
            L[emloop] += np.multiply(pxy_chunk_array, logeps(tpxy_chunk)).sum()
            #OPTIONAL: SAVE tpxy_chunk to disk if we're in the last iteration ####
            if emloop == opts.maxit - 1:
                file_name = 'tpxy/%s' % str(pxy_chunk)
                np.save(file_name, tpxy_chunk)
        print "Likelihood for EM loop %s is: %.3f" % (emloop, L[emloop])        
        # Terminate if converged
        if emloop > 1:
            if (L[emloop] - L[emloop - 1] < opts.tol) or (emloop == opts.maxit - 1):
                break
        
        # EM step for each chunk
        tpygz_sum = np.zeros(Z)
        print datetime.now()
        for pxy_chunk in pxy_list:
            # E-STEP BIT
            Y_chunk = np.shape(np.load(pxy_chunk))[1]            
            qzgxy_chunk = np.zeros((Z,X,Y_chunk))
            #load tpygz_chunk
            file_name = 'tpygz/%s' % str(pxy_chunk)
            tpygz_chunk = np.load(file_name)
    
            for z in range(Z):
                qzgxy_chunk[z,:,:] = np.multiply(np.matrix(tpxgz)[:,z] * np.matrix(tpygz_chunk.transpose())[z,:], tpz[z]) + np.spacing(1)

            #below operation doesn't need to be chunked as Y axis is independent
            qzgxy_chunk = qzgxy_chunk / np.tile(qzgxy_chunk.sum(axis = 0), [Z,1,1])

            #M-STEP BIT
            #initialise tpygz_chunk
            tpygz_chunk = np.zeros((Y_chunk, Z))
            #initialise tpxgz_chunk

            #load in pxy_chunk_array
            pxy_chunk_array = np.load(pxy_chunk)
            
            for z in range(Z):
                tpygz_chunk[:,z] = np.multiply(pxy_chunk_array, qzgxy_chunk[z,:,:]).sum(axis = 0)
            #save tpygz_chunk
            file_name = 'tpygz/%s' % str(pxy_chunk)
            np.save(file_name, tpygz_chunk)
            #store tpygz_sum for later use (and to update tpz)
            tpygz_sum = tpygz_sum + tpygz_chunk.sum(axis=0)
            
        #update tpz at the end of the emloop - this time based on tpygz (as tpxgz fixed)
        tpz = tpygz_sum

            
    # Plot progress at the end
    if emloop > 1:
        if opts.plotprogress:
            plt.figure()
            plt.ylabel("Log likelihood")
            plt.plot(L[1:])

    return tpxgz, tpz, L      


#%%   
if __name__ == "__main__":
    start_time = datetime.now()
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
    os.chdir(directory)
    tpxgz = np.load('cox_tpxgz.npy')
    
    Z = 20 # number of hidden components
    opts = Options(maxit = 60, randinit = True)

    #run weights learner from fixed basis (House 3 test data)
    for i in range(1,5):
        print datetime.now()
        print "Learning NMF weights for HF_lean_%s..." % i
        directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test/lean_%s/no_background' % i
        os.chdir(directory)
        tpxgz, tpz, L = nmf(directory + '/chunks', Z, tpxgz, opts)
        os.chdir(directory)
        np.save('cox_tpxgz', tpxgz)
        np.save('cox_tpz', tpz)
        np.save('cox_L', L)
    
    print datetime.now() - start_time    



