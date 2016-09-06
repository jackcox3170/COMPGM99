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
def nmf(pxy_directory, Z, opts):
    ### INPUTS
    # pxy_directory: directory with all of the pxy chunks contained (chunked as we want)
    # Z: number of hidden components
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
        #initialise tpxgz
        tpxgz = np.random.random((X, Z))
        tpxgz = tpxgz/ tpxgz.sum(axis = 0)
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
        tpxgz = opts.tpxgz
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
        tpxgz_new = np.zeros((X, Z))
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

            #Normalise qzgxy_chunk (used to be in E-step)
            #below operation doesn't need to be chunked as Y axis is independent
            qzgxy_chunk = qzgxy_chunk / np.tile(qzgxy_chunk.sum(axis = 0), [Z,1,1])

            
            #M-STEP BIT
            #initialise tpygz_chunk
            tpygz_chunk = np.zeros((Y_chunk, Z))
            #initialise tpxgz_chunk
            tpxgz_chunk = np.zeros((X, Z))

            #load in pxy_chunk_array
            pxy_chunk_array = np.load(pxy_chunk)
            
            for z in range(Z):
                tpxgz_chunk[:,z] = np.multiply(pxy_chunk_array, np.squeeze(qzgxy_chunk[z,:,:])).sum(axis = 1)
                tpygz_chunk[:,z] = np.multiply(pxy_chunk_array, np.squeeze(qzgxy_chunk[z,:,:])).sum(axis = 0)
            #save tpygz_chunk
            file_name = 'tpygz/%s' % str(pxy_chunk)
            np.save(file_name, tpygz_chunk)
            #store tpygz_sum for later use
            tpygz_sum = tpygz_sum + tpygz_chunk.sum(axis=0)
            # update tpxgz_new            
            tpxgz_new = tpxgz_new + tpxgz_chunk

        #update tpxgz and tpz at the end of the emloop
        tpxgz = tpxgz_new
        tpz = tpxgz.sum(axis = 0)
        tpxgz = tpxgz / np.tile(tpxgz.sum(axis = 0), (X,1))
            
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
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
    os.chdir(directory)
    opts = Options(maxit = 100, randinit = False, tpxgz = np.load('cox_tpxgz.npy'), tpz = np.load('cox_tpz.npy'))
    Z = 20 # number of hidden components
    tpxgz, tpz, L = nmf(directory + '/chunks', Z, opts)
    os.chdir(directory)
    np.save('cox_tpxgz', tpxgz)
    np.save('cox_tpz', tpz)
    np.save('cox_L', L)
    
    print datetime.now() - start_time    




