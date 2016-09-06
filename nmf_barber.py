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
    def __init__(self, tol = 0.0001, plotprogress = True, maxit = 100, randinit = True, tpxgz = None, tpygz = None, tpz = None):
        self.tol = tol
        self.plotprogress = plotprogress
        self.maxit = maxit
        self.randinit = randinit
        self.tpxgz = tpxgz
        self.tpygz = tpygz
        self.tpz = tpz

# Define main routine
def nmf(pxy, Z, opts):
    ### INPUTS
    # pxy: frequency count matrix
    # Z: number of hidden components
    # opts: Options
    ### OUTPUTS
    # tpxgz : approximation of p(x|z)
    # tpygz : approximation of p(y|z)
    # tpz : approximation of p(z)
    # tpxy : approximation of p(x,y)
    [X, Y] = np.shape(pxy)
    if opts.randinit:
        #initialise tpz
        tpz = np.random.random((Z,1))
        tpz = tpz/ tpz.sum()
        
        #initialise tpxgz
        tpxgz = np.random.random((X, Z))
        tpxgz = tpxgz/ tpxgz.sum(axis = 0)

        #initialise tpygz
        tpygz = np.random.random((Y, Z))
        tpygz = tpygz/ tpygz.sum(axis = 0)
    else:
        tpz = opts.tpz
        tpxgz = opts.tpxgz
        tpygz = opts.tpygz
    
    # EM
    L = [0] * opts.maxit
    for emloop in range(opts.maxit):
        print "EM loop number %s" % emloop
        #mem-maps (stored in disk)
        # Likelihood evaluation (can definitely be chunked - just use a smaller Y)
        tpxy = np.zeros((X,Y))
        for z in range(Z):
            tpxy = tpxy + np.multiply(np.matrix(tpxgz)[:,z] * np.matrix(tpygz.transpose())[z,:], tpz[z])
        # Record Log "likelihood"
        L[emloop] = np.multiply(pxy, logeps(tpxy)).sum()
            
        # Terminate if converged
        if emloop > 1:
            if L[emloop] - L[emloop - 1] < opts.tol:
                break
        
        # E-STEP (***Need to chunk***)
        qzgxy = np.zeros((Z,X,Y))
        
        for z in range(Z):
            qzgxy[z,:,:] = np.multiply(np.matrix(tpxgz)[:,z] * np.matrix(tpygz.transpose())[z,:], tpz[z]) + np.spacing(1)
            
        for z in range(Z):
            qzgxy[z,:,:] = np.squeeze(qzgxy[z,:,:]) / qzgxy[z,:,:].sum(axis = 1).sum(axis = 0)
        
        qzgxy = qzgxy / np.tile(qzgxy.sum(axis = 0), [Z,1,1])
        
        # M-STEP (***Need to chunk***)
        for z in range(Z):
            tpxgz[:,z] = np.multiply(pxy, np.squeeze(qzgxy[z,:,:])).sum(axis = 1)
            tpygz[:,z] = np.multiply(pxy, np.squeeze(qzgxy[z,:,:])).sum(axis = 0)

        tpz = tpxgz.sum(axis = 0)
        tpxgz = tpxgz / np.tile(tpxgz.sum(axis = 0), (X,1))
        tpygz = tpygz / np.tile(tpygz.sum(axis = 0), (Y,1))
        
    # Plot progress
    if opts.plotprogress:
        plt.figure()
        plt.xlabel("EM iteration")
        plt.ylabel("'Log likelihood'")
        plt.title("'Log likelihood' of NMF approximation")
        plt.plot(L[1:])

    return tpxgz, tpygz, tpz, tpxy, L      

#define logeps
def logeps(x):
    return np.log(x + np.spacing(1)) # np.spacing(1) distance between 1 and nearest f.p.


#%% code to run if file executed        
if __name__ == "__main__":
    start_time = datetime.now()
    opts = Options()
    pxy = np.random.random((500, 800)) # run algorithm with randomly generated data
    Z = 5 # number of hidden components
    tpxgz, tpygz, tpz, tpxy, L = nmf(pxy, Z, opts)
    print datetime.now() - start_time
    
