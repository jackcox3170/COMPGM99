# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:13:56 2016

@author: Jack
"""

### Classification of unlabelled test data using multi-class RF classifier

# NB this does not give multi-label predictions as it assumes exactly one class is satisfied at each time period
# For better results, use binary classifier


#%% Import packages
from __future__ import print_function # comment out if using python3
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from datetime import datetime


#%% Running classifier on actual test data

def multiclass_test_classifier(X_train, y_train, X_test):

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = pd.Series(le.transform(y_train))
    
    #fit random forest with best parameters from initial CV checker
    rfc = RandomForestClassifier(n_estimators = 500, max_depth = 10, min_samples_leaf = 1)
    rfc.fit(X_train, y_train)
    
    #generate classification probabilities
    rfc_train = pd.DataFrame(rfc.predict_proba(X_train))
    rfc_test = pd.DataFrame(rfc.predict_proba(X_test))
    
    #best-guess classification
    rfc_train_pred = pd.DataFrame(rfc.predict(X_train))
    rfc_test_pred = pd.DataFrame(rfc.predict(X_test))
    
    #log loss scores
    rfc_train_log_score = log_loss(np.array(y_train), np.array(rfc_train))
    #rfc_test_log_score = log_loss(np.array(y_test), np.array(rfc_test))
    
    #write classification report to text file
    with open("RFC_multiclass_report_train.txt", "w") as text_file:
        print("RFC: Training set performance\n", file=text_file)
        print(classification_report(y_train, rfc_train_pred, target_names = le.inverse_transform(np.arange(np.max(y_train+1)))), file=text_file) #
        print("Training set prediction: LOG LOSS = %.2f \n" % rfc_train_log_score, file=text_file)
    #with open("RFC_class_report_test.txt", "w") as text_file:
    #    print("RFC: Test set performance\n", file=text_file)
    #    print("The best parameters are %s with a score of %0.2f\n\n" % (rfc_grid.best_params_, -rfc_grid.best_score_), file=text_file)
    #    print(classification_report(y_test, rfc_test_pred, target_names = le.inverse_transform(np.arange(np.max(y_test+1)))), file=text_file)
    #    print("Test set prediction: LOG LOSS = %.2f \n" % rfc_test_log_score, file=text_file)
    
    #rename encoded labels
    rfc_test_pred[0] = le.inverse_transform(rfc_test_pred[0])
    rfc_test.columns = le.inverse_transform(rfc_test.columns.values)
    
    rfc_test.to_csv('nmf_rfc_probs.csv')
    rfc_test_pred.to_csv('nmf_rfc_pred.csv')
    
    return rfc_test


#%% Code to execute if file is run
if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)

    #get training data
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
    os.chdir(source_directory)
    
    # Load in training data
    X_train = pd.DataFrame(np.load('nmf_weights_no_background.npy')) # change if want to compare PCA/ Spectrogram weights
    y_train = pd.Series(np.load('appliance_labels.npy'))

    # Load in test data
    test_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy_test/no_background'
    os.chdir(test_directory)
    X_test = pd.DataFrame(np.load('nmf_weights_1.npy')) # change if want to compare PCA/ Spectrogram weights

    # Run function
    multiclass_test_classifier(X_train, y_train, X_test)

    print(datetime.now() - start_time)

