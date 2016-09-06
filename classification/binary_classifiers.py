# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 08:46:21 2016

@author: Jack
"""

#### Create per-appliance binary classifiers to run on learned basis


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer


#import xgboost as xgb
import operator


#%% Define Unix Time converter
def unix_to_normal_time(time):
    return datetime.datetime.fromtimestamp(int(time)).strftime('%Y-%m-%d %H:%M:%S')


#%% Define multi-label classifier
def multi_lab_class(X_train, y_train):

    # convert y into binary labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)

    # Train 1 vs Rest classifier
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = 500, max_depth = 10, min_samples_leaf = 1))
    classifier.fit(X_train, y_train)
    
    return classifier, lb

#%% Define prediction maker
def prediction_maker(classifier, lb, X_test):
    # Make predictions on test data using classifier and store in dataframe
    prob_preds = classifier.predict_proba(X_test)
    prob_preds = pd.DataFrame(prob_preds, columns = lb.classes_)

    return prob_preds

#%% Define time_stamp_converter
def time_stamp_creator(time_tags, first_time, Last_time, time_step = 60):
    #create list of time stamps    
    time_stamp_list = range(first_time, last_time + 1, 60)

    #add timestamp column to time_tags
    time_tags['TimeStamp'] = 0

    #iteratively re-label timestamp column to achieve correct labelling
    for time_stamp in time_stamp_list:
        time_tags.loc[time_tags[0] > time_stamp, 'TimeStamp'] = time_stamp

    return time_tags, time_stamp_list


#%% Remove data from outside the test range
def waste_data_remover(preds, last_time):
    return preds.loc[(preds['TimeStamp'] > 0) & (preds['TimeStamp'] < last_time)]


#%% Convert predictions into Kaggle submission files
def submission_maker(prob_preds, submission, time_tags, first_time, last_time, pred_threshold = 0.5, time_step = 60, binary_threshold = 1):
    # First Get list of appliances
    appliance_list = prob_preds.columns.values
    appliance_list = list(appliance_list)
    appliance_list.remove('ignore')
    appliance_list.remove('no appliance')

    # gives raw predictions according to the prob threshold
    preds = (prob_preds > pred_threshold).astype(int)
    
    # Read tagging info and create dictionary mapping appliances to codes in the submission file
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin'
    os.chdir(directory)
    tagging_info = pd.read_csv('H3_AllTaggingInfo.csv', header = None)
    appliance_code_dict = dict(zip(tagging_info[1], tagging_info[0]))

    #RELEVANT INFO FROM KAGGLE FORUMS:
    #marked in an "inclusive" way. So if an OFF event happened at 15:45:07,
    #then the entire 15:45 minute is tagged as "ON". In other words
    #an event that was ON at 15:43:00 and OFF at 15:45:07

    #add time_stamps to the time tags and add to prediction
    time_tags_ignore, time_stamp_list = time_stamp_creator(time_tags, first_time, last_time)
    preds['TimeStamp'] = time_tags[1]

    #remove unnecessary data and sums predictions by time_stamp
    #preds = waste_data_remover(preds, last_time) # don't need this any more
    preds_grouped = preds.groupby(by = 'TimeStamp').sum()

    #create binary predictions for each time step
    binary_predictions = (preds_grouped > binary_threshold).astype(int)

    # Fill in submission from binary_predictions
    for appliance in appliance_list:
        for index in submission.index.values:
            if submission.loc[index, 'House'] == 'H3':
                if submission.loc[index, 'Appliance'] == appliance_code_dict[appliance]:
                    if submission.loc[index, 'TimeStamp'] in time_stamp_list:
                        time_stamp = submission.loc[index, 'TimeStamp']
                        submission.loc[index, 'Predicted'] = binary_predictions.loc[time_stamp, appliance]

    return submission


#%% Code to execute if file is run
if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)
    
    # Bring in training data
    print "Loading in training data"    
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/no_background/downsampled'
    #source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
    os.chdir(source_directory)
    #X_train = pd.DataFrame(np.load('pca_weights.npy')) # pca_weights already run vertically down
    X_train = pd.DataFrame(np.load('nmf_weights_no_background.npy'))
    #X_train = pd.DataFrame(np.load('HF_train_2.npy').transpose()) #have to transpose HF_train
    
    y_train = pd.Series(np.load('appliance_labels.npy'))
    
    #Train classifier
    print "Training classifier"
    classifier, lb = multi_lab_class(X_train, y_train)

    # Read submission file
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin'
    os.chdir(directory)
    submission = pd.read_csv('SampleSubmission.csv')


    # make predictions for each test file
    for i in range(1,5):
        #source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy_test'
        source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test'
        os.chdir(source_directory)

        print datetime.now()
        print "Making submission for test set number %s..." % i

        #lean_name = 'lean_%s/pca_weights.npy' % i
        #lean_name = 'lean_%s/no_background_global/pca_weights.npy' % i
        #lean_name = 'lean_%s/no_background/HF_lean_%s.npy' % (i,i)
        #lean_name = 'lean_%s/HF_lean_%s.npy' % (i,i)
        lean_name = 'lean_%s/no_background/nmf_weights.npy' % i
        
        X_test = pd.DataFrame(np.load(lean_name))
        
        # Output multi-label classifier result
        prob_preds = prediction_maker(classifier, lb, X_test)
        
        time_name = 'lean_%s/time_lean_%s.npy' % (i,i)
        time_tags = pd.DataFrame(np.load(time_name))

    
#        first_time = 1343923680
#        last_time = 1343940660 + 60
        first_time = int(time_tags[1].min())
        print "first time is %s" % first_time
        last_time = int(time_tags[1].max())
        print "last time is %s" % last_time

        submission = submission_maker(prob_preds, submission, time_tags, first_time, last_time, pred_threshold = 0.5, binary_threshold = 1)

    print "All done. Saving submission to csv file..."
    #directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin'
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV'
    #directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV'
    os.chdir(directory)

    #submission.to_csv('pca_submission_local.csv', index = False)
    #submission.to_csv('pca_submission_global.csv', index = False)
    submission.to_csv('nmf_submission_local.csv', index = False)
    #submission.to_csv('spec_submission.csv', index = False)

