# -*- coding: utf-8 -*-
"""
Created on Tue May 31 12:01:43 2016

@author: jack
"""

### Green Running: Classification

#%% Import packages
from __future__ import print_function # comment out if using python3
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
from datetime import datetime
#import xgboost as xgb  ##unable to download xgboost in windows - uncomment to run on linux
import operator

#%% Get data for classification
def get_classifier_data():
    #set working directory
#    os.chdir('/home/jack/Dropbox/Green Running/Classification')
    os.chdir('C:/Users/Jack/Dropbox/Green Running/Classification')
    #set random seed
    np.random.seed(1)
    
    # Import trainset
    tr_set = pd.read_csv('trainset.csv', header=[0, 1, 2], skipinitialspace=True, tupleize_cols=True) # grabs the MultiIndex rows
    tr_set.columns = pd.MultiIndex.from_tuples(tr_set.columns) # converts the columns
    tr_set = tr_set[tr_set.columns.sort_values()] #lexsort columns so that .loc will work properly
    
    # Some of those columns give additional metadata. For now treat the 'Unq. Type' column as your y values and the 'steady state' and 'transient' supersets the X values. Load them like this:
    y = tr_set.loc(axis=1)['Unq. Type', 'Unq. Type','ground truth']
    X = tr_set.loc(axis=1)[:,:,['steady state', 'transient']]
    
    return X, y

#%% Split data into appropriate subsets   
def data_splitter(X, y):
    # Pre-process label encoder: turns categorical classification into numerical encoding
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = pd.Series(le.transform(y))
    
    # Stratified Train/ test split
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    # Stratified train/ valid split for probability calibration
    sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.2, random_state=42)
    for train_index, valid_index in sss:
        X_train_train, X_train_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_train, y_train_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
        
    return X_train, X_test, X_train_train, X_train_valid, y_train, y_test, y_train_train, y_train_valid, le


#%% Utility function to move the midpoint of a colormap to be around the values of interest.
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#%% define heatmap plotter
def heatmap_grid(grid, a_range, b_range, c_range, filename, ylabel = 'Max Depth', xlabel = 'Num Estimators', title = 'Validation accuracy with Learning Rate of', three_params = True):
    #plot results
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(a_range), len(b_range), len(c_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    
    for a in np.arange(len(a_range)):
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores[a,:,:], interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=-0.95, midpoint=-0.85))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.colorbar()
        plt.yticks(np.arange(len(b_range)), b_range)
        plt.xticks(np.arange(len(c_range)), c_range, rotation=45)
        if three_params:
            plt.title(title + ' %.2f' % a_range[a])
            plt.show()
            plt.gcf().savefig(filename + '%.2f' + '.png' % a_range[a])
        else:
            plt.title(title)
            plt.show()
            plt.gcf().savefig(filename + '.png')
        
    return None

#%% Plotting function
def plot_confusion_matrix(cm, y, le, filename, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(np.max(y+1))
    plt.xticks(tick_marks, le.inverse_transform(np.arange(np.max(y+1))), rotation=90)
    plt.yticks(tick_marks, le.inverse_transform(np.arange(np.max(y+1))))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().savefig(filename + '.png')
    return None
	
#save plots confusion matrix
def save_confusion_matrix(cm, y_test, le, classifier_name):
    #a. Raw
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cm, y_test, le, 'raw_cm_'+classifier_name, title=classifier_name + ': Raw confusion matrix')
    
    #b. Normalized by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, y_test, le, 'norm_cm_'+classifier_name, title=classifier_name + ': Normalized confusion matrix')
    
    return None


#%% Feature creator    
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    return None

#%% log-loss per appliance calculator
def appliance_log_loss(y_true, y_pred, le, dec_place = 2):
    appliance_log_loss_scores = dict()    
    for appliance in list(y_true.unique()):
        appliance_log_loss_scores[le.inverse_transform(appliance)] = round(log_loss(np.array(y_true[y_true == appliance]), y_pred[y_true == appliance]), dec_place)
    return appliance_log_loss_scores

#%% Log-loss appliance level report writer
def appliance_log_loss_writer(y_test, y_pred, le, model_name):
    log_loss_appliance_dict = appliance_log_loss(y_test, y_pred, le)
    with open(model_name + "_class_log_loss_report_test.txt", "w") as text_file:
        print(model_name + ": Log-loss summary\n", file=text_file)
        print("Overall LOG LOSS = %.2f \n\n" % log_loss(np.array(y_test), np.array(y_pred)), file=text_file)
        print("Per Appliance LOG LOSS\n\n")
        print(log_loss_appliance_dict)
    return None


#%% XGBoost
#Cross validation function on stratified subsets to tune hyper-parameters
def xgboost_cv(X_train, y_train, n_folds = 3):

    #define grid of parameters to search    
    lr_range = [0.05, 0.1, 0.2] #learning rate range
    ne_range = [50, 100, 200, 300, 400, 500] #num_estimators range
    md_range = [4, 6, 8, 10] #max_depth range    
    param_grid = dict(learning_rate = lr_range, n_estimators = ne_range, max_depth = md_range) #dictionary of parameter options    
    
    #cross validate with XGB grid search    
    cv = StratifiedKFold(y_train, n_folds = n_folds, shuffle = True, random_state = 42) #get n-fold cross validation
    xgb_grid = GridSearchCV(xgb.XGBClassifier(objective = 'multi:softprob'), param_grid=param_grid, cv=cv, scoring = 'log_loss') #define grid-search
    xgb_grid.fit(X_train, y_train) #fit the grid search model
    
#    #plot grid
#    heatmap_grid(xgb_grid, lr_range, md_range, ne_range, filename = 'xgb_heatmap')    
    
    return xgb_grid


#Train classifier with best performing parameters and test results vs test set
def xgboost_fit(X_train, y_train, X_test, y_test, le):

    #cross-validate parameters
    xgb_grid = xgboost_cv(X_train, y_train)

    #fit model based on best parameters
    xgbc = xgb.XGBClassifier(max_depth = xgb_grid.best_params_['max_depth'], learning_rate = xgb_grid.best_params_['learning_rate'], n_estimators = xgb_grid.best_params_['n_estimators'], objective = 'multi:softprob')
    xgbc.fit(X_train, y_train)
    
    #probabilistic predictions of train and test set
    xgbc_train = pd.DataFrame(xgbc.predict_proba(X_train))
    xgbc_test = pd.DataFrame(xgbc.predict_proba(X_test))
    
    #best-guess predictions of train and test set 
    xgbc_train_pred = pd.DataFrame(xgbc.predict(X_train))
    xgbc_test_pred = pd.DataFrame(xgbc.predict(X_test))
    
    #calculate log-loss score for classifier
    xgbc_train_log_score = log_loss(np.array(y_train), np.array(xgbc_train))
    xgbc_test_log_score = log_loss(np.array(y_test), np.array(xgbc_test))    

    #write classification report to text file
    with open("XGB_class_report_train.txt", "w") as text_file:
        print("XGB: Training set performance\n", file=text_file)
        print(classification_report(y_train, xgbc_train_pred, target_names = le.inverse_transform(np.arange(np.max(y_train+1)))), file=text_file)
        print("Training set prediction: LOG LOSS = %.2f \n" % xgbc_train_log_score, file=text_file)
    with open("XGB_class_report_test.txt", "w") as text_file:
        print("XGB: Test set performance\n", file=text_file)
        print("The best parameters are %s with a score of %0.2f\n\n" % (xgb_grid.best_params_, -xgb_grid.best_score_), file=text_file)
        print(classification_report(y_test, xgbc_test_pred, target_names = le.inverse_transform(np.arange(np.max(y_test+1)))), file=text_file)
        print("Test set prediction: LOG LOSS = %.2f \n" % xgbc_test_log_score, file=text_file)

    #calculate confusion matrix
    cm = confusion_matrix(y_test, xgbc_test_pred)
    
    return xgbc, cm

#plot XGB features and their importance
def xgb_feature_importance_plot(xgbc, X_train):
    booster = xgbc.booster()    
    features = list(X_train.columns)
    ceate_feature_map(features)
    
    importance = booster.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(14, 24))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')
    return None

	

#%% Random forest
#Cross validation to tune hyper-parameters
def rfc_cv(X_train, y_train, n_folds = 3):

    #specify parameter grid to search
    min_samples_leaf_range = [1]#[1, 5, 10, 15, 20]    
    n_estimators_range = [500]#[50, 100, 200, 300, 400, 500]
    max_depth_range = [10]#[4, 6, 8, 10]
    param_grid = dict(min_samples_leaf = min_samples_leaf_range, n_estimators = n_estimators_range, max_depth = max_depth_range)
    
    #fit grid search model
    cv = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True, random_state=42)
    rfc_grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, scoring = 'log_loss')
    rfc_grid.fit(X_train, y_train)

#    #plot grid
#    heatmap_grid(rfc_grid, min_samples_leaf_range, max_depth_range, n_estimators_range, filename = 'rfc_heatmap', ylabel = 'Max Depth', xlabel = 'Num Estimators', title = 'Validation accuracy with Min_Samples_Leaf of')    
    
    return rfc_grid

#Train classifier with best performing parameters and test results vs test set
def rfc_fit(X_train, y_train, X_test, y_test, X_train_train, y_train_train, X_train_valid, y_train_valid, le, calib = True):
    
    #cross validation step
    rfc_grid = rfc_cv(X_train, y_train)    
    
    #fit random forest with best parameters
    rfc = RandomForestClassifier(n_estimators = rfc_grid.best_params_['n_estimators'], max_depth = rfc_grid.best_params_['max_depth'], min_samples_leaf = rfc_grid.best_params_['min_samples_leaf'])
    rfc.fit(X_train, y_train)
    
    #generate classification probabilities
    rfc_train = pd.DataFrame(rfc.predict_proba(X_train))
    rfc_test = pd.DataFrame(rfc.predict_proba(X_test))
    
    #best-guess classification
    rfc_train_pred = pd.DataFrame(rfc.predict(X_train))
    rfc_test_pred = pd.DataFrame(rfc.predict(X_test))
    
    #log loss scores
    rfc_train_log_score = log_loss(np.array(y_train), np.array(rfc_train))
    rfc_test_log_score = log_loss(np.array(y_test), np.array(rfc_test))

    #write classification report to text file
    with open("RFC_class_report_train.txt", "w") as text_file:
        print("RFC: Training set performance\n", file=text_file)
        print(classification_report(y_train, rfc_train_pred, target_names = le.inverse_transform(np.arange(np.max(y_train+1)))), file=text_file)
        print("Training set prediction: LOG LOSS = %.2f \n" % rfc_train_log_score, file=text_file)
    with open("RFC_class_report_test.txt", "w") as text_file:
        print("RFC: Test set performance\n", file=text_file)
        print("The best parameters are %s with a score of %0.2f\n\n" % (rfc_grid.best_params_, -rfc_grid.best_score_), file=text_file)
        print(classification_report(y_test, rfc_test_pred, target_names = le.inverse_transform(np.arange(np.max(y_test+1)))), file=text_file)
        print("Test set prediction: LOG LOSS = %.2f \n" % rfc_test_log_score, file=text_file)

    #calculate confusion matrix
    cm = confusion_matrix(y_test, rfc_test_pred)

    if calib == True:
        #Calibration code
        # Train random forest classifier, calibrate on validation data and evaluate
        # on test data
        rfc_calib = RandomForestClassifier(n_estimators = rfc_grid.best_params_['n_estimators'], max_depth = rfc_grid.best_params_['max_depth'], min_samples_leaf = rfc_grid.best_params_['min_samples_leaf'])
        rfc_calib.fit(X_train_train, y_train_train)
        #rfc_calib_probs = rfc_calib.predict_proba(X_test)
        
        sig_rfc = CalibratedClassifierCV(rfc_calib, method="sigmoid", cv="prefit")
        sig_rfc.fit(X_train_valid, y_train_valid)
        sig_rfc_probs = sig_rfc.predict_proba(X_test)
        sig_rfc_score = log_loss(y_test, sig_rfc_probs)
        
        iso_rfc = CalibratedClassifierCV(rfc_calib, method="isotonic", cv="prefit")
        iso_rfc.fit(X_train_valid, y_train_valid)
        iso_rfc_probs = sig_rfc.predict_proba(X_test)
        iso_rfc_score = log_loss(y_test, iso_rfc_probs)
        
        #write summary to .txt file
        with open("RFC_calib_report_test.txt", "w") as text_file:
            print("RFC: Calibration summary\n\n", file=text_file)
            print("Test set prediction: LOG LOSS = %.2f \n" % rfc_test_log_score, file=text_file)
            print("Sigmoid calibration: LOG LOSS = %.2f \n" % sig_rfc_score, file=text_file)
            print("Isotonic calibration: LOG LOSS = %.2f \n" % iso_rfc_score, file=text_file)        
    
        return rfc, cm, sig_rfc, iso_rfc
    else:
        return rfc, cm


#%% SVM - rbf kernel
#Cross validation to tune hyper-parameters
def svc_rbf_cv(X_train, y_train, n_folds = 3):
    #specify param range
    gamma_range = [0.001,0.003,0.01,0.03,0.1,0.3]
    C_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    param_grid = dict(gamma = gamma_range, C = C_range)
    
    #fit grid
    cv = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True, random_state=42)
    svc_rbf_grid = GridSearchCV(SVC(kernel = 'rbf', probability = True), param_grid=param_grid, cv=cv, scoring = 'log_loss')
    svc_rbf_grid.fit(X_train, y_train)
    
#    #plot results
#    heatmap_grid(svc_rbf_grid, [1], gamma_range, C_range, filename = 'svc_rbf_heatmap', ylabel = 'Gamma', xlabel = 'C', title = 'Validation accuracy', three_params = False)

    return svc_rbf_grid

#Train classifier with best performing parameters and test results vs test set
def svc_rbf_fit(X_train, y_train, X_test, y_test, X_train_train, y_train_train, X_train_valid, y_train_valid, le, calib = True): 
    #cross validate
    svc_rbf_grid = svc_rbf_cv(X_train, y_train)
    
    #train svc
    svc_rbf = SVC(kernel = 'rbf', probability = True, gamma = svc_rbf_grid.best_params_['gamma'], C = svc_rbf_grid.best_params_['C'])
    svc_rbf.fit(X_train, y_train)
    
    #probabilistic definitions
    svc_rbf_train = pd.DataFrame(svc_rbf.predict_proba(X_train))
    svc_rbf_test = pd.DataFrame(svc_rbf.predict_proba(X_test))
    
    #best guess predictions
    svc_rbf_train_pred = pd.DataFrame(svc_rbf.predict(X_train))
    svc_rbf_test_pred = pd.DataFrame(svc_rbf.predict(X_test))
    
    #log-loss
    svc_rbf_train_log_score = log_loss(np.array(y_train), np.array(svc_rbf_train))
    svc_rbf_test_log_score = log_loss(np.array(y_test), np.array(svc_rbf_test))
    
    #write classification report to text file
    with open("SVC_RBF_class_report_train.txt", "w") as text_file:
        print("SVC(RBF): Training set performance\n", file=text_file)
        print(classification_report(y_train, svc_rbf_train_pred, target_names = le.inverse_transform(np.arange(np.max(y_train+1)))), file=text_file)
        print("Training set prediction: LOG LOSS = %.2f \n" % svc_rbf_train_log_score, file=text_file)
    with open("SVC_RBF_class_report_test.txt", "w") as text_file:
        print("SVC(RBF): Test set performance\n", file=text_file)
        print("The best parameters are %s with a score of %0.2f\n\n" % (svc_rbf_grid.best_params_, -svc_rbf_grid.best_score_), file=text_file)
        print(classification_report(y_test, svc_rbf_test_pred, target_names = le.inverse_transform(np.arange(np.max(y_test+1)))), file=text_file)
        print("Test set prediction: LOG LOSS = %.2f \n" % svc_rbf_test_log_score, file=text_file)

    #calculate confusion matrix
    cm = confusion_matrix(y_test, svc_rbf_test_pred)

    if calib == True:
        #Calibration code
        # Train random forest classifier, calibrate on validation data and evaluate
        # on test data
        svc_rbf_calib = SVC(kernel = 'rbf', probability = True, gamma = svc_rbf_grid.best_params_['gamma'])
        svc_rbf_calib.fit(X_train_train, y_train_train)
        #svc_rbf_calib_probs = svc_rbf_calib.predict_proba(X_test)
        
        sig_svc_rbf = CalibratedClassifierCV(svc_rbf_calib, method="sigmoid", cv="prefit")
        sig_svc_rbf.fit(X_train_valid, y_train_valid)
        sig_svc_rbf_probs = sig_svc_rbf.predict_proba(X_test)
        sig_svc_rbf_score = log_loss(y_test, sig_svc_rbf_probs)
        
        iso_svc_rbf = CalibratedClassifierCV(svc_rbf_calib, method="isotonic", cv="prefit")
        iso_svc_rbf.fit(X_train_valid, y_train_valid)
        iso_svc_rbf_probs = sig_svc_rbf.predict_proba(X_test)
        iso_svc_rbf_score = log_loss(y_test, iso_svc_rbf_probs)
        
        #write summary to .txt file
        with open("SVC_RBF_calib_report_test.txt", "w") as text_file:
            print("SVC_RBF: Calibration summary\n\n", file=text_file)
            print("Test set prediction: LOG LOSS = %.2f \n" % svc_rbf_test_log_score, file=text_file)
            print("Sigmoid calibration: LOG LOSS = %.2f \n" % sig_svc_rbf_score, file=text_file)
            print("Isotonic calibration: LOG LOSS = %.2f \n" % iso_svc_rbf_score, file=text_file)        
    
        return svc_rbf, cm, sig_svc_rbf, iso_svc_rbf
    else:
        return svc_rbf, cm

#%% code to execute
if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)

    #get data
    ##OPTION 1: Extracted event features to classify
    #X, y = get_classifier_data() #original classifier data 

    ##OPTION 2: Learned matrix factorisations to classify kaggle data
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy/downsampled'
    os.chdir(source_directory)    
    #X = pd.DataFrame(np.load('HF_train_2.npy').transpose())
    #X = pd.DataFrame(np.load('pymf_weights_aggregated.npy'))
    #X = pd.DataFrame(np.load('pca_weights.npy')).transpose() #have to transpose pca_weights
    #X = pd.DataFrame(np.load('pca_weights.npy'))
    #X = pd.DataFrame(np.load('nmf_weights.npy'))
    X = pd.DataFrame(np.load('nmfX_no_ignore.npy'))
    #X = pd.DataFrame(np.load('specX_no_ignore.npy'))
    #X = pd.DataFrame(np.load('pcaX_no_ignore.npy'))

    
    #y = pd.Series(np.load('appliance_labels.npy'))
    y = pd.Series(np.load('y_no_ignore.npy'))
    
    #break out data into relevant subsets
    X_train, X_test, X_train_train, X_train_valid, y_train, y_test, y_train_train, y_train_valid, le = data_splitter(X, y)

    #run XGboost - doesn't work on windows! does work on linux!
    #xgbc, xgbc_cm = xgboost_fit(X_train, y_train, X_test, y_test, le)
    #save_confusion_matrix(xgbc_cm, y_test, le, 'xgb_pca_downsampled') #change file name if different data

    #run random forest classifier
    rfc, rfc_cm = rfc_fit(X_train, y_train, X_test, y_test, X_train_train, y_train_train, X_train_valid, y_train_valid, le, calib = False)
    save_confusion_matrix(rfc_cm, y_test, le, 'nmf_no_ignore') #change file name if different data

    #svc_rbf, svc_rbf_cm = svc_rbf_fit(X_train, y_train, X_test, y_test, X_train_train, y_train_train, X_train_valid, y_train_valid, le, calib = False)
    #save_confusion_matrix(rfc_cm, y_test, le, 'svc_pca_downsampled') #change file name if different data

    print(datetime.now() - start_time)

##%% RFC feature importance for each frequency
#plt.plot(rfc.feature_importances_[::-1])
#plt.title('Feature importance of different frequencies')
#plt.ylabel('Feature Importance')
#plt.xlabel('Frequency (kHz)')
#plt.xticks(range(0,4200,200*4095/1000), range(0,1050,200))





