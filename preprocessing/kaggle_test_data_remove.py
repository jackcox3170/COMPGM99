# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:01:59 2016

@author: Jack
"""

#### H3 waste test data finder

import numpy as np
import pandas as pd
import os
from datetime import datetime


#%% find time limits for H3 test files
def time_limit_finder(house = 'H3', num_test_files = 4):

    #read in sample submission
    directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin'
    os.chdir(directory)
    sample_sub = pd.read_csv('SampleSubmission.csv')

    # create a dummy dataframe with info for just the house in question and one appliance
    dummy = sample_sub.loc[(sample_sub['House'] == house) & (sample_sub['Appliance'] == 1)]

    # iterate through time stamps to find the gaps between each row
    index_list = list(dummy.index.values)
    gap_list = []
    for i in range(1,len(index_list)):
        index = index_list[i]
        index_pre = index_list[i-1]
        gap = dummy.loc[index, 'TimeStamp'] - dummy.loc[index_pre, 'TimeStamp']
        gap_list.append(gap)

    # find the max three gaps and return the indices
    gap_arr = np.array(gap_list)
    biggest_3 = gap_arr.argsort()[-(num_test_files - 1):][::-1]

    # create sorted list of start times
    start_time_stamp_list = []
    for i in biggest_3:
        index = index_list[i+1]
        time_stamp = dummy.loc[index, 'TimeStamp']
        start_time_stamp_list.append(time_stamp)
    
    start_time_stamp_list.append(dummy.loc[index_list[0], 'TimeStamp'])
    start_time_stamp_list = sorted(start_time_stamp_list)

    # create sorted list of end times
    end_time_stamp_list = []
    for i in biggest_3:
        index = index_list[i]
        time_stamp = dummy.loc[index, 'TimeStamp']
        end_time_stamp_list.append(time_stamp)
    
    end_time_stamp_list.append(dummy.loc[index_list[-1], 'TimeStamp'])
    end_time_stamp_list = sorted(end_time_stamp_list) 

    return start_time_stamp_list, end_time_stamp_list


#%% Define time_stamp_converter
def time_stamp_creator(time_tags, first_time, last_time, time_step = 60):
    #create list of time stamps    
    time_stamp_list = range(first_time, last_time + 1, 60)

    #add timestamp column to time_tags
    time_tags['TimeStamp'] = 0

    #iteratively re-label timestamp column to achieve correct labelling
    for time_stamp in time_stamp_list:
        time_tags.loc[time_tags[0] > time_stamp, 'TimeStamp'] = time_stamp

    return time_tags


#%% Define data remover for parts of HF data that aren't tested
def untested_data_remover(data_frame, time_tags, first_time, last_time):
    #create time stamp info next to time_tags
    time_tags = time_stamp_creator(time_tags, first_time, last_time)

    #remove rows outside of our time_stamp range
    df_lean = data_frame.loc[(time_tags['TimeStamp'] > 0) & (time_tags['TimeStamp'] < last_time)]    
    time_tags_lean = time_tags.loc[(time_tags['TimeStamp'] > 0) & (time_tags['TimeStamp'] < last_time)]    
    
    return df_lean, time_tags_lean
    
#%% Define function to save
def lean_data_saver(source_directory, time_tags_directory_list, start_time_stamp_list, end_time_stamp_list, file_number):
    # file number is an integer 1-4
    os.chdir(time_tags_directory_list[file_number-1])
    time_tags = pd.read_csv('TimeTicksHF.csv', header = None)
    os.chdir(source_directory)

    first_time = start_time_stamp_list[file_number-1]
    last_time = end_time_stamp_list[file_number-1] + 60

    file_name = 'HF_test_%s.npy' % (file_number-1)
    data_frame = pd.DataFrame(np.load(file_name)).transpose()


    df_lean, time_tags_lean = untested_data_remover(data_frame, time_tags, first_time, last_time)

    df_name = 'HF_lean_%s.npy' % file_number
    time_name = 'time_lean_%s.npy' % file_number
    
    np.save(df_name, df_lean)
    np.save(time_name, time_tags_lean)
    
    return df_lean

#%% Code to execute if file is run
if __name__ == "__main__":
    
#    # House 3
#    
#    #set source directory
#    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/numpy_test'
#    
#    #find time limits
#    start_time_stamp_list, end_time_stamp_list = time_limit_finder()
#
#    #read in time tags
#    time_tags_directory1 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Testing_08_02_1343890801'
#    time_tags_directory2 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Testing_08_09_1344495601'
#    time_tags_directory3 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Testing_08_22_1345618801'
#    time_tags_directory4 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H3_CSV/Testing_01_21_1358755201'    
#    time_tags_directory_list = [time_tags_directory1, time_tags_directory2, time_tags_directory3, time_tags_directory4]
#    
#    for i in range(1,5):
#        print "Computing file number %s..." % i
#        df_lean = lean_data_saver(source_directory, time_tags_directory_list, start_time_stamp_list, end_time_stamp_list, i)

    # House 2
    
    #set source directory
    source_directory = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/numpy_test'
    
    #find time limits
    start_time_stamp_list, end_time_stamp_list = time_limit_finder(house = 'H2')

    #read in time tags
    time_tags_directory1 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Testing_07_17_1342508401'
    time_tags_directory2 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Testing_07_18_1342594801'
    time_tags_directory3 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Testing_07_19_1342681201'
    time_tags_directory4 = 'C:/Users/Jack/Desktop/ubuntu_share/kaggle_belkin/H2_CSV/Testing_07_20_1342767601'    
    time_tags_directory_list = [time_tags_directory1, time_tags_directory2, time_tags_directory3, time_tags_directory4]
    
    for i in range(4,5):
        print "Computing file number %s..." % i
        df_lean = lean_data_saver(source_directory, time_tags_directory_list, start_time_stamp_list, end_time_stamp_list, i)

    
    
