# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 10:50:52 2016

@author: Jack
"""

### UK Dale pre-processing

#%% import packages and functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import NMF
import nimfa #has many NMF variants
import librosa #has inverse stft function
import json

import tables as tb
import io
import glob
import soundfile as sf

os.chdir('C:/Users/Jack/Dropbox/Green Running/Bayesian Inference/')
from SourceUKDale import SourceUKDale

from datetime import datetime


#%% load in super meta data (latest one correct acc.Lionel)
with open("C:/Users/Jack/Desktop/ubuntu_share/output_UKDale/House_2/events/SuperMetaData_003.json", "r") as f:
  super_meta_data = json.loads(f.read())
  
super_meta_data = super_meta_data['entries']

#%%
super_meta_data[2257]['meta']['event_info']['power step']

#%%
#device
print super_meta_data[0]['meta']['type']

#Onset/Offset
print super_meta_data[0]['meta']['instances']['status']

#collection time
print super_meta_data[0]['meta']['header']['collection_time']

#id
print super_meta_data[0]['id']

#%% create data_frame for all relevant info for house2
appliance_list = []
status_list = []
collection_list = []
id_list = []
power_step_list = []

for event in super_meta_data:
    appliance_list.append(event['meta']['type'])
    status_list.append(event['meta']['instances']['status'])
    collection_list.append(event['meta']['header']['collection_time'])
    id_list.append(event['id'])
    power_step_list.append(event['meta']['event_info']['power step'])

data_dict = {'id': id_list, 'appliance': appliance_list, 'status': status_list, 'collection_time': collection_list, 'power_step': power_step_list}
house2_meta = pd.DataFrame(data_dict)

#sort data-frame in ascending order
house2_meta = house2_meta.sort_values(by = 'collection_time')

#get list of appliances that are tagged in the data
tagged_appliances = list(pd.unique(house2_meta['appliance']))

#create dataframe to record onsets and offsets
house2_appliance_on_off = pd.DataFrame(index = house2_meta.index.values, columns = tagged_appliances)

#replace NaNs with 0s
house2_appliance_on_off = house2_appliance_on_off.fillna(0)

#create dataframe to record power levels
house2_appliance_power = pd.DataFrame(index = house2_meta.index.values, columns = tagged_appliances)

#replace NaNs with 0s
house2_appliance_power = house2_appliance_power.fillna(0)

#%% write a routine for recording onsets and offsets dataframe
for appliance in tagged_appliances:
    appliance_status = 0
    for index in range(len(appliance_list)):
        if house2_meta.loc[index, 'appliance'] == appliance:
            if house2_meta.loc[index, 'status'] == 'Onset':
                appliance_status = 1
            if house2_meta.loc[index, 'status'] == 'Offset':
                appliance_status = 0
        house2_appliance_on_off.loc[index, appliance] = appliance_status

#%% write a routine for recording power upds and downs dataframe
for appliance in tagged_appliances:
    appliance_power = 0
    for index in range(len(appliance_list)):
        if house2_meta.loc[index, 'appliance'] == appliance:
            appliance_power += house2_meta.loc[index, 'power_step']
        house2_appliance_power.loc[index, appliance] = appliance_power

#%% Add total columns
house2_appliance_on_off['Total'] = house2_appliance_on_off.sum(axis = 1)
house2_appliance_power['Total'] = house2_appliance_power.sum(axis = 1)

#%% Concatenate with meta daata
house2_appliance_on_off = pd.concat([house2_meta, house2_appliance_on_off], axis=1)
        
#%% find location with loads of things on (26th May 2013)
pd.Series.idxmax(house2_appliance_on_off.sum(axis = 1))
# have now brought in flac files for that time-period



#%% Now want to read flac files for UK-dale, read data 

