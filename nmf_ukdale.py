# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:11:04 2016

@author: jack
"""

#%% import packages and functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import NMF
import nimfa #has many NMF variants
import librosa #has inverse stft function

import tables as tb
import io
import glob
import soundfile as sf
from pymf import pca

os.chdir('C:/Users/Jack/Dropbox/Green Running/Bayesian Inference/')
from SourceUKDale import SourceUKDale

from datetime import datetime



#%% read data
#### N.B. remember to mount data first

def spec_maker(instant_power):
    # create spectrograms
    instant_power_spectrogram = spectrogram(instant_power,nperseg=2048, noverlap=512)[2] # from signal plots - 3000 per seg seems appropriate
    return instant_power_spectrogram


#%% create spectrogram for all hours in our dataset
start_time = datetime.now()
os.chdir('C:/Users/Jack/Desktop/ubuntu_share')
num_hours = 24
for hour in range(num_hours):
    print "Saving Spectrogram for hour %s" % hour
    giraffe = SourceUKDale(path = 'C:/Users/Jack/Desktop/ubuntu_share', house = 2, dataset_start = hour)
    giraffe.parse_next_dataset()
    file_name = 'ukdale_numpy/hour_%s' % hour
    np.save(file_name, spec_maker(giraffe.hf_raw[:,1] * giraffe.hf_raw[:,2]))
print datetime.now() - start_time

#1min 11 seconds for 2 ==> 1 hour for all 100

#%%
np.load('ukdale_numpy/hour_0.npy')

#%%
60*60 / 16055.

#%%

### LOOP through dataset start = 1,...,24 to create large spectrogram
num_hours = 1

for hour in np.arange(num_hours):

    giraffe = SourceUKDale(path = 'C:/Users/Jack/Desktop/ubuntu_share', house = 2, dataset_start = hour)
    
    # read lf data
    giraffe.parse_database()
    giraffe.subm_raw #low-frequency channel by channel dataframe
    
    # read hf data
    giraffe.parse_next_dataset() #grabs each hour of hf data    
    instant_power = (giraffe.hf_raw[:,1] * giraffe.hf_raw[:,2]) #57,543,134 rows per hour of data
    time_lims = giraffe.time_lims
    instant_power_description = pd.DataFrame(instant_power).describe()

#    # record start and end times
#    if hour == 0:
#        start_hf = int(np.floor(giraffe.hf_raw[0,0]))
#    if hour == num_hours - 1:
#        end_hf = int(np.floor(giraffe.hf_raw[-1,0]))+1
#        
#    # create spectrograms
#    instant_power_spectrogram = spectrogram(instant_power,nperseg=4096) # from signal plots - 3000 per seg seems appropriate
#    new_spec_matrix = instant_power_spectrogram[2]
#    if hour == 0:
#        spec_matrix = new_spec_matrix
#    else:
#        spec_matrix = np.concatenate((spec_matrix, new_spec_matrix), axis = 1)
        
#%% read in ukdale data
        
giraffe = SourceUKDale(path = 'C:/Users/Jack/Desktop/ubuntu_share', house = 2) #dataset_start = hour optional argument
giraffe.parse_next_dataset()


#%%
os.chdir('C:/Users/Jack/Dropbox/Green Running/figures')

#%% Current
plt.figure()
plt.plot(giraffe.hf_raw[0:2000,1])
plt.title("16kHz Current Signal")
plt.xlabel("Time steps")
plt.ylabel("Current (A)")
plt.savefig("ukdale_current.png")


#%% Voltage
plt.figure()
plt.plot(giraffe.hf_raw[0:2000,2])
plt.title("16kHz Voltage Signal")
plt.xlabel("Time steps")
plt.ylabel("Potential Difference (V)")
plt.savefig("ukdale_voltage.png")


#%% Instant Power
plt.figure()
plt.plot(giraffe.hf_raw[0:2000,2] * giraffe.hf_raw[0:2000,1])
plt.title("16kHz Instant Power Signal")
plt.xlabel("Time steps")
plt.ylabel("Instant Power")
plt.savefig("ukdale_power.png")



#%% Compute spectrogram (librosa best package for this)


# HOW TO CHOOSE SPECTROGRAM PARAMETERS
# http://uk.mathworks.com/matlabcentral/answers/104135-how-to-choose-spectrogram-parameter


n_fft = 2048 #window size

power_stft = librosa.core.stft(giraffe.hf_raw[:,1]*giraffe.hf_raw[:,2], n_fft = n_fft)

#%%
power_stft_chunk = power_stft[:, 0:5000]

#%%
power_stft_chunk_R = np.real(power_stft_chunk)
power_stft_chunk_I = np.imag(power_stft_chunk)


#%% Create positive matrices from STFT
power_stft_chunk_Rplus = power_stft_chunk_R.clip(min=0)
power_stft_chunk_Rminus = (-power_stft_chunk_R).clip(min=0)
power_stft_chunk_Iplus = power_stft_chunk_I.clip(min=0)
power_stft_chunk_Iminus = (-power_stft_chunk_I).clip(min=0)

#%%
power_chunk_spec = np.abs(power_stft_chunk)

### PCA - no hope of disentangling things??
### Try it to do it and disentangle at appropriate times

#%%
power_spec = np.abs(power_stft)


#%% Voltage stft
voltage_stft = librosa.core.stft(giraffe.hf_raw[:,2])
voltage_stft_chunk = voltage_stft[:,0:5000]
voltage_chunk_spec = np.abs(voltage_stft_chunk)


#%% PCA - non-transpose basis is best
n_components = 20
power_pca = pca.PCA(power_chunk_spec, num_bases = n_components)
power_pca.factorize()

non_trans_basis = power_pca.W

#%% Transposed
power_pca = pca.PCA(power_chunk_spec.transpose(), num_bases = n_components)
power_pca.factorize()

trans_basis = power_pca.H.transpose()


#%%
np.shape(power_pca.H)


#%% learn corresponding start and end time indices of lf data
start_lf = start_hf
end_lf = end_hf

while True:
    try:
        print giraffe.subm_raw.loc[start_lf]
        print giraffe.subm_raw.loc[end_lf]
        break
    except KeyError:
        start_lf = start_lf + 1
        end_lf = end_lf + 1



#%% sklearn basic NMF: Experiment 1
#RESULTS: doesn't look like what we want - we want channels to be constrained to be 0 or 1
## Too optimistic - perhaps the weights will be discriminative nonetheless
from datetime import datetime

startTime = datetime.now()

NMF_power = NMF(n_components = 20)
NMF_power.fit(spec_matrix)
NM_power_weights = NMF_power.components_
NM_power_basis = NMF_power.transform(spec_matrix)
NM_weights_desc = pd.DataFrame(NM_power_weights.transpose()).describe()

print NM_weights_desc

print datetime.now() - startTime # prints execution time of the cell: 4mins for 1 hour spec


#%% TRY nimfa methods.factorization.snmf #### sparse nonnegative matrix factorisation
snmf = nimfa.Snmf(spec_matrix, seed="random_vcol", rank=40, max_iter=20, version='r',
                  eta=1., beta=1e-4, i_conv=10, w_min_change=0)
snmf_fit = snmf()

#%% Looks better - sparse at least
SNMF_basis = snmf_fit.basis()
SNMF_weights = snmf_fit.coef()
SNMF_weights_desc = pd.DataFrame(SNMF_weights.transpose()).describe()
print SNMF_weights_desc


# with rank = 10 and max_iter = 12, bad results - most components just 0

#%%
librosa_spec = librosa.stft(instant_power)

#%% example signal

#%%
ones = np.array([1000] * 100000)

#%%
ones_spec = librosa.stft(ones)

#%%
np.all(spectrogram(ones)[2] == 0)

#%%
spectrogram(ones)

#%%
reconstr = librosa.istft(ones_spec)

#%%
igram = librosa.ifgram(ones_spec)



#%% plot individual channels over same time period
plt.figure()
plt.plot(giraffe.subm_raw.loc[start_lf:end_lf])
plt.title("Submeter channels")

#%% initial view of data
# plot total power
plt.figure()
plt.plot(instant_power[0:2000])
plt.title("Aggregate instant power")


#%% load in super meta data (latest one correct acc.Lionel)
import json
with open("C:/Users/Jack/Desktop/ubuntu_share/output_UKDale/House_2/events/SuperMetaData_003.json", "r") as f:
  super_meta_data = json.loads(f.read())
  
#%%
super_meta_data = super_meta_data['entries']

#%%
super_meta_data[0]['id']

#%% jupyter notebook flac_event loader (from Lionel)
# override init of soundfile, as it doesn't work properly for reading a file buffer due a checking
class mysoundfile(sf.SoundFile):
    def __init__(self, file, mode='r', format=None, samplerate=None, channels=None, closefd=True,
                 subtype=None, endian=None):
        _snd = sf._snd
        _ffi = sf._ffi
        self._name = file
        if mode is None:
            mode = getattr(file, 'mode', None)
        mode_int = sf._check_mode(mode)
        self._mode = mode
        self._info = _ffi.new("SF_INFO*")
        if samplerate is not None:
            self._info.samplerate = samplerate
            self._info.channels = channels
            self._info.format = sf._format_int(format, subtype, endian)
        self._file = self._open(file, mode_int, closefd)
        if set(mode).issuperset('r+') and self.seekable():
            # Move write position to 0 (like in Python file objects)
            self.seek(0)
        _snd.sf_command(self._file, _snd.SFC_SET_CLIPPING, _ffi.NULL, _snd.SF_TRUE)
        
        

def flac_to_array(data, scaling_factor=400.0):
    # create file buffer
    f = io.BytesIO()
    f.write(data)
    f.seek(0)
    with mysoundfile(f, format='FLAC') as sf:
        data = sf.read()
    try:
        f.close()
    except:
        pass
    return data * scaling_factor
    
database_dir = r"C:/Users/Jack/Desktop/ubuntu_share/output_UKDale/House_2/events"
os.chdir(database_dir)

evdata_h5_files = [x for x in glob.glob("*.h5") if "EventData_" in x]
evdata_h5_files

# load an event
with tb.open_file(evdata_h5_files[0], mode='r') as h5f:
    print h5f # print .h5 file information
    iv_hf_flac = h5f.root.iv_hf_flac
    iv_hf_flac = iv_hf_flac.read()
    flac = iv_hf_flac[1849] # should be the 1849th event in the house 2 .json file...

a = flac_to_array(flac)
a.shape
pd.Series(a[:,0]).plot()

#%%
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


#%%
plt.plot(movingaverage(giraffe.hf_raw[:,2] * giraffe.hf_raw[:,1], 160000))
plt.title("1 hour of power data at 0.1Hz")


#%%
split_limit = (np.shape(giraffe.hf_raw)[0] / 360) * 360

array_list = np.split(giraffe.hf_raw[0:split_limit,2] * giraffe.hf_raw[0:split_limit,1], 360)

average_power = []
for array in array_list:
    average_power.append(np.average(array))


#%%
minute_time_tags = np.arange(360) / 6.

#%%
plt.plot(minute_time_tags, average_power)
plt.title("0.1Hz average power signal over 1 hour")
plt.xlabel("Minute")
plt.ylabel("Power (W)")
plt.savefig("ukdale_average_power.png")


#%%