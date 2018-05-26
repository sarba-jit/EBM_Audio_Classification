# -*- coding: utf-8 -*-
#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

$python usingSavedModel_RawConvNet.py
This code is used to validate our RawConvNet model by testing it against the out of sample data. 
Also replace the 'n' in the first layer with your choice of filter size. 
In this paper we used 3,10,30,80,100
'''

import tflearn
from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.normalization import batch_normalization
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
import os
import utility
from audioChunking import audio_input_for_chunking

AUDIO_LENGTH = 20000
NEW_SAMPLERATE = 12000

ORIGINAL_AUDIO_FOLDER  = 'Test_Data/whole_sounds/'
CHUNKED_AUDIO_FOLDER = 'Test_Data/chunked_sounds/'
RESAMPLED_AUDIO_FOLDER = 'Test_Data/Data_Resampled/'

##################################
CHUNKED_AUDIO_FOLDER = 'out_of_sample_data/bee_test/'
# CHUNKED_AUDIO_FOLDER = 'out_of_sample_data/cricket_test/'
# CHUNKED_AUDIO_FOLDER = 'out_of_sample_data/noise_test/'
#################################


def extract_feature(file_name):
    try:
        old_samplerate, old_audio = wavfile.read(file_name)
        resampled_file = RESAMPLED_AUDIO_FOLDER + file_name.split('/')[2]
        if old_samplerate != NEW_SAMPLERATE:
            duration = old_audio.shape[0] / old_samplerate

            time_old = np.linspace(0, duration, old_audio.shape[0])
            time_new = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

            interpolator = interpolate.interp1d(time_old, old_audio.T)
            new_audio = interpolator(time_new).T

            wavfile.write(resampled_file, NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))

        sample_rate, X = wavfile.read(resampled_file)
        X = X.reshape(-1, 1)
        audio_buf = (X - np.mean(X)) / np.std(X)
        original_length = len(audio_buf)
        if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
        elif original_length > AUDIO_LENGTH:
            audio_buf = audio_buf[0:AUDIO_LENGTH]
        return audio_buf
    except:
        print 'wav error:'+file_name
        return None

network1 = input_data(shape=[None,AUDIO_LENGTH,1])
#replace filter_size 'n' with your choice of size
network2 = tflearn.layers.conv.conv_1d (network1,
                             nb_filter=256,
                             filter_size='n',
                             strides=4,
                             padding='same',
                             activation='relu',
                             bias=True,
                             weights_init='xavier',
                             bias_init='zeros',
                             regularizer='L2',
                             weight_decay=0.0001,
                             trainable=True,
                             restore=True,
                             reuse=False,
                             scope=None,
                             name='Conv1D_1')
network3 = batch_normalization(network2)
network4 = tflearn.layers.conv.max_pool_1d(network3,kernel_size=4,strides=None)
network5 = tflearn.layers.conv.conv_1d (network4,
                             nb_filter=256,
                             filter_size=3,
                             strides=1,
                             padding='same',
                             activation='relu',
                             bias=True,
                             weights_init='xavier',
                             bias_init='zeros',
                             regularizer='L2',
                             weight_decay=0.0001,
                             trainable=True,
                             restore=True,
                             reuse=False,
                             scope=None,
                             name='Conv1D_2')
network6 = batch_normalization(network5)
network7 = tflearn.layers.conv.max_pool_1d(network6,kernel_size=4,strides=None)
network8 = tflearn.layers.core.custom_layer(network7,lambda x: utility.tfmean(x,axis=1) )
network9 = fully_connected(network8,3,activation='softmax')

model = tflearn.DNN(network9, tensorboard_verbose=0)

model.load('saved_model/RawConvNet')

test_data ={}
test = 0

count_b  =0
count_c  =0
count_n  =0

for root, dirs, files in os.walk(CHUNKED_AUDIO_FOLDER):
    for item in files:
        label = 0
        if item.endswith('.wav'):
            print (item)
            x = os.path.join(root, item)
            spec_pad = extract_feature(x)
            if spec_pad is not None:
                ext_features = np.hstack([spec_pad])
                ext_features = np.array(ext_features)
                test_data[test] = ext_features
                predictions = model.predict(np.array(test_data.values()))
                # print (model.predict_label(test_data.values()))
                #is_bee = np.argmax(predictions[0]) == 1
                # print (predictions)
                if np.argmax(predictions[0]) == 1:
                     # print (item)
                     count_b = count_b + 1
                     print(item, "That's a bee sound!")
                    #shutil.move(x, final_path_bees + item)
                elif np.argmax(predictions[0]) == 0:
                     # print (item)
                     count_n = count_n + 1
                     print(item, "That's a noise!")
                     # count = count + 1
                    #shutil.move(x, final_path_noise + item)
                elif np.argmax(predictions[0]) == 2:
                     count_c = count_c + 1
                     print(item, "That's a cricket!")
                     # continue
                    #shutil.move(x, final_path_cricket + item)
            test_data.clear()
print 'bee: ',count_b,'cricket: ',count_c,'noise: ',count_n