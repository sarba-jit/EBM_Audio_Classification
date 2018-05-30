#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

$python ConvNet1.py > saved_model/ConvNet1.txt
This code generates the audio classifcation model without the custom layer and replaces it with a FC layer 
with 256 units. Also replace the 'n' in the first layer with your choice of filter size. 
In this paper we used 3,10,30,80,100
The model is saved in a directory named saved_model
'''

from tflearn.layers.estimator import regression
import pydub
pydub.AudioSegment.ffmpeg = '/usr/bin/ffmpeg'
import pickle
import tflearn
from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.normalization import batch_normalization
import utility


pickle_path = 'pickle/'
############Loading from pickle#############################
f = open(pickle_path + 'train_data.pickle')
X = pickle.load(f)
f.close()

f = open(pickle_path + 'train_labels.pickle')
Y = pickle.load(f)
f.close()

###########################################################

AUDIO_LENGTH = 20000

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

#network8 = tflearn.layers.core.custom_layer(network7,lambda x: utility.tfmean(x,axis=1) )
network8 = fully_connected(network7,256,activation='softmax')
network9 = fully_connected(network8,3,activation='softmax')
network10 = regression(network9, optimizer='adam',
                       loss='categorical_crossentropy',learning_rate=0.0001)

model = tflearn.DNN(network10, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=100, validation_set=0.3, snapshot_step=400, shuffle= True,
          show_metric=True, batch_size=128,run_id='ConvNet1')

model.save('saved_model/ConvNet1')