# -*- coding: utf-8 -*-
#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

$python knn.py
Create a base directory with the name and the corresponding data.
The mother vector files should already be created.
'''

import numpy as np
import os
import glob
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle

base_dir = 'three_way_audio_test/'
sub_dir = ['bee','noise','cricket']

## Reading the files and creating feature and response object

def read_mother_vector():
    D = []
    L = []

    for label,class_names in enumerate(sub_dir,start = 1):
        mvector_fft_path = os.path.join(base_dir,class_names,"mother_files","*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files :
            value = np.load(f)
            D.append(value[:])
            L.append(label)

    return np.array(D),np.array(L)

''' The feature matrix below is raw and therefore called X_unscaled. 
Different feature scaling can be applied on it in below section'''

X_unscaled ,y = read_mother_vector()

def feature_preprocess(input):
    if input == 'one':
        # Unscaled
        X = X_unscaled
    elif input == 'two':
        # Standard scaling
        X = preprocessing.scale(X_unscaled)
    elif input == 'three':
        # Min_max scaling
        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        X = minmax_scaler.fit_transform(X_unscaled)
    elif input == 'four':
        # Normalize L1
        X = preprocessing.normalize(X_unscaled,norm='l1')
    elif input == 'five':
        # Normalize L2
        X = preprocessing.normalize(X_unscaled,norm='l2')

    return X

'''
Chose of the option for feature scaling and give it as input below
'''
X = feature_preprocess('one')

'''
Applying KNN Classification using train_test_split procedure ####
'''
def train_test_split_evaluation_procedure():
    print 'Starting with train_test_split procedure','\n'
    accuracy_score =[]
    training_data, test_data, training_label, test_label = train_test_split(X, y, test_size=0.4, random_state=4)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(training_data, training_label)
    predict = model.predict(test_data)
    accuracy_score.append(metrics.accuracy_score(test_label, predict))
    filename = 'pickle/KNN_5.pkl'
    pickle.dump(model, open(filename, 'wb'))
    return accuracy_score

print train_test_split_evaluation_procedure()

