# -*- coding: utf-8 -*-
#!/usr/bin/python
'''
Author: SARBAJIT MUKHERJEE
Email: sarbajit.mukherjee@aggiemail.usu.edu

$python knn.py
Create a base directory with the name and the corresponding data.
The mother vector files should already be created.
'''

import pylab
import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import scipy.io
from scipy.io import wavfile
import os
import glob
from matplotlib.pyplot import specgram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import librosa
from librosa import display
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn import svm
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

base_dir = 'BUZZ1/'
sub_dir = ['bee','noise','cricket']

## Reading the files and creating feature and response object

def read_mother_vector():
    D = []
    L = []

    for label,class_names in enumerate(sub_dir,start = 0):
        mvector_fft_path = os.path.join(base_dir,class_names,"mother_files","*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files :
            value = np.load(f)
            D.append(value[:])
            L.append(label)

    return np.array(D),np.array(L)

X_unscaled ,y = read_mother_vector()

print len(X_unscaled)
print y

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

X = feature_preprocess('one')

#
# '''
# Below  is the function that uses OneVsAll classifier using SVM
# It also plots the ROC curves
# '''
#
def roc_multiclass(X,y):

    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    '''
    Given below are two techniques for doing train-test split
    1. train_test_split
    2. KFold evaluation
    While using one, comment out other
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,random_state=0)

# Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0))
    classifier.fit(X_train, y_train)
    filename = 'pickle/svm.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    predicted_labels = classifier.predict(X_test)
    print predicted_labels
    accuracy_score = metrics.accuracy_score(y_test,predicted_labels)
    score = classifier.score(X_test, y_test,sample_weight=None)
    print score
    print accuracy_score


roc_multiclass(X,y)

