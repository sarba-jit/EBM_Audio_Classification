### Mother vector based classifier
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

base_dir = 'BUZZ2/train/'
sub_dir_train = ['bee_train','noise_train','cricket_train']

base_dir_test = 'BUZZ2/test/'
sub_dir_test = ['bee_test','noise_test','cricket_test']

## Reading the files and creating feature and response object
def read_mother_vector():
    D_train = []
    L_train = []

    for label,class_names in enumerate(sub_dir_train,start = 0):
        mvector_fft_path = os.path.join(base_dir,class_names,"mother_files","*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files :
            value = np.load(f)
            D_train.append(value[:])
            L_train.append(label)

    D_test = []
    L_test = []

    for label, class_names in enumerate(sub_dir_test, start=0):
        mvector_fft_path = os.path.join(base_dir_test, class_names, "mother_files", "*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files:
            value = np.load(f)
            D_test.append(value[:])
            L_test.append(label)

    return np.array(D_train),np.array(L_train),np.array(D_test),np.array(L_test)

''' The feature matrix below is raw and therefore called X_unscaled.
Different feature scaling can be applied on it in below section'''

X_train ,Y_train, X_test, Y_test = read_mother_vector()

def roc_multiclass(X_train ,Y_train, X_test, Y_test):
    # print Y_train, Y_test
    Y_train = label_binarize(Y_train, classes=[0, 1, 2])
    Y_test = label_binarize(Y_test, classes=[0, 1, 2])

    #n_classes = Y_train.shape[1]

    print len(X_train),len(Y_train),len(X_test),len(Y_test)
    '''
    Given below are two techniques for doing train-test split
    1. train_test_split
    2. KFold evaluation
    While using one, comment out other
    '''

# Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0))
    classifier.fit(X_train, Y_train).decision_function(X_test)
    filename = 'pickle_modified/svm.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    predicted_labels = classifier.predict(X_test)
    print predicted_labels
    predicted_labels = np.argmax(predicted_labels,axis=1)
    Y_test = np.argmax(Y_test,axis=1)
    print Y_test,predicted_labels
    print("Confusion Matrix")
    print(confusion_matrix(Y_test, predicted_labels))
    print("Precision")
    print(precision_score(Y_test, predicted_labels, average=None))
    print("Recall")
    print(recall_score(Y_test, predicted_labels, average=None))
    print("F1 score")
    print(f1_score(Y_test, predicted_labels, average=None))
    accuracy_score = metrics.accuracy_score(Y_test,predicted_labels)
    print accuracy_score


roc_multiclass(X_train ,Y_train, X_test, Y_test)

