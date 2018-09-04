### Mother vector based classifier

import numpy as np
import os
import glob
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import librosa

base_dir = 'BUZZ2/train/'
sub_dir_train = ['bee_train','noise_train','cricket_train']

base_dir_test = 'BUZZ2/test/'
sub_dir_test = ['bee_test','noise_test','cricket_test']

###########generating mother files for out-of-sample data##################
#
# def create_mother_vector(file_name):
#
#     X, sample_rate = librosa.load(file_name)
#
#     stft = np.abs(librosa.stft(X))
#     mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#     mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
#     tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
#     sr=sample_rate).T,axis=0)
#
#     mother_vector = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
#
#     head, tail = os.path.split(file_name)
#     add_path = os.path.join(head, "mother_files")
#     name_file, ext = os.path.splitext(tail)
#     new_name = name_file + ".mvector"
#     new_path_file = os.path.join(add_path, new_name)
#     np.save(new_path_file, mother_vector)
#
#
# def generate_mother_vector():
#     os.chdir('BUZZ2/test/noise_test/')
#     for filename in glob.glob('*.wav'):
#         print filename
#         create_mother_vector(filename)
#
# generate_mother_vector()

#####################################################################

## Reading the files and creating feature and response object
def read_mother_vector():
    D_train = []
    L_train = []

    for label,class_names in enumerate(sub_dir_train,start = 1):
        mvector_fft_path = os.path.join(base_dir,class_names,"mother_files","*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files :
            value = np.load(f)
            D_train.append(value[:])
            L_train.append(label)

    D_test = []
    L_test = []

    for label, class_names in enumerate(sub_dir_test, start=1):
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

'''
Applying KNN Classification using train_test_split procedure ####
'''
def train_test_split_evaluation_procedure(X_train ,Y_train, X_test, Y_test):
    print 'Starting with train_test_split procedure','\n'
    accuracy_score =[]
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    predicted_labels = model.predict(X_test)
    accuracy_score.append(metrics.accuracy_score(Y_test, predicted_labels))
    print("Confusion Matrix")
    print(confusion_matrix(Y_test, predicted_labels))
    print("Precision")
    print(precision_score(Y_test, predicted_labels, average=None))
    print("Recall")
    print(recall_score(Y_test, predicted_labels, average=None))
    print("F1 score")
    print(f1_score(Y_test, predicted_labels, average=None))
    filename = 'pickle_modified/KNN_5.pkl'
    pickle.dump(model, open(filename, 'wb'))
    return accuracy_score

print train_test_split_evaluation_procedure(X_train ,Y_train, X_test, Y_test)