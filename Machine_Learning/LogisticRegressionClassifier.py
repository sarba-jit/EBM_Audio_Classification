### Mother vector based classifier

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
from scikits.talkbox.features import mfcc
import librosa
from librosa import display
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.colors
import time

base_dir = 'BUZZ1/'
sub_dir = ['bee','noise','cricket']

def readingfiles_and_plotting_spectogram(number_of_files_to_read) :

    for i in xrange(number_of_files_to_read):
        j = random.randint(0,1)
        n = random.randint(1,125)

        #file_name = base_dir + sub_dir[j] + '/' + sub_dir[j]  + str(n) + ".wav"

        file_name = ['']
        sample_rate , X = wavfile.read(file_name)
        heading = sub_dir[j]  + str(n) + ".wav"
        plt.title(heading)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        specgram(X, Fs=sample_rate, xextent= None)
        plt.savefig('BUZZ1/spectograms/' + heading + ".png", dpi = 100)
        plt.show()


#readingfiles_and_plotting_spectogram(10)


def create_mother_vector(file_name):

    X, sample_rate = librosa.load(file_name)

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)

    mother_vector = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

    head, tail = os.path.split(file_name)
    add_path = os.path.join(head,"mother_files")
    name_file,ext = os.path.splitext(tail)
    new_name = name_file + ".mvector"
    new_path_file = os.path.join(add_path,new_name)
    np.save(new_path_file,mother_vector)


def generating_mother_vector_files_bee() :
    os.chdir("BUZZ1/bee/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


def generating_mother_vector_files_noise() :
    os.chdir("BUZZ1/noise/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)

def generating_mother_vector_files_cricket() :
    os.chdir("BUZZ1/cricket/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)

def generate_mother_vector_out_of_sample():
    os.chdir("BUZZ1/out_of_sample/noise_test")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)

# generating_mother_vector_files_bee()
# generating_mother_vector_files_noise()
# generating_mother_vector_files_cricket()
# generate_mother_vector_out_of_sample()

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

def read_out_of_sample():
    D = []
    L = []
    os.chdir("BUZZ1/out_of_sample/bee_test/mother_files")
    for files in glob.glob('*.mvector.npy'):
        value = np.load(files)
        D.append(value[:])
        L.append(0)
    return np.array(D),np.array(L)

X_test ,y_test = read_out_of_sample()


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

X = feature_preprocess('two')

### Applying Logistic regression using train_test_spli procedure ####

def train_test_split_evaluation_procedure():
    print 'Starting with train_test_split procedure','\n'
    #training_data,test_data,training_label,test_label=train_test_split(X,y,test_size=0.4,random_state=4)
    training_data,test_data,training_label,test_label = X,X_test,y,y_test
    model = LogisticRegression()
    model.fit(training_data,training_label)

    predict = model.predict(test_data)
    print 'Predicted values: ',predict,'\n'

    #predict_prob = model.predict_proba(test_data[:10])
    #print predict_prob

    print 'Accuracy score through sklearn is'
    print metrics.accuracy_score(test_label,predict),'\n'

    print 'The confusion matrix is'
    cm = confusion_matrix(test_label,predict)
    print cm,'\n'

    diagonal_sum=0
    for i in  cm.diagonal():
        diagonal_sum +=i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print 'The accuracy of Logistic classifier through train_test_split procedure is ',confusion_accuracy,"%" ,'\n'

#train_test_split_evaluation_procedure()


### Applying Logistic regression using the K-fold cross validation ###

def kfold_evaluation_procedure():
    print 'Starting with kfold_evaluation procedure'
    kf= KFold(n_splits=10,shuffle=True)
    accuracy_score =[]

    for train_index,test_index in kf.split(X):
        X_train = np.array(X)[train_index]
        X_test = np.array(X)[test_index]
        y_train = np.array(y)[train_index]
        y_test = np.array(y)[test_index]
        model = LogisticRegression()
        model.fit(X_train,y_train)
        predict = model.predict(X_test)
        accuracy_score.append(metrics.accuracy_score(y_test,predict))

    sum=0
    for i in range(len(accuracy_score)):
        sum = sum + accuracy_score[i]

    average_accuracy = (sum/len(accuracy_score))* 100
    print 'The accuracy of Logistic classifier through Kfold cross validation procedure is ',average_accuracy,"%"

#kfold_evaluation_procedure()
