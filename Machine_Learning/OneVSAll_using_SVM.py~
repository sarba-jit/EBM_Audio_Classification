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
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.colors
from sklearn.grid_search import GridSearchCV

base_dir = 'three_way_audio_test/'
sub_dir = ['bee','noise','cricket']

def readingfiles_and_plotting_spectogram(number_of_files_to_read) :

    for i in xrange(number_of_files_to_read):
        j = random.randint(0,1)
        n = random.randint(1,125)

        file_name = base_dir + sub_dir[j] + '/' + sub_dir[j]  + str(n) + ".wav"
        sample_rate , X = wavfile.read(file_name)
        heading = sub_dir[j]  + str(n) + ".wav"
        plt.title(heading)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        specgram(X, Fs=sample_rate, xextent= None)
        plt.savefig('audio_test/spectograms/' + heading + ".png", dpi = 100)
        #plt.show()

#readingfiles_and_plotting_spectogram(50)

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


def generating_mother_vector_files_bee(number_of_files_to_read) :
    for i in xrange(1,number_of_files_to_read):
        file_name = base_dir + sub_dir[0] + '/' + sub_dir[0]  + str(i) + ".wav"
        create_mother_vector(file_name)


def generating_mother_vector_files_noise(number_of_files_to_read) :
    for i in xrange(1,number_of_files_to_read):
        file_name = base_dir + sub_dir[1] + '/' + sub_dir[1]  + str(i) + ".wav"
        create_mother_vector(file_name)

def generating_mother_vector_files_cricket(number_of_files_to_read) :
    for i in xrange(1,number_of_files_to_read):
        file_name = base_dir + sub_dir[2] + '/' + sub_dir[2]  + str(i) + ".wav"
        create_mother_vector(file_name)

def generate_mother_vector_out_of_sample():
    os.chdir("three_way_audio_test/out_of_sample/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)

#generating_mother_vector_files_bee(1101)
#generating_mother_vector_files_noise(1101)
#generating_mother_vector_files_cricket(1101)
#generate_mother_vector_out_of_sample()

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

# def plotting_matrix(X):
#      plt.matshow(X,fignum=int)
#
# plotting_matrix(X)
print np.shape(X)

'''
Below  is the function that uses OneVsAll classifier using SVM
It also plots the ROC curves
'''

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

    # kf= KFold(n_splits=10,shuffle=True)
    # X_train,X_test,y_train,y_test = [],[],[],[]
    # for train_index,test_index in kf.split(X):
    #     X_train = np.array(X)[train_index]
    #     X_test = np.array(X)[test_index]
    #     y_train = np.array(y)[train_index]
    #     y_test = np.array(y)[test_index]

# Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    predict = classifier.predict(X_test)
    accuracy_score = metrics.accuracy_score(y_test,predict)
    score = classifier.score(X_test, y_test,sample_weight=None)
    print score
    print accuracy_score

# Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
    for i in range(0,3):
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure()
        lw = 2
# keep varying the index as 0,1,2 to keep roc for different classes
        plt.plot(fpr[i], tpr[i], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if i == 0:
            plt.title('ROC with No Scaling : Bee class')
        elif i ==1:
            plt.title('ROC with No Scaling : Noise class')
        else:
            plt.title('ROC with No Scaling : Cricket class')
        plt.legend(loc="lower right")
        #plt.show()
        if i ==0:
            plt.savefig('three_way_audio_test/figures/roc1.png')
        elif i ==1:
            plt.savefig('three_way_audio_test/figures/roc2.png')
        else:
            plt.savefig('three_way_audio_test/figures/roc3.png')

#roc_multiclass(X,y)
