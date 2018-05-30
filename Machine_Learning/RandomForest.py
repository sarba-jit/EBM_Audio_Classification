### Mother vector based classifier

import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from scipy.io import wavfile
import os
import glob
from matplotlib.pyplot import specgram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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

'''
For this --- from sklearn.cross_validation import train_test_split, it has started giving deprecation warnings. If you face any issue while
installing this module, try sklearn.model_selection for the same purpose.
'''

'''
Folder structure:
BUZZ1 (Main folder)
                 --------------------------> bee (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> noise (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> cricket (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> spectograms (sub folder)
                 --------------------------> out_of_sample_data (sub folder) --------> bee_test (sub-sub folder)
                                                                             --------> noise_test (sub-sub folder)
                                                                             --------> cricket_test (sub-sub folder)

If you plan to change the folder structure, please vary the path locations througout the file accordingly

'''

base_dir = 'BUZZ1/'
sub_dir = ['bee','noise','cricket']

def readingfiles_and_plotting_spectogram(number_of_files_to_read) :
    """ Function to generate and save spectogram figures from audio samples """

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
        plt.savefig('BUZZ1/spectograms/' + heading + ".png", dpi = 100)
        #plt.show()

'''
Function call to plot spectograms for 50 audio files.
Given folder location, it randomly selects (number of files) given as input and creates spectogram for them.
The plotted figures are then saved in directory location mentioned as a part of plt.savefig
'''
#readingfiles_and_plotting_spectogram(50)

def create_mother_vector(file_name):
    """ Function to read an audio file and extract spectral features from it to create a feature vector .npy file """
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

'''
Below are the functions for generating feature vectors (.npy files) for respective categories by reading all the files in a folder and generating
.npy corresponding to each file. Theses .npy files are then saved in a folder called mother_files.
Input : These functions take folder path as input. If you change path location, vary it accordingly in these functions.
'''

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
    """ In path, add bee_test|noise_test|cricket_test accordingly to extract respective features"""
    os.chdir("BUZZ1/out_of_sample_data/")
    for filename in glob.glob('*.wav'):
        create_mother_vector(filename)


'''
Function Calls
Below function calls are commented as features are already extracted in mother_files folder for each category.
If you plan to re-extract features, delete the files from mother_files folder and uncomment these function calls before running the code.
'''
#generating_mother_vector_files_bee()
#generating_mother_vector_files_noise()
#generating_mother_vector_files_cricket()
#generate_mother_vector_out_of_sample()


## Reading the files and creating feature and response object

def read_mother_vector():
    """ Function to read the extracted features and assigning respective target variables to them"""
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

X,y = read_mother_vector()

def read_out_of_sample():
    """ Out of sample data : Function to read the extracted features and assigning respective target variables to them"""
    D = []
    L = []
    os.chdir("BUZZ1/out_of_sample/old_bee_test/mother_files")
    for files in glob.glob('*.mvector.npy'):
        value = np.load(files)
        D.append(value[:])
        L.append(0)
    return np.array(D),np.array(L)

'''
if you need test on out of sample data, uncomment function call below.
Also, uncomment (training_data,test_data,training_label,test_label = X,X_test,y,y_test) and
comment training_data,test_data,training_label,test_label=train_test_split(X,y,test_size=0.4,random_state=4) in train_test_split_evaluation_procedure()
'''
#X_test ,y_test = read_out_of_sample()


### Applying Random Forest using train_test_spli procedure ####

def train_test_split_evaluation_procedure():
    """ Train test split procedure : 2 variations ( 60/40 train split) and on out of sample data. Comment one and uncomment other accordingly"""
    print 'Starting with train_test_split procedure','\n'
    training_data,test_data,training_label,test_label=train_test_split(X,y,test_size=0.4,random_state=4)
    #training_data,test_data,training_label,test_label = X,X_test,y,y_test

    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(training_data,training_label)

    predict = forest.predict(test_data)
    #print 'Predicted values: ',predict,'\n'

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
    print 'The accuracy of Random Forest classifier through train_test_split procedure is ',confusion_accuracy,"%" ,'\n'

'''
Function call : Comment/uncomment it, according to use.
'''

#train_test_split_evaluation_procedure()


### Applying Random Forest using the K-fold cross validation ###

def kfold_evaluation_procedure():
    print 'Starting with kfold_evaluation procedure'
    kf= KFold(n_splits=10,shuffle=True)
    accuracy_score =[]

    for train_index,test_index in kf.split(X):
        X_train = np.array(X)[train_index]
        X_test = np.array(X)[test_index]
        y_train = np.array(y)[train_index]
        y_test = np.array(y)[test_index]
        forest = RandomForestClassifier(n_estimators=100, random_state=0)
        forest.fit(X_train,y_train)
        predict = forest.predict(X_test)
        accuracy_score.append(metrics.accuracy_score(y_test,predict))

    sum=0
    for i in range(len(accuracy_score)):
        sum = sum + accuracy_score[i]

    average_accuracy = (sum/len(accuracy_score))* 100
    print 'The accuracy of Random Forest classifier  through Kfold cross validation procedure is ',average_accuracy,"%"

'''
Function call : Comment/uncomment it, according to use.
'''

#kfold_evaluation_procedure()
