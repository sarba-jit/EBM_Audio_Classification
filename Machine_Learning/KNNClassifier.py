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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import librosa
from librosa import display
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.colors


base_dir = 'BUZZ1/'
sub_dir = ['bee','noise','cricket']

def readingfiles_and_plotting_spectogram(number_of_files_to_read) :

    for i in xrange(number_of_files_to_read):
        j = random.randint(0,2)
        n = random.randint(1,125)

        file_name = base_dir + sub_dir[j] + '/' + sub_dir[j]  + str(n) + ".wav"
        sample_rate , X = wavfile.read(file_name)
        heading = sub_dir[j]  + str(n) + ".wav"
        plt.title(heading)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        specgram(X, Fs=sample_rate, xextent= None)
        plt.savefig('/BUZZ1/spectograms/' + heading + ".png", dpi = 100)
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
    os.chdir("/BUZZ1/out_of_sample/")
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

    for label,class_names in enumerate(sub_dir,start = 1):
        mvector_fft_path = os.path.join(base_dir,class_names,"mother_files","*.mvector.npy")
        all_files = glob.glob(mvector_fft_path)
        for f in all_files :
            value = np.load(f)
            D.append(value[:])
            L.append(label)

    return np.array(D),np.array(L)

''' The feature matrix below is raw and therefore called X_unscaled. Different feature scaling can be applied on it in below section'''

X_unscaled ,y = read_mother_vector()

def feature_preprocess(input):
""" Different feature scaling options are provided to run series of different experiments"""
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
    k_range = range(1,26)
    for k in k_range:
        training_data,test_data,training_label,test_label = train_test_split(X,y,test_size=0.4, random_state=4)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(training_data,training_label)
        predict = model.predict(test_data)
        accuracy_score.append(metrics.accuracy_score(test_label,predict))
    return accuracy_score


'''
Below function looks for best value of K when varied from 1 to 26. The accuraxy plots will be saved in figure folder.
'''

def plot_with_train_test():
    accuracy_score = train_test_split_evaluation_procedure()
    accuracy_score = [round(i,4) for i in accuracy_score]
    print accuracy_score,'\n'
    print (sum(accuracy_score)/len(accuracy_score)) * 100
    k_range = range(1,26)

    plt.plot(k_range,accuracy_score)
    plt.xlabel('Value of k for knn')
    plt.ylabel('Testing accuracy through train_test_split')
    plt.title('Normalization L1')
    plt.savefig('BUZZ1/figures/5.png',bbox_inches='tight')

#plot_with_train_test()

'''
Applying KNN Classification using the K-fold cross validation
'''
def kfold_evaluation_procedure():
    print 'Starting with kfold_evaluation procedure'
    kf= KFold(n_splits=10,shuffle=True)
    knn = KNeighborsClassifier()
    accuracy_score =[]
    k_accuracies =[]
    k_range = range(1,26)
    for k in k_range:
        for train_index,test_index in kf.split(X):
            X_train = np.array(X)[train_index]
            X_test = np.array(X)[test_index]
            y_train = np.array(y)[train_index]
            y_test = np.array(y)[test_index]
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train,y_train)
            predict = model.predict(X_test)
            accuracy_score.append(metrics.accuracy_score(y_test,predict))

        sum=0
        for i in range(len(accuracy_score)):
            sum = sum + accuracy_score[i]
        average_accuracy = (sum/len(accuracy_score))* 100
        k_accuracies.append(average_accuracy)
    return k_accuracies

# accuracy_score = kfold_evaluation_procedure()
# print (sum(accuracy_score)/len(accuracy_score))

def plot_with_kfold():
    k_accuracies = kfold_evaluation_procedure()
    print k_accuracies,'\n'
    print (sum(k_accuracies)/len(k_accuracies))
    k_range = range(1,26)
    #print k_accuracies
    plt.plot(k_range,k_accuracies)
    plt.xlabel('Value of k for knn')
    plt.ylabel('Testing accuracy through K fold')
    plt.title('Normalize L2')
    plt.savefig('BUZZ1/figures/5.png',bbox_inches='tight')

#plot_with_kfold()

'''
Hyper parameters tuning using Grid Search CV. To look for the best parameters by checking on the trainin data set
Checking for 2 parameters , K value for KNN and weights option
'''
def parameters_tuning():
    print 'Looking for best data on the '
    knn = KNeighborsClassifier()
    k_range = range(1,26)
    weight_options=['uniform','distance']
    param_grid = dict(n_neighbors=k_range,weights=weight_options)
    grid = GridSearchCV(knn,param_grid,cv=10,scoring="accuracy")
    training_data,test_data,training_label,test_label = train_test_split(X,y,test_size=0.4, random_state=4)
    grid.fit(training_data,training_label)
    print grid.grid_scores_
    print(grid.best_score_)
    print(grid.best_params_)

#parameters_tuning()
