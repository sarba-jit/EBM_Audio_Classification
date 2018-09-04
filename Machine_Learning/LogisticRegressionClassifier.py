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
import pickle
from sklearn.linear_model import LogisticRegression
'''
For this --- from sklearn.cross_validation import train_test_split, it has started giving deprecation warnings. If you face any issue while
installing this module, try sklearn.model_selection for the same purpose.
'''

'''
Folder structure:
BUZZ2 (Main folder)
                 --------------------------> bee (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> noise (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> cricket (sub folder) ------> mother_files (sub-sub folder)
                 --------------------------> spectograms (sub folder)
                 --------------------------> out_of_sample_data (sub folder) --------> bee_test (sub-sub folder)
                                                                             --------> noise_test (sub-sub folder)
                                                                             --------> cricket_test (sub-sub folder)
If you plan to change the folder structure, please vary the path locations througout the file accordingly
'''

base_dir = 'BUZZ2/train/'
sub_dir_train = ['bee_train','noise_train','cricket_train']

base_dir_test = 'BUZZ2/test/'
sub_dir_test = ['bee_test','noise_test','cricket_test']


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


X_train ,Y_train, X_test, Y_test = read_mother_vector()

### Applying Logistic Regression using train_test_spli procedure ####

def train_test_split_evaluation_procedure(X_train ,Y_train, X_test, Y_test):
    print 'Starting with train_test_split procedure','\n'

    model = LogisticRegression()
    model.fit(X_train ,Y_train)

    filename = 'pickle_modified/L.pkl'
    pickle.dump(model, open(filename, 'wb'))

    predict = model.predict(X_test)
    print 'Predicted values: ',predict,'\n'

    print 'Accuracy score through sklearn is'
    print metrics.accuracy_score(Y_test,predict),'\n'

    print 'The confusion matrix is'
    cm = confusion_matrix(Y_test,predict)
    print cm,'\n'

    diagonal_sum=0
    for i in  cm.diagonal():
        diagonal_sum +=i

    confusion_accuracy = ((diagonal_sum)/float(cm.sum())) * 100
    print 'The accuracy of Logistic classifier through train_test_split procedure is ',confusion_accuracy,"%" ,'\n'

'''
Function call : Comment/uncomment it, according to use.
'''

train_test_split_evaluation_procedure(X_train ,Y_train, X_test, Y_test)


