import numpy as np
import os
import glob
from sklearn import preprocessing
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


base_dir = 'BUZZ2/out_of_sample_data_for_validation/'
sub_dir = ['bee_test','noise_test','cricket_test']

###########generating mother files for out-of-sample data##################

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
# def generate_mother_vector_out_of_sample():
#     os.chdir("three_way_audio_test/out_of_sample_data_for_testing/noise_test/")
#     for filename in glob.glob('*.wav'):
#         print filename
#         create_mother_vector(filename)
#
# generate_mother_vector_out_of_sample()

#####################################################################

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

''' The feature matrix below is raw and therefore called X_unscaled.
Different feature scaling can be applied on it in below section'''

X_test ,y = read_mother_vector()

filename = 'pickle_modified/svm.pkl'
model = pickle.load(open(filename, 'rb'))

def train_test_split_evaluation_procedure(X_test ,y):
    Y_test = label_binarize(y, classes=[0, 1, 2])
    print 'Starting with train_test_split procedure','\n'
    predict = model.predict(X_test)
    return predict,Y_test

predicted_labels,y = train_test_split_evaluation_procedure(X_test ,y)


predicted_labels = np.argmax(predicted_labels,axis=1)
y = np.argmax(y,axis=1)
print predicted_labels,y
print("Confusion Matrix")
print(confusion_matrix(y, predicted_labels))
print("Precision")
print(precision_score(y, predicted_labels, average=None))
print("Recall")
print(recall_score(y, predicted_labels, average=None))
print("F1 score")
print(f1_score(y, predicted_labels, average=None))
print 'accuracy', accuracy_score(y, predicted_labels)