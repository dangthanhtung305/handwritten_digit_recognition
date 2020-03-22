#!/usr/bin/python

# Import the modules
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import datasets, svm, metrics
from sklearn import preprocessing
import numpy as np
import datetime as dt
import time
from collections import Counter

from MNIST_Dataset_Loader.mnist_loader import MNIST

# Load the dataset
#dataset = datasets.fetch_mldata("MNIST Original")
#dataset = datasets.load_digits()
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')

print('\nLoading training data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)
# print(train_labels)

# Extract the features and labels
features = train_img #np.array(dataset.data, 'int16') 
labels = train_labels #np.array(dataset.target, 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False, block_norm='L2')
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)

print('\nCount of digits in dataset', Counter(labels))

# Create an linear SVM object
#clf = LinearSVC()
clf = svm.SVC()

# Set params
clf.set_params(kernel='linear')

# param_C = 5
# param_gamma = 0.05
# clf = svm.SVC(C=param_C,gamma=param_gamma,kernel='rbf')

# Perform the training
start_time = dt.datetime.now()
print('\nStart learning at {}'.format(str(start_time)))
clf.fit(hog_features, labels)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))

# Save the classifier
joblib.dump((clf, pp), "digits_cls_py3_L2_4x4_2x2.pkl", compress=3)
#joblib.dump(clf, "digits_cls_py3_test5.pkl", compress=3)

