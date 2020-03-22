#!/usr/bin/python

# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Get the path of the training set
# parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
# parser.add_argument("-i", "--image", help="Path to Image", required="True")
# args = vars(parser.parse_args())

# Load testing data
print('\nLoading MNIST Data...')
data = MNIST('./MNIST_Dataset_Loader/dataset/')
print('\nLoading testing data...')
img_test, labels_test = data.load_testing()
test_imgs = np.array(img_test)
test_labels = np.array(labels_test)
print(test_labels)
# Load the classifier
print('\nLoading classifier...')
clf, pp = joblib.load("./digits_cls_py3_L2_4x4_2x2.pkl")

# Calculate the HOG features and predict
print('\nPredicting...')
pred_labels = []
for img in test_imgs:
    hog_fd = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False, block_norm='L2')
    hog_fd = pp.transform(np.array([hog_fd], 'float64'))
    nbr = clf.predict(hog_fd)
    pred_labels.append(nbr[0])

# Validation accuracy
print('\nCalculating accuracy of predictions...')
accuracy = accuracy_score(test_labels, pred_labels)
print('\nAccuracy of classifier: ',accuracy)
# Confusion matrix
# test_labels = [0, 2, 1, 8, 9, 6, 2, 3, 4, 5, 8, 7, 7]
# pred_labels = [0, 2, 1, 7, 9, 6, 2, 3, 4, 5, 8, 7, 7]
print('\nCreating confusion matrix...')
conf_mat = confusion_matrix(test_labels,pred_labels)
# print(conf_mat)
# Plot Confusion Matrix
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9]
plt.figure()
plot_confusion_matrix(conf_mat, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()