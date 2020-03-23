# Handwritten digit recognition

## Overview
Design software that can identify information and score on the test paper. Then automatically enter this information into the excel file. It uses image processing and machine learning to identify handwriting digit.

Sample transcript: 
<p align="center">
  <img width="460" height="200" src="https://github.com/dangthanhtung305/handwritten_digit_recognition/blob/master/image/transcript.png">
</p>

We need to detect student code, total score and rubric point

## Methods
Image processing: 
- Cut transcript frame from test paper: find and detect the biggest contour which has rectangle shape
- Use perspective transform convert transcript picture into a bird's-eye view
- Based on the edge ratio, morphological transformations and the area of the contour to find student code and total score area. Besides, we can detect rubric point
- Split student code and total score into handwritten digit pictures

Recognition handwritten digit pictures:
- HOG feature extraction
- Use SVM (Support vector machine) to recognition each digit

After that, export all the information into an excel file.

We use the MNIST database of handwritten digits for training, it has a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

## Dependencies
- Ubuntu 16.04: environment
- opencv: image processing
- PyQt5: design UI
- scikit-learn: machine learning library for python
- scikit-image: HOG feature extraction
- xlrd, xlwt: read, write excel file
- IP camera software in the smartphone: transmit a real-time picture
- python 3

## Build and run
Run software
```
python3 main.py
```
Training (import MNIST dataset)
```
python3 generateClassifier.py
```
Validation accuracy and confusion matrix
```
python3 validateClassifier.py
```
## Result
UI software
<p align="center">
  <img width="460" height="300" src="https://github.com/dangthanhtung305/handwritten_digit_recognition/blob/master/image/GUI.png">
</p>

Validation
<p align="center">
  <img width="460" height="300" src="https://github.com/dangthanhtung305/handwritten_digit_recognition/blob/master/image/confusion_maxtrix.png">
  
  
</p>
