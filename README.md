# Handwritten digit recognition

## Overview
Design software that can identify information and score on the test paper. Then automatically enter this information into the excel file. It uses image processing and machine learning to identify handwriting digit.

Sample transcript: 


We need to detect student code, total score and rubric point

### Methods
Image processing: 
- Cut transcript frame from test paper: find and detect the biggest contour which has rectangle shape
- Use perspective transform convert transcript picture into a bird's-eye view
- Based on the edge ratio, morphological transformations and the area of the contour to find student code and total score area. Besides, we can detect rubric point
- Split student code and total score into handwritten digit pictures

Recognition handwritten digit pictures:
- HOG feature extraction
- Use SVM (Support vector machine) to recognition each digit

