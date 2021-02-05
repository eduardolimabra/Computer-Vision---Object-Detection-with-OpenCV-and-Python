#Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#Open Image
img = cv2.imread('Images/eye_face.jpg')

#Show Image
plt.imshow(img)

#Fix Image
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Show Image
plt.imshow(fix_img)

#Create Classifier
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

#It's a Kind of Magic
def detect_eyes(fix_img):
    eyes_rects = eye_classifier.detectMultiScale(fix_img)
    
    for (x,y,w,h) in eyes_rects:
        cv2.rectangle(fix_img,
                     (x,y),
                     (x+w, y+h),
                     (255,255,255),
                      10)
    return fix_img
	
#Show Results
result = detect_eyes(fix_img)
plt.imshow(result)