#Import Libraries 
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline 

#Create Classifiers
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

#Open Image
img = cv2.imread('Images/eye_face.jpg')

#Fix Image 
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Face Classifier
faces = face_classifier.detectMultiScale(image, 1.3, 5)

#If No Faces Detected
if faces is ():
    print('No Faces Found')
	
#It's a Kind of Magic
def detect_faces_eyes(fix_img):

	face_rects = face_classifier.detectMultiScale(fix_img)
	
	for (x,y,w,h) in face_rects:
		cv2.rectangle(fix_img,
					 (x,y),
					 (x+w, y+h),
					 (255,0,0),
					 5)
					 
	return fix_img
	
result =detect_faces_eyes(fix_img)
plt.imshow(result)