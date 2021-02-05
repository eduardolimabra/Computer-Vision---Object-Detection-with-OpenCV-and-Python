#Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#Create Classifier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

#Open Image
image = cv2.imread('Images/eye_face.jpg')
fix_img = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)

#Detect Faces
faces = face_classifier.detectMultiScale(image, 1.3, 5)

#No Faces Detected
if faces is ():
    print('No Faces Found')
	
#It's a kind of Magic
def detect_face(fix_img):
    face_rects = face_classifier.detectMultiScale(fix_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(fix_img,
                     (x,y),
                     (x+w, y+h),
                     (255,0,0),
                     10)
        
    return fix_img

result = detect_face(fix_img)
plt.imshow(result)