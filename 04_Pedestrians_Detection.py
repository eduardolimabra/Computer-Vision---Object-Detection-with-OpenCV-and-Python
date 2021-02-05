#Import Libraries
import cv2
import numpy as np

#Create Classifier
body_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

#Capture Video 

cap = cv2.VideoCapture('Video/People_Walking.mp4')

#It's a Kind of Magic
#While Loop
while cap.isOpened():

	#Read the capture
	ret, frame = cap.read()
	
	#Pass the Frame to the Classifier
	bodies = body_classifier.detectMultiScale(frame, 1.2, 3)
	
	#If Statement
	if ret == True:
	
		#Bound Boxes to Identified Bodies
		for (x,y,w,h) in bodies:
			cv2.rectangle(frame,
						 (x,y),
						 (x+w, y+h),
						 (25,125,225),
						 5)
			cv2.imshow('Pedestrians', frame)
			
		#Exit with Esc button
		if cv2.waitKey(1) == 27:
			break
			
	#else Statement
	else:
		break
		
#Release the Capture & Destroy All Windows
cap.release()
cv2.destroyAllWindows
		
		
		
		
		
		
		
		