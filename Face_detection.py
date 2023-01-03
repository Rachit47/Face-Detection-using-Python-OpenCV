import numpy as np
import cv2
from random import randrange as r

#dataset load
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#Start webcam
webcam = cv2.VideoCapture(0) # default: 0 indicates video to be captured through webcam
while True:
    success, frame = webcam.read() # success -> status wheteher image is read successfully or not
    
    #convert to black & white (grayscale)
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    faceCoordinates = trained_data.detectMultiScale(grayimg) #will consider all possible occurances for the face and grab its co-ordinates
    #face Co-ordinates => [[x,y,width,height]] = [[476 343 261 261]]
    
    for f in faceCoordinates:
        x,y,w,h = f
        cv2.rectangle(frame, (x,y),(x+w,y+h), (r(0,255),r(0,255),r(0,255)), 2) #to generate a rectangle of obtained size and random color around the face
    
    cv2.imshow('Window', frame)
    key = cv2.waitKey(1) # wait or donot close the window until a key(r or R in this case) is pressed
                         # change frame after every 1 millisecond
    if(key == 82 or key == 114):
        break

webcam.release()
print('END OF PROGRAM')