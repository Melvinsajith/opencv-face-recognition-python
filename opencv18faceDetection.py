import numpy as np
import cv2
import os

dispW=420
dispH=380
#cam = cv2.VideoCapture('nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade=cv2.CascadeClassifier('/home/melvin/Desktop/python/CASDES/eye.xml')
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    #eye=eye_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    #for (x,y,w,h) in eye:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    framesmall=cv2.resize(frame,(640,480))
    cv2.imshow('frame', framesmall)
    cv2.moveWindow('frame',0,0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
