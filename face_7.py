import face_recognition
import numpy as np
import cv2
import os
import pickle
import time
print(cv2.__version__)
fpsReport=0
scaleFactor=.5
Encodings=[]
Names=[]
dispW=480
dispH=380
img1=np.zeros((480,640,1),np.uint8)
with open('/home/nano/Desktop/maths/train9.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX

cam= cv2.VideoCapture(0)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
timeStamp=time.time()
while True:

    _,frame=cam.read()
    melvin=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameSmall=cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB)
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
        top=int(top/scaleFactor)
        right=int(right/scaleFactor)
        bottom=int(bottom/scaleFactor)
        left=int(left/scaleFactor)
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2) 
        if name=='Melvin sajith':              
                   cv2.imshow('melvin',melvin)
                   cv2.moveWindow('melvin',680,0)
        else:
                       cv2.destroyWindow('melvin')
                       
    dt=time.time()-timeStamp
    fps=1/dt
    fpsReport=.90*fpsReport +1*fps
    timeStamp=time.time()
    #print('fis is:',round(fpsReport,1))
    #cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)
    #cv2.putText(frame,str(round(fpsReport,1))+'fps',(0,25),font,.75,(0,255,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()