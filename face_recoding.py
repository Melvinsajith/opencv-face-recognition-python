import face_recognition
import cv2
import os
import pickle
import time
print(cv2.__version__)
fpsReport=0
scaleFactor=.5
Encodings=[]
Names=[]
dispW  =640
dispH= 480
font=cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

print('Enter a Long Name')
number=str(input('Enter a name to save the video:'))
filename ='video'+number+'.avi'
frames_per_sec =19
myres = '720'

def change_res(cam, width, height):
    cam.set(3, width)
    cam.set(4, height)

STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
def get_dims(cam,res='480p'):
    width,height=STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width,height =STD_DIMENSIONS[res]
    change_res(cam,width,height)
    return width,height

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}
def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
#cap = cv2.VideoCapture(0)
#cam= cv2.VideoCapture(0)
dims = get_dims(cam, res=myres)
video_type_c2 = get_video_type(filename)

out =cv2.VideoWriter('/home/nano/Desktop/maths/'+filename,video_type_c2,frames_per_sec,dims)

with open('/home/nano/Desktop/maths/train9.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX


#cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
timeStamp=time.time()
while True:

    _,frame=cam.read()

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

    dt=time.time()-timeStamp
    fps=1/dt
    fpsReport=.90*fpsReport +1*fps
    timeStamp=time.time()
    out.write(frame)
    
     
    #print('fis is:',round(fpsReport,1))
    #cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)
    #cv2.putText(frame,str(round(fpsReport,1))+'fps',(0,25),font,.75,(0,255,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()