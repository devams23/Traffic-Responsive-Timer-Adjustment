import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone
from google.colab.patches import cv2_imshow

classnames  = []
with open('classes.txt','r') as f:
    classnames = f.read().splitlines()

print(classnames)

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture("vehicles_noml.mp4")
tracker = Sort(max_age=20)
line = [0,300,850,300]
counter = []

for i in range(0,50):
  ret,frame = cap.read()
  result = model(frame,stream=1)
  detections = np.empty((0,5))

  for info in result:
      boxes = info.boxes
      for box in boxes:
          x1,y1,x2,y2 = box.xyxy[0]
          conf = box.conf[0]
          classindex = box.cls[0]
          conf = math.ceil(conf * 100)
          classindex = int(classindex)
          detectedobject = classnames[classindex]

          if detectedobject == 'car' or detectedobject == 'truck' or detectedobject == 'bus' and conf>=60:
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            new_detections = np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections,new_detections))


  track_result = tracker.update(detections)
  cv2.line(frame,(line[0],line[1]),(line[2],line[3]),(0,255,255),7)
  for track in track_result:
    x1,y1,x2,y2,id = track
    x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
    print(id)
    #calculating the center point using the x1,y1,x2,y2 (diagonals)
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
    cvzone.putTextRect(frame,f'{id}',(x1,y1) , thickness=1 ,scale=1)
    if line[0]<cx<line[2] and line[1]-15<cy<line[1]+15:
      if id not in counter:
        counter.append(id)
        cv2.line(frame,(line[0],line[1]),(line[2],line[3]),(0,0,255),7)
    cvzone.putTextRect(frame,f'Total Count: {len(counter)}',(50,50))
    print(classindex)


  cv2_imshow(frame)
  cv2.waitKey(1)