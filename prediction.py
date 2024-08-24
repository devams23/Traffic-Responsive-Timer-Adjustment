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
for i in range(0,10):
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
          objectdetect = classnames[classindex]

          if objectdetect == 'car' or objectdetect == 'truck' or objectdetect == 'bus' and conf>=60:
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            new_detections = np.array([x1,y1,x2,y2,conf])
            detections = np.vstack((detections,new_detections))

  track_result = tracker.update(detections)
  for track in track_result:
    x1,y1,x2,y2,id = track
    x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
    print(id)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    cvzone.putTextRect(frame,f'{id}',(x1,y1) , thickness=2)
    print(classindex)


  cv2_imshow(frame)
  cv2.waitKey(1)