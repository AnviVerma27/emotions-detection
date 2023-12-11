import os
import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier('C:/Users/hp/Desktop/openCV/Face recognition/haarcascade_frontalface_default.xml')

emotion = []
dir = r'C:/Users/hp/Desktop/openCV/Face recognition/New folder'
for i in os.listdir(dir):
    emotion.append(i)

features = []
labels = []

def create_train():
    for feeling in emotion:
        path = os.path.join(dir,feeling)
        label = emotion.index(feeling)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            face_rect = face_cascade.detectMultiScale(gray,1.1,3)
            
            for (x,y,w,h) in face_rect:
                faces = gray[y:y+h,x:x+h]
                features.append(faces)
                labels.append(label)

create_train()
print('training done')
print(features)




            