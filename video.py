import cv2 as cv
import numpy as np
from keras.models import load_model

model = load_model("handwritten digit recognition.h5")

face_cascade = cv.CascadeClassifier('C:/Users/hp/Desktop/openCV/Face recognition/haarcascade_frontalface_default.xml')

capture = cv.VideoCapture(0)

def predict(img):
    emotions = ['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise'] 
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    image = cv.resize(gray,(80,80))
    feature = np.array(image.tolist())
    feature = feature.reshape(1,80,80)
    a = model.predict(feature)
    A=a[0]
    result = max(A)
    for i in range (0,len(A)):
        if A[i]==result:
            idx = i
    
    return (emotions[idx])
    

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = predict(frame)
        cv.putText(frame,text,(x-10,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
          
    cv.imshow('text',frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()