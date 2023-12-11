import cv2 as cv
import numpy as np
from keras.models import load_model

model = load_model("handwritten digit recognition.h5")

emotions = ['Anger','contempt','disgust','fear','happy','neutral','sad','surprise']

face_cascade = cv.CascadeClassifier('C:/Users/hp/Desktop/openCV/Face recognition/haarcascade_frontalface_default.xml')

img = cv.imread("sample.jpg")
cv.imshow('orignal',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

image = cv.resize(gray,(80,80))
cv.imshow('resized',image)

feature = np.array(image.tolist())

feature = feature.reshape(1,80,80)

a = model.predict(feature)
A = a[0]

result = max(A)
for i in range (0,len(A)):
    if A[i]==result:
        idx = i

print(emotions[idx])


cv.waitKey(0)

