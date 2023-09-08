import cv2
import os
import numpy as np
import AlFaceRecognition as afr

cwd = os.path.dirname(os.path.realpath(__file__))

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
tpath = os.path.join(cwd+'\AlRecognizer','training.yml')
faceRecognizer.read(tpath)
name = {0:'Unknown',1:'Dhoni'}

cap = cv2.VideoCapture(0)
while True:
    ret, testImg = cap.read()
    facesDetected,grayImg = afr.faceDetection(testImg)
    for(x,y,w,h) in facesDetected:
        cv2.rectangle(testImg,(x,y),(x+w,y+h),(255,0,0),thickness=6)
    resizedImg = cv2.resize(testImg,(1000,700))
    cv2.imshow('Face Detection Tutorial ',resizedImg)
    cv2.waitKey(10)

    for face in facesDetected:
        (x,y,w,h) = face
        roiGray = grayImg[y:y+w,x:x+h]
        label,confidence = faceRecognizer.predict(roiGray)
        print(confidence,label)
        afr.drawRect(testImg,face)
        predictedName = name[label]
        if confidence>70:
            afr.putText(testImg,predictedName,x,y)

    resizedImg = cv2.resize(testImg,(1000,700))
    cv2.imshow('AlRecognition',resizedImg)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

