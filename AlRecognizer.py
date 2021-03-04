import os
import cv2
import numpy as np
import AlFaceRecognition as afr 

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(os.getcwd()+'\\AlRecognizer\\training.yml')

name = {0:'Unknown',1:'Dhoni'}

def AlRecognizer():
    cap = cv2.VideoCapture(0)
    i = 0
    a = 0
    while a < 30:
        a += 1
        ret,test_img = cap.read()
        facesDetected,grayImg = afr.faceDetection(test_img)
        cv2.waitKey(10)
        resizedImg = cv2.resize(test_img,(1000,700))
        for face in facesDetected:
            (x,y,w,h) = face
            roiGray = grayImg[y:y+h,x:x+w]
            label,confidence = faceRecognizer.predict(roiGray)
            print("confidence: ",confidence)
            print("label: ",label)
            predictedName = name[label]
            a += 1
            if confidence<70:
                i += 1
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows        
    if i >= 3:
        return 1
    else:
        return 0