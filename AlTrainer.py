import cv2
import os
import numpy as np 
import AlFaceRecognition as afr 

cwd = os.path.dirname(os.path.realpath(__file__))

ipath = os.path.join(cwd+'\\AlRecognizer\\TestImages','Dhoni.png')
testImg = cv2.imread(ipath)
facesDetected,grayImg = afr.faceDetection(testImg)

print("faceDetected: ",facesDetected)

tpath = cwd+'\\AlRecognizer\\TrainingImages'
faces,faceID = afr.labelsForTraining(tpath)
faceRecognizer = afr.trainClassifier(faces,faceID)
ypath = cwd+'\\AlRecognizer\\training.yml'
faceRecognizer.write(ypath)

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(ypath)

name = {0:'Unknown',1:'Dhoni'}

for face in facesDetected:
    (x,y,w,h) = face
    roiGray = grayImg[y:y+h,x:x+h]
    label,confidence = faceRecognizer.predict(roiGray)
    print("confidence: ",confidence)
    print("label ",label)
    afr.drawRect(testImg,face)
    predictedName = name[label]
    print(predictedName)
    if(confidence>70):
        continue
    afr.putText(testImg,predictedName,x,y)

resizedImg = cv2.resize(testImg,(1000,1000))
cv2.imshow("Face Detection Tutorial",resizedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()    
