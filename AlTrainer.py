import cv2
import os
import numpy as np 
import AlFaceRecognition as afr 

cwd = os.path.dirname(os.path.realpath(__file__))

testImg = cv2.imread(os.path.join(cwd+'\\AlRecognizer\\TestImages','Dhoni.png'))
facesDetected,grayImg = afr.faceDetection(testImg)

print("faceDetected: ",facesDetected)

faces,faceID = afr.labelsForTraining(cwd+'\\AlRecognizer\\TrainingImages')
faceRecognizer = afr.trainClassifier(faces,faceID)
faceRecognizer.write(cwd+'\\AlRecognizer\\training.yml')

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(cwd+'\\AlRecognizer\\training.yml')

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
