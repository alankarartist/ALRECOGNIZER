import cv2
import os
import numpy as np

cwd = os.path.dirname(os.path.realpath(__file__))

def faceDetection(testImg):
    grayImg = cv2.cvtColor(testImg,cv2.COLOR_BGR2GRAY)
    xpath = os.path.join(cwd+'\AlRecognizer\HaarCascade','haarcascade_frontalface_default.xml')
    faceHaarCascade = cv2.CascadeClassifier(xpath)
    faces = faceHaarCascade.detectMultiScale(grayImg,scaleFactor=1.32,minNeighbors=5)
    return faces,grayImg

def labelsForTraining(directory):
    faces = []
    faceID = []
    for path,_,fileNames in os.walk(directory):
        for fileName in fileNames:
            if fileName.startswith("."):
                print("Skipping system file")
                continue
            id = os.path.basename(path)
            imgPath = os.path.join(path,fileName)
            print('imgPath: ',imgPath)
            print('id: ',id)
            testImg = cv2.imread(imgPath)
            if testImg is None:
                print('Image not loaded properly')
                continue
            facesRect,grayImg = faceDetection(testImg)
            if len(facesRect) != 1:
                continue
            (x,y,w,h) = facesRect[0]
            roiGray = grayImg[y:y+w,x:x+h]
            faces.append(roiGray)
            faceID.append(int(id))
    return faces,faceID

def trainClassifier(faces,faceID):
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.train(faces,np.array(faceID))
    return faceRecognizer

def drawRect(testImg,face):
    (x,y,w,h) = face
    cv2.rectangle(testImg,(x,y),(x+w,y+h),(255,0,0),thickness=4)

def putText(testImg,text,x,y):
    cv2.putText(testImg,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)