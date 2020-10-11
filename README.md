# opencv-facerecognization

AttributeError: module 'cv2.cv2' has no attribute 'face'

# code

import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
path=("Dataset")

def getImagesAndLabels(path):
#get the path of all the files in the folder
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
#create empth face list
faceSamples=[]
#create empty ID list
Ids=[]
#now looping through all the image paths and loading the Ids and the images
for imagePath in imagePaths:
#loading the image and converting it to gray scale
pilImage=Image.open(imagePath).convert('L')
#Now we are converting the PIL image into numpy array
imageNp=np.array(pilImage,'uint8')
#getting the Id from the image
Id=int(os.path.split(imagePath)[-1].split(".")[1]) # extract the face from the training image sample
faces=detector.detectMultiScale(imageNp)
#If a face is there then append that in the list as well as Id of it
for (x,y,w,h) in faces:
faceSamples.append(imageNp[y:y+h,x:x+w])
Ids.append(Id)
return faceSamples,Ids

faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer\\trainner.yml')
cv2.destroyAllWindows()

# error getting as

Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

> > > ========= RESTART: C:/Users/Dell/Desktop/face recongisation/trainer.py =========
> > > Traceback (most recent call last):
> > > File "C:/Users/Dell/Desktop/face recongisation/trainer.py", line 5, in <module>

    recognizer = cv2.face.LBPHFaceRecognizer_create()

AttributeError: module 'cv2.cv2' has no attribute 'face'

> > >
