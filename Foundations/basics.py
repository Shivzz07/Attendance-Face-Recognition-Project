import cv2
import numpy as np
import dlib
import face_recognition

imgRag = face_recognition.load_image_file('test_images/Shivam.jpg')
imgRag = cv2.cvtColor(imgRag, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('test_images/Shivam_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRag)[0]
encodeRag = face_recognition.face_encodings(imgRag)[0]
cv2.rectangle(imgRag,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
#generating the 128 measurements or encodings for Test image
encodeTest = face_recognition.face_encodings(imgTest)[0]
#face image rectangle with left right top down , coordinates
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#checking if the input face and test face are same or not from the 128 measurements or encodings
results = face_recognition.compare_faces([encodeRag], encodeTest)
# the lower the face dis the higher is both faces matched
facedis = face_recognition.face_distance([encodeRag], encodeTest)
print(results)
print(facedis)
cv2.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv2.imshow('Shivam', imgRag)
cv2.imshow('Samar', imgTest)
cv2.waitKey(0)