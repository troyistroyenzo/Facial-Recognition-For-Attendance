# -*- coding: utf-8 -*-

"""
Author: Pastoral, Lorenzo Troy
Date Created: 03/09/2021
Face Detection for Attendance

A simple program that logs attendance and export to excel

HOW TO ADD MORE FACES: Just rename add more photos in the image attendance folder. Make
your filename as user-friendly or as readable possible (JohnMarkReyes, TroyEnzo etc.)
"""
__version__ = "1.0.1"
__email__ = "troyenzoo@gmail.com"
__status__ = "Production"
#--------------------------------------#

# Main Modules
import cv2
from face_recognition.api import face_distance
import numpy as np
import face_recognition

# Third Party Modules
import os
from datetime import datetime


class mainAttendance:

    # Choose directory where you want to store images; create array of names
    path = 'ImageAttendance'  # Create Path
    images = []  # List of Images
    classNames = []
    myList = os.listdir(path)  # List of Names
    print("PROGRAM[1]: Encoding Images...")

    # Create array of Faces
    for cls in myList:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])
    print("PROGRAM[2]: Logged faces:")
    print(classNames)

    # Compares images and checks if face exists in database | Encodes each image
    def findEncoding(images):
        encodeList = []  # Required list of images.
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
            encode = face_recognition.face_encodings(img)[0]  # Encode image
            encodeList.append(encode)  # Append to list
        return encodeList

    encodeListKnown = findEncoding(images)
    print('PROGRAM[3]: ENCODING DONE')

    # Writes Name and Timestamp to CSV File
    def markAttendance(name):
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            namesList = []
            for line in myDataList:
                entry = line.split(',')
                namesList.append(entry[0])
            if name not in namesList:
                now = datetime.now()
                dateString = now.strftime('%H:%M%S')
                f.writelines(f'\n{name},{dateString}')

    # Initialize webcam (0-3) Change value to change webcam input
    cap = cv2.VideoCapture(0)
    # cap.release()
    # cv2.destroyAllWindows()
    print('PROGRAM[4]: Intializing Webcam...')

    # Shows GUI and Compares webcam footage to database
    while True:
        success, img, = cap.read(0)  # Checks if webcam is open
        # Reduces size of image, makes processing faster
        imgS = cv2.resize(img, (0, 0), None, 0.15, 0.15)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Find faces in image
        faceCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # Iterate through faces to encodings using loop
        # Grab face and encoding then compare
        for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)  # Sends in list of known faces and compares it
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            # If image mataches, put frame and text
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # Create box and label around face
                y1, x2, y2, x1 = faceLoc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2),
                              (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Face Attendance Scanner | Webcam', img)
        cv2.waitKey(1)
