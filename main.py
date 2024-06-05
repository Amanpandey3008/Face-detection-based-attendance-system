import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = 'student_images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    try:
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is None:
            print(f"Failed to load image: {cl}")
            continue

        # Check image dimensions
        height, width, channels = curImg.shape
        if height == 0 or width == 0:
            print(f"Invalid dimensions for image: {cl}")
            continue

        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    except Exception as e:
        print(f"Error loading image {cl}: {e}")


def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoded_face = face_recognition.face_encodings(imgRGB)[0]
            encodeList.append(encoded_face)
        except Exception as e:
            print(f"Error encoding face: {e}")
    return encodeList


encoded_face_train = findEncodings(images)


def markAttendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')
        date = now.strftime('%d-%B-%Y')
        f.write(f'{name}, {time}, {date}\n')


# take pictures from webcam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame from webcam")
        break

    try:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faces_in_frame = face_recognition.face_locations(imgRGB)
        encoded_faces = face_recognition.face_encodings(imgRGB, faces_in_frame)
        for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
        cv2.imshow('webcam', img)
    except Exception as e:
        print(f"Error processing frame: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
