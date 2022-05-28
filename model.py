# Authentication system using Face recognition

# importing all the required libraries
import cv2
import numpy as np
import face_recognition
from datetime import datetime
# import time
import os
import pyttsx3  # used for converting text to speech

# defining variables to be used in the functions
engine = pyttsx3.init()  # used for calling the pyttsx3 library
path = 'dataset'
images = []
names = []
my_list = os.listdir(path)
# print(my_list)

# creating the list of names of people present in the database
for current_image in my_list:
    image_list = cv2.imread(f'{path}/{current_image}')
    images.append(image_list)
    names.append(os.path.splitext(current_image)[0])
print(names)


# encoding the images present in the dataset
def encode_face(images):
    encoding_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)[0]
        encoding_list.append(encodes)
    return encoding_list


encodes_known = encode_face(images)
print("Encoding is success!!")


# creating an access log to keep the track of the working of application
def accesslog(name_test):
    with open('accesslog.csv', 'r+') as f:
        log_details = f.readlines()
        access_list = []
        for line in access_list:
            enter_into_log = line.split(',')
            access_list.append(enter_into_log[0])
        # namelist = []

        if name_test not in access_list:
            # namelist.append(name_test)
            time_now = datetime.now()
            t_str = time_now.strftime('%H:%M:%S')
            d_str = time_now.strftime('%d/%m/%y')
            f.writelines(f'{name_test},{t_str},{d_str}\n')


# initialing/calling the camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    read_face = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    read_face = cv2.cvtColor(read_face, cv2.COLOR_BGR2RGB)

    current_face_frame = face_recognition.face_locations(read_face)
    current_face_encode = face_recognition.face_encodings(read_face, current_face_frame)

    # calculating the face distance and comparing the input face with the existing ones for authentication
    for ef, face_Loca in zip(current_face_encode, current_face_frame):
        face_match = face_recognition.compare_faces(encodes_known, ef)
        face_dist = face_recognition.face_distance(encodes_known, ef)

        match_index = np.argmin(face_dist)

        if face_match[match_index]:
            name1 = names[match_index].upper()
            print(name1)
            y1, x2, y2, x1 = face_Loca
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.rectangle(frame, (x1, y2 - 40), (x2, y2), (255, 255, 255), cv2.FILLED)

            # drawing lines around the face
            cv2.line(frame, (x1, y1), (x1+30,y1), (255,0,255), 10) #Top left
            cv2.line(frame, (x1, y1), (x1,y1+30), (255,0,255), 10)

            cv2.line(frame, (x2, y1), (x2-30,y1), (255,0,255), 10) #Top right
            cv2.line(frame, (x2, y1), (x2,y1+30), (255,0,255), 10)

            cv2.line(frame, (x1, y2), (x1+30,y2), (255,0,255), 10) #Bottom left
            cv2.line(frame, (x1, y2), (x1,y2-30), (255,0,255), 10)

            cv2.line(frame, (x2, y2), (x2-30,y2), (255,0,255), 10) #Bottom right
            cv2.line(frame, (x2, y2), (x2,y2-30), (255,0,255), 10)

            cv2.putText(frame, "Access Granted", (250, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            cv2.putText(frame, name1, (x1 + 5, y2 + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            engine.say("Access Granted")
            engine.runAndWait()
            accesslog(name1)
            # time.sleep(10)
        else:
            y1, x2, y2, x1 = face_Loca
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)

            # drawing lines around the face
            cv2.line(frame, (x1, y1), (x1 + 30, y1), (255, 0, 255), 10)  # Top left
            cv2.line(frame, (x1, y1), (x1, y1 + 30), (255, 0, 255), 10)

            cv2.line(frame, (x2, y1), (x2-30,y1), (255,0,255), 10) #Top right
            cv2.line(frame, (x2, y1), (x2,y1+30), (255,0,255), 10)

            cv2.line(frame, (x1, y2), (x1 + 30, y2), (255, 0, 255), 10)  # Bottom left
            cv2.line(frame, (x1, y2), (x1, y2 - 30), (255, 0, 255), 10)

            cv2.line(frame, (x2, y2), (x2-30,y2), (255,0,255), 10) #Bottom right
            cv2.line(frame, (x2, y2), (x2,y2-30), (255,0,255), 10)
            cv2.putText(frame, "Access Denied", (250, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            engine.say("Access Denied")
            engine.runAndWait()
            # time.sleep(10)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(10) == 13:
        break
cam.release()
cv2.destroyAllWindows()
