from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition
import os
import pyttsx3

app = Flask(__name__)
# camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(0)

# defining variables to be used in the functions
engine = pyttsx3.init()  # used for calling the pyttsx3 library
text1 = "Access Granted"
text2 = "Access Denied"

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


def gen_frames():
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
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
                    cv2.line(frame, (x1, y1), (x1 + 30, y1), (255, 0, 255), 10)  # Top left
                    cv2.line(frame, (x1, y1), (x1, y1 + 30), (255, 0, 255), 10)

                    cv2.line(frame, (x2, y1), (x2 - 30, y1), (255, 0, 255), 10)  # Top right
                    cv2.line(frame, (x2, y1), (x2, y1 + 30), (255, 0, 255), 10)

                    cv2.line(frame, (x1, y2), (x1 + 30, y2), (255, 0, 255), 10)  # Bottom left
                    cv2.line(frame, (x1, y2), (x1, y2 - 30), (255, 0, 255), 10)

                    cv2.line(frame, (x2, y2), (x2 - 30, y2), (255, 0, 255), 10)  # Bottom right
                    cv2.line(frame, (x2, y2), (x2, y2 - 30), (255, 0, 255), 10)

                    cv2.putText(frame, "Access Granted", (250, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    cv2.putText(frame, name1, (x1 + 5, y2 + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
                else:

                    y1, x2, y2, x1 = face_Loca
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    # cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 255, 255), cv2.FILLED)

                    # drawing lines around the face
                    cv2.line(frame, (x1, y1), (x1 + 30, y1), (255, 0, 255), 10)  # Top left
                    cv2.line(frame, (x1, y1), (x1, y1 + 30), (255, 0, 255), 10)

                    cv2.line(frame, (x2, y1), (x2 - 30, y1), (255, 0, 255), 10)  # Top right
                    cv2.line(frame, (x2, y1), (x2, y1 + 30), (255, 0, 255), 10)

                    cv2.line(frame, (x1, y2), (x1 + 30, y2), (255, 0, 255), 10)  # Bottom left
                    cv2.line(frame, (x1, y2), (x1, y2 - 30), (255, 0, 255), 10)

                    cv2.line(frame, (x2, y2), (x2 - 30, y2), (255, 0, 255), 10)  # Bottom right
                    cv2.line(frame, (x2, y2), (x2, y2 - 30), (255, 0, 255), 10)
                    cv2.putText(frame, "Access Denied", (250, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                # engine.startLoop(True)
                if face_match[match_index]:
                    name1 = names[match_index].upper()
                    # engine.startLoop(True)
                    engine.say(name1 + text1)
                    # engine.runAndWait()
                    # engine.endLoop()
                else:
                    # engine.startLoop(False)
                    engine.say(text2)
                    # engine.runAndWait()
                    # engine.endLoop()
                engine.startLoop(False)
                engine.iterate()
                # engine.runAndWait()
                engine.endLoop()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
