import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


video_capture = cv2.VideoCapture(0)

# Load Known faces

kanye_image = face_recognition.load_image_file("faces/Kanye.jpg")
kanye_encoding = face_recognition.face_encodings(kanye_image)[0]

steve_image = face_recognition.load_image_file("faces/steve.jpeg")
steve_encoding = face_recognition.face_encodings(steve_image)[0]


known_face_encoding = [kanye_encoding, steve_encoding]
known_face_names = ["Kanye", "Steve"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognizes faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)


        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

    # Add the text
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            lineType = 2
            thickness = 3
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font,
                        fontScale, fontColor, thickness, lineType)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])


    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()

