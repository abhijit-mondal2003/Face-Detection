import cv2
import numpy as np
import face_recognition

# Load known face encodings and their corresponding names
known_face_encodings = []#this are the list
known_face_names = []

# Load your known faces and their names into the above list
known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("folder/abhijit/abhijit photo.jpg"))[0])
known_face_names.append("Abhijit")
known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("folder/Bishal/Bishal.jpeg"))[0])
known_face_names.append("Bishal")


# Initialize variables for storing face locations and names
face_locations = []
face_names = []

# Initialize the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)
video_capture.set(4,720)
print(video_capture.get(3))
print(video_capture.get(4))
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize the frame for faster face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR color to RGB color
    rgb_frame = small_frame[:, :, ::-1]

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
