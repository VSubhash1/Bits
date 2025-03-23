import cv2
import json
import numpy as np
import os

# Load face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load known faces and names from JSON file
def load_training_data(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    labels = []
    label_map = {}

    for i, person in enumerate(data["persons"]):
        image_path = person["image"]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_samples.append(img[y:y + h, x:x + w])
            labels.append(i)
            label_map[i] = person["name"]

    # Train the face recognizer
    if len(face_samples) > 0:
        face_recognizer.train(face_samples, np.array(labels))

    return face_recognizer, label_map

# Load trained model
recognizer, label_map = load_training_data("faces.json")

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face_roi)

        name = "Unknown"
        if confidence < 80:  # Lower confidence means better match
            name = label_map.get(label, "Unknown")

        # Draw rectangle and name
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
