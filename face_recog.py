import cv2
import json

# Load trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trained.yml")  # Load trained model

with open("label_map.json", "r") as file:
    label_map = json.load(file)  # Load label map

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow to fix camera issues

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Extract face region
        label, confidence = recognizer.predict(roi_gray)  # Recognize face

        name = label_map.get(str(label), "Unknown")
        if confidence < 70:  # Confidence threshold
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
