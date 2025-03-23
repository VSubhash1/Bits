import cv2
import json
import numpy as np
import os

# Load images and names from JSON
def load_training_data(faces):
    with open(faces, "r") as file:
        data = json.load(file)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map = {}  # Mapping between person name and ID
    label_id = 0

    for person in data["persons"]:
        img_path = person["image"]
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_id)
            label_map[label_id] = person["name"]
            label_id += 1
        else:
            print(f"Warning: Image not found - {img_path}")

    # Train the recognizer
    if faces:
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save("face_trained.yml")  # Save trained model
        with open("label_map.json", "w") as file:
            json.dump(label_map, file)  # Save label map
        print("Training complete. Model saved.")
    else:
        print("No valid images found for training.")

# Train the model
load_training_data("faces.json")
