import cv2
import face_recognition
import os
import pandas as pd
from openpyxl import load_workbook

# Define branch mappings
branches = {
    "BCE": "Computer Science",
    "BEE": "Electrical",
    "BAI": "CSE with AI",
    "BEC": "Electronics",
    "BLA": "LAW"
}

# Load Excel workbook
wb = load_workbook("/Users/USER/Desktop/smart track attendance/face_names.xlsx")
ws = wb.active

# Load known faces
known_faces_dir = "/Users/USER/Desktop/smart track attendance/known_faces"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        try:
            known_image = face_recognition.load_image_file(image_path)
            # Ensure the image is RGB
            if known_image.shape[2] == 4:  # RGBA to RGB
                known_image = cv2.cvtColor(known_image, cv2.COLOR_RGBA2RGB)
            elif known_image.shape[2] == 1:  # Grayscale to RGB
                known_image = cv2.cvtColor(known_image, cv2.COLOR_GRAY2RGB)

            # Validate that the image is RGB
            if known_image.dtype != 'uint8' or known_image.shape[2] != 3:
                print(f"Skipping unsupported image format: {image_path}")
                continue
            
            # Encode face
            face_encodings = face_recognition.face_encodings(known_image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
            else:
                print(f"No faces found in image: {image_path}")
                continue
            
            # Add known face names
            if filename == "gopi.jpg":
                known_face_names.append("GOPIKRISHNA J 21BCE5155")
        
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            continue

video_capture = cv2.VideoCapture(0)
detected_names_set = set()

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    # Ensure the frame is in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
            if name not in detected_names_set:
                ind = name.index('2')
                realname = name[:ind].rstrip()
                reg_no = name[ind:].lstrip()
                branch = branches.get(name[ind+2:ind+5])
                ws.append([realname, reg_no, branch])
                detected_names_set.add(name)

        # Draw rectangles around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
wb.save("/Users/USER/Desktop/smart track attendance/face_names.xlsx")
