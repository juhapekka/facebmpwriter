#!/usr/bin/env python3

import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


save_images = False  # Flag to start saving images
face_rect = None  # To store the face rectangle dimensions
img_counter = 0  # Initialize counter for image filenames

# Function to detect the face
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces
        faces = detect_face(frame)

        for (x, y, w, h) in faces:
            if save_images and face_rect:
                # Use the dimensions of the first detected face after pressing 's'
                x, y, w, h = face_rect
            else:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if not save_images:
                    face_rect = (x, y, w, h)

            if save_images:
                # Save image of the face as BMP
                face_img = frame[y:y+h, x:x+w]
                filename = f'face_{img_counter}.bmp'
                cv2.imwrite(filename, face_img)
                print(f'Image saved: {filename}')
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

                img_counter += 1  # Increment the counter after saving an image
            
            break

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to exit
        if key == ord('q'):
            break
        # Press 's' to start saving images
        elif key == ord('s'):
            save_images = True

finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

