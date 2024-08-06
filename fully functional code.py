import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound file
pygame.mixer.music.load(r"C:\Users\vinay\collegefold\projecxt\onefile\music.wav")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_indices):
    # Extract the coordinates of the eye landmarks
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])

    # Calculate the distances between the horizontal and vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    # Compute the Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Indices for the left and right eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Threshold for EAR to detect drowsiness
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 45  # 2 seconds at 30 fps

# Initialize the frame counter and the drowsiness flag
frame_counter = 0
drowsy = False

# Time threshold for no face detection
NO_FACE_TIME_THRESHOLD = 10  # seconds
no_face_start_time = None

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find the face landmarks
    results = face_mesh.process(rgb_frame)

    # Check if any face is detected
    if results.multi_face_landmarks:
        no_face_start_time = None  # Reset the no face timer

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                frame_counter += 1

                # If the EAR is below the threshold for a sufficient number of frames, alert the driver
                if frame_counter >= EAR_CONSEC_FRAMES:
                    drowsy = True
                    cv2.putText(frame, "***DROWSINESS ALERT***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    # Play the sound alert
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                frame_counter = 0
                drowsy = False
                pygame.mixer.music.stop()
    else:
        # If no face is detected, start the no face timer
        if no_face_start_time is None:
            no_face_start_time = time.time()
        elif time.time() - no_face_start_time > NO_FACE_TIME_THRESHOLD:
            # If no face is detected for the threshold time, alert the driver
            cv2.putText(frame, "***NO FACE DETECTED ALERT***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            # Play the sound alert
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

    # Display the image with landmarks
    cv2.imshow('MediaPipe FaceMesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
