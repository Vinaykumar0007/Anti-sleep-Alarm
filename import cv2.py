import cv2
import imutils 
from imutils import face_utils 
import dlib 
from scipy.spatial import distance 

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B)/(2.0*C)
    return ear 

thresh = 0.25
flag = 0
frame_check = 20

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture (0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect (gray, 0)
    for subject in subjects:
        shape = predict(gray, subjects)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEndl]
        rightEye = shape[rStart:rEndl]
        leftEye = eye_aspect_ratio(leftEye)
        rightEye = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar)/2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0),1)
        if ear<thresh:
            flag+=1
            print(flag)
            if flag>=frame_check:
                cv2.putText(frame, "***ALERT***",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                flag=0
        
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)