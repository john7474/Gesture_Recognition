import cv2
import mediapipe as mp
import numpy as np
from Test_Data_Class import *
import time

gesture_classifier = Gesture_Classifier()
fps = 0
count = 0
start = time.time()
prev_gesture = -1
gestures_list = ['Stop', 'Go', 1, 2, 3, 4, 5, 0,
                 'Left', 'Right', 'Forward', 'Backwards',
                 'Rotate', 'Get']

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
while True:
    key = cv2.waitKey(10)
    if key == 27:
        break
    
    success, image = cap.read()

    img = gesture_classifier.Draw_Hands(image)
    hand_landmarks = gesture_classifier.Landmark_Data(img)
    landmark_list = gesture_classifier.Relative_to_Wrist(hand_landmarks)
    landmark_array = gesture_classifier.Convert_to_1D(landmark_list)
    normalised_array = gesture_classifier.Normalise_Data(landmark_array)
    gesture_id = gesture_classifier.Model(normalised_array)
    gesture = gesture_classifier.Result_Average(gesture_id)
    gesture_classifier.Gesture_Command(gesture)
    if gesture != -1:
        prev_gesture = gesture

    img = cv2.flip(img, 1)
    cv2.putText(img, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    #if prev_gesture != -1:
    cv2.putText(img, str(gestures_list[int(gesture_id)]), (400, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', img)

    count += 1
    if (time.time()-start) > 1:
        fps = round(count/(time.time()-start), 1)
        count = 0
        start = time.time()
    
    key = cv2.waitKey(1)

    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
