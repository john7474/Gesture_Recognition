import cv2
import mediapipe as mp
import numpy as np
from Test_Data_Class import *
import time

gesture_classifier = Gesture_Classifier()

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
    
    cv2.imshow('MediaPipe Hands', cv2.flip(img, 1))
    
    key = cv2.waitKey(1)

    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
