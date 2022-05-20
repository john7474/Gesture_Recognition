import cv2
import mediapipe as mp
import numpy as np
import csv
from Mediapipe_Data_Class import *

data = Hand_Data_Collection()

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
while True:
    key = cv2.waitKey(10)
    if key == 27:
        break
    
    success, image = cap.read()

    img = data.Draw_Hands(image)
    hand_landmarks = data.Landmark_Data(img)
    landmark_list = data.Relative_to_Wrist(hand_landmarks)
    landmark_array = data.Convert_to_1D(landmark_list)
    normalised_array = data.Normalise_Data(landmark_array)
    class_num = data.Class_Num(key)
    data.Save_to_CSV(normalised_array, class_num)

    cv2.imshow('MediaPipe Hands', cv2.flip(img, 1))
    
cap.release()
