import mediapipe as mp
import cv2
import numpy as np
import csv

class Hand_Data_Collection:
    def __init__(self, mode=False, maxHands=2, model_complex=0, detectionCon=0.5,trackCon=0.5):
        
        self.mode = mode
        self.model_complex = model_complex
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands()

        self.img = []
        self.point_list = []
        self.count = 0
        self.keyVal = -1
        
    def Draw_Hands(self, img, draw=True):
        self.img = img
        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks,
                                                   self.mp_hands.HAND_CONNECTIONS)
        
        return img

    def Landmark_Data(self, img):
        hand_landmarks = []
        img_h, img_w = img.shape[0], img.shape[1]

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                for index, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * img_w), img_w - 1)
                    landmark_y = min(int(landmark.y * img_h), img_h - 1)
                    
                    hand_landmarks.append([index, landmark_x, landmark_y])

        return hand_landmarks

    def Relative_to_Wrist(self, landmarks):
        landmark_list = []
        
        wrist_x, wrist_y = 0, 0

        if self.results.multi_hand_landmarks:
            for index, landmark in enumerate(landmarks):
                if landmark[0] == 0:
                    wrist_x, wrist_y = landmark[1], landmark[2]

                landmarks[index][1] = landmarks[index][1] - wrist_x
                landmarks[index][2] = landmarks[index][2] - wrist_y

                landmark_list.append(landmarks[index][1:3])
                
        return landmark_list

    def Convert_to_1D(self, landmark_list):
        temp_landmark_list = np.array(landmark_list)
        landmark_array = temp_landmark_list.flatten()

        return landmark_array

    def Normalise_Data(self, landmark_array):
        normalised_list = []

        if self.results.multi_hand_landmarks:
            normalised_list = landmark_array/np.linalg.norm(landmark_array)
                
        return normalised_list

    def Class_Num(self, key):
        class_num = -1
        
        if self.results.multi_hand_landmarks:
            if 97 <= key <= 122:
                if self.keyVal != key:
                    self.keyVal = key
                    self.count = 0
                else:
                    self.count += 1
                    class_num = key - 97
                print(self.count)
                

        return class_num

    def Save_to_CSV(self, norm_array, class_num):
        if self.results.multi_hand_landmarks:
            if class_num != -1:
                with open('landmark_data.csv', 'a', newline="") as f:
                    write = csv.writer(f)
                    write.writerow([class_num, *norm_array])


                
