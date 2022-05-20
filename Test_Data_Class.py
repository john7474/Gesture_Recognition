import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

class Gesture_Classifier:
    def __init__(self, mode=False, maxHands=1, model_complex=0, detectionCon=0.5,trackCon=0.5, model_path='gesture_recognition_model.tflite', num_threads=1):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(max_num_hands=maxHands, min_detection_confidence=detectionCon, min_tracking_confidence=trackCon)

        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.img = []
        self.point_list = []
        self.GestureAvg = -1
        self.GestureCount = 0
        self.prevGesture = -1
        self.count = 1
        self.gestureList = ['Stop', 'Go', '', 1, 2, 3, 4, 5, 0,
                            'Left', 'Right', 'Forward', 'Backwards',
                            'Rotate', 'Get', '']
        
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

    def Model(self, landmark_list):

        result_index = -1

        if self.results.multi_hand_landmarks:
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()

            output_details_tensor_index = self.output_details[0]['index']

            result = self.interpreter.get_tensor(output_details_tensor_index)

            result_index = np.argmax(np.squeeze(result))

        return result_index

    def Result_Average(self, gestureID):
        if gestureID != self.prevGesture:
            self.GestureCount += gestureID
            self.count += 1
            if self.count % 51 == 0:
                self.GestureAvg = int(self.GestureCount/50)
                self.count = 1
                self.GestureCount = 0
                if self.prevGesture != self.GestureAvg and self.GestureAvg != -1:
                    self.prevGesture = self.GestureAvg
                    gesture = self.Get_Gesture_Name(self.GestureAvg)
                    print(gesture)
                    return gesture
        return -1
            
    def Get_Gesture_Name(self, gestureID):
        return self.gestureList[gestureID]




                
