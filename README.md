# Gesture_Recognition

Look at Hand_Gesture_List.xlsx for the gestures used and what they do
landmark_data.csv is the training data file

Mediapipe_Data.py is used to collect landmark data by pressing the key a-z which corrosponds to the gesture being added
Train_Data.py uses the landmark data to train a model and creates a tflite model
Test_Data.py tests the trained model and prints the gesture to the terminal
