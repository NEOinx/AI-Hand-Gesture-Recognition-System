import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    # Skip non-directory files
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue

    print(f'Processing gesture: {dir_}')
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Normalize landmarks relative to the wrist (landmark 0)
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - wrist_x) # X coordinate
                    data_aux.append(y - wrist_y) # Y coordinate
            
            # Ensure we have data for 21 landmarks (42 values)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created and saved as data.pickle")