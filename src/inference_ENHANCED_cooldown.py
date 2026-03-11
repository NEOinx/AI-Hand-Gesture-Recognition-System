import subprocess
import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import time
import os
from datetime import datetime
import pyautogui
from collections import deque, Counter

#########################################################################################
last_openapp_time = 0
openapp_cooldown = 10  # 10 seconds cooldown

# ===== NEW: IMAGE DISPLAY COOLDOWN =====
last_image_time = 0
image_display_cooldown = 10  # 10 seconds cooldown for image
# ========================================

###########################################################################################

def open_application():
    global last_openapp_time
    current_time = time.time()
    if current_time - last_openapp_time >= openapp_cooldown:
        try:
            subprocess.Popen(['SlideToShutDown.exe'])
            print("slide to shutdown is executed......")
            last_openapp_time = current_time
        except Exception as e:
            print(f"Failed to open application: {e}")
    else:
        remaining = openapp_cooldown - (current_time - last_openapp_time)
        print(f"Open Application cooldown active. Wait {remaining:.1f} more seconds.")

#....................................................................................................
'''
def open_application1():
    global last_openapp_time
    current_time = time.time()
    if current_time - last_openapp_time >= openapp_cooldown:
        try:
            subprocess.Popen(['Notepad.exe'])
            print("slide to shutdown is executed......")
            last_openapp_time = current_time
        except Exception as e:
            print(f"Failed to open application: {e}")
    else:
        remaining = openapp_cooldown - (current_time - last_openapp_time)
        print(f"Open Application cooldown active. Wait {remaining:.1f} more seconds.") '''
#............................................................................................................

# Load the trained model and label encoder
model = tf.keras.models.load_model('gesture_model.h5')
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands with OPTIMIZED settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,  # IMPROVED: Increased from 0.7
    min_tracking_confidence=0.6  # IMPROVED: Increased from 0.5
)

mp_drawing = mp.solutions.drawing_utils

# Screenshot functionality variables
last_screenshot_time = 0
screenshot_cooldown = 5
screenshot_dir = './screenshots'
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

def take_screenshot():
    """Take a screenshot and save it with timestamp"""
    global last_screenshot_time
    current_time = time.time()
    if current_time - last_screenshot_time >= screenshot_cooldown:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(screenshot_dir, filename)
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        print(f"✓ Screenshot saved: {filename}")
        last_screenshot_time = current_time
        return True
    else:
        remaining_time = screenshot_cooldown - (current_time - last_screenshot_time)
        print(f"Screenshot cooldown active. Wait {remaining_time:.1f} more seconds.")
        return False

def show_image():
    """Display image and handle window properly with cooldown check"""
    global last_image_time
    
    current_time = time.time()
    time_since_last = current_time - last_image_time
    
    # ===== NEW: CHECK COOLDOWN =====
    if time_since_last < image_display_cooldown:
        remaining = int(image_display_cooldown - time_since_last)
        print(f"⏳ Image display cooldown active: {remaining}s remaining...")
        return False
    # ================================
    
    try:
        img = cv2.imread('56666.jpg')
        if img is not None:
            cv2.imshow("Peace Sign Image", img)
            print("✓ Image displayed (Thumbs Up gesture)")
            cv2.waitKey(2000)
            cv2.destroyWindow("Peace Sign Image")
            print("✓ Image window closed")
            last_image_time = time.time()  # ===== NEW: Reset cooldown timer =====
            return True
        else:
            print("❌ Image file not found: 56666.jpg")
            print("   Make sure 56666.jpg is in the same directory")
            return False
    except Exception as e:
        print(f"❌ Error showing image: {e}")
        return False

# FIXED: Changed 'take_screenshot' to 'screenshot'
gesture_actions = {
    'open_palm': 'screenshot',      # FIXED: was 'take_screenshot'
    'thumbs_up': 'show_image',
    'peace_sign': 'open_application',
    'random': 'nothing'
}

print("="*80)
print("🚀 Gesture Recognition - ENHANCED VERSION WITH 10s IMAGE COOLDOWN")
print("="*80)
print("✓ Screenshot action name fixed")
print("✓ Gesture smoothing enabled (reduces confusion)")
print("✓ Confidence threshold added")
print("✓ Frame-by-frame processing for reliability")
print("✓ IMAGE DISPLAY COOLDOWN: 10 seconds (NEW!)")
print("Press 'q' to quit")
print("="*80 + "\n")

cap = cv2.VideoCapture(0)

# Resolution optimization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# LIGHTWEIGHT CURSOR SMOOTHING
cursor_buffer = deque(maxlen=2)
use_lightweight_smoothing = True
prev_x, prev_y = 0, 0
left_click_active = False
right_click_active = False

# NEW: Gesture smoothing to reduce confusion
gesture_buffer = deque(maxlen=5)  # Store last 5 predictions
last_executed_gesture = None
gesture_execution_cooldown = 1.5  # 1.5 seconds between same gesture executions
last_gesture_execution_time = 0
CONFIDENCE_THRESHOLD = 0.70  # Only execute gestures with >70% confidence

# FPS monitoring
frame_count = 0
fps_start_time = time.time()
fps_list = deque(maxlen=30)

def distance(point1, point2):
    """Calculate Euclidean distance between two landmarks"""
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def smooth_cursor_light(x, y, buffer):
    """LIGHTWEIGHT cursor smoothing - 2 point moving average only"""
    buffer.append((x, y))
    avg_x = sum(pos[0] for pos in buffer) / len(buffer)
    avg_y = sum(pos[1] for pos in buffer) / len(buffer)
    return int(avg_x), int(avg_y)

print("🔍 DIAGNOSTIC TEST")
print(f"Model loaded: {model is not None}")
print(f"Label encoder loaded: {label_encoder is not None}")
print(f"Model labels: {label_encoder.classes_}")
print(f"Camera initialized: OK")
print("Ready to detect gestures...\n")

# Main loop
print("✅ Application ready with enhanced gesture recognition!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # IMPROVED: Process gestures every frame for reliability (INFERENCE_SKIP = 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_confidence = hand_handedness.classification[0].score

            if hand_label == "Right":
                # RIGHT HAND: CURSOR CONTROL
                index_tip = hand_landmarks.landmark[8]
                screen_w, screen_h = pyautogui.size()
                cx = int(index_tip.x * screen_w)
                cy = int(index_tip.y * screen_h)

                if use_lightweight_smoothing:
                    smoothed_x, smoothed_y = smooth_cursor_light(cx, cy, cursor_buffer)
                else:
                    smoothed_x, smoothed_y = cx, cy

                pyautogui.moveTo(smoothed_x, smoothed_y)
                prev_x, prev_y = smoothed_x, smoothed_y

                thumb_tip = hand_landmarks.landmark[4]
                middle_tip = hand_landmarks.landmark[12]
                pinky_tip = hand_landmarks.landmark[20]

                # LEFT CLICK: Thumb + Pinky
                if distance(thumb_tip, pinky_tip) < 0.05:
                    if not left_click_active:
                        pyautogui.click(button='left')
                        left_click_active = True
                else:
                    left_click_active = False

                # RIGHT CLICK: Thumb + Middle
                if distance(thumb_tip, middle_tip) < 0.05:
                    if not right_click_active:
                        pyautogui.click(button='right')
                        right_click_active = True
                else:
                    right_click_active = False

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            elif hand_label == "Left" and hand_confidence > 0.8:  # High confidence it's left hand
                # LEFT HAND: GESTURE RECOGNITION WITH SMOOTHING
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data_aux = []
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - wrist_x)
                    data_aux.append(hand_landmarks.landmark[i].y - wrist_y)

                if len(data_aux) == 42:
                    # Make prediction
                    prediction = model.predict(np.asarray([data_aux]), verbose=0)
                    confidence = np.max(prediction)
                    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                    # Add to gesture buffer for smoothing
                    if confidence > CONFIDENCE_THRESHOLD:
                        gesture_buffer.append(predicted_class)

                    # Use majority voting from last 5 frames
                    if len(gesture_buffer) >= 3:  # Need at least 3 consistent readings
                        gesture_counts = Counter(gesture_buffer)
                        most_common_gesture, count = gesture_counts.most_common(1)[0]

                        # Execute only if gesture is consistent and cooldown passed
                        current_time = time.time()
                        if count >= 3:  # At least 3 out of 5 frames agree
                            # Check cooldown
                            if (most_common_gesture != last_executed_gesture or
                                    current_time - last_gesture_execution_time > gesture_execution_cooldown):

                                if most_common_gesture in gesture_actions:
                                    action = gesture_actions[most_common_gesture]
                                    print(f"Executing: {most_common_gesture} (Confidence: {confidence*100:.1f}%, Consistency: {count}/5)")

                                    # FIXED: Now checks for 'screenshot' not 'take_screenshot'
                                    if action == 'screenshot':
                                        take_screenshot()
                                        last_executed_gesture = most_common_gesture
                                        last_gesture_execution_time = current_time

                                    elif action == 'show_image':
                                        show_image()
                                        last_executed_gesture = most_common_gesture
                                        last_gesture_execution_time = current_time

                                    elif action == 'open_application':
                                        open_application()
                                        last_executed_gesture = most_common_gesture
                                        last_gesture_execution_time = current_time
                                    
                                   # elif action == 'open_application1':
                                      #  open_application1()
                                       # last_executed_gesture = most_common_gesture
                                      #  last_gesture_execution_time = current_time    

                    # Draw bounding box and label
                    x1 = int(min(x_coords) * W) - 10
                    y1 = int(min(y_coords) * H) - 10
                    x2 = int(max(x_coords) * W) + 10
                    y2 = int(max(y_coords) * H) + 10

                    # Color based on confidence
                    if confidence > 0.8:
                        color = (0, 255, 0)  # Green - high confidence
                    elif confidence > 0.6:
                        color = (0, 255, 255)  # Yellow - medium confidence
                    else:
                        color = (0, 0, 255)  # Red - low confidence

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{predicted_class} ({confidence*100:.0f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Calculate and display FPS
    current_time = time.time()
    elapsed = current_time - fps_start_time
    if elapsed >= 1:
        fps = frame_count / elapsed
        fps_list.append(fps)
        fps_start_time = current_time
        frame_count = 0

    frame_count += 1
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    status_color = (0, 255, 0) if avg_fps > 25 else (0, 165, 255) if avg_fps > 15 else (0, 0, 255)

    cv2.putText(frame, f"FPS: {avg_fps:.1f} | ENHANCED MODE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

    # ===== NEW: DISPLAY IMAGE COOLDOWN ON FRAME =====
    current_time = time.time()
    time_since_last = current_time - last_image_time
    if time_since_last < image_display_cooldown:
        remaining = int(image_display_cooldown - time_since_last)
        cv2.putText(frame, f"Image Cooldown: {remaining}s", (10, H - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # =================================================

    cv2.putText(frame, "LEFT HAND: Gestures | RIGHT HAND: Mouse Control",
                (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Gesture Recognition - ENHANCED', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*80)
print("✓ Application closed.")
print("="*80)
