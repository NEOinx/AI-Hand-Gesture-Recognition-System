import cv2
import os
import time
import mediapipe as mp

# --- Create Folders to Store Data ---
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize MediaPipe Hands (same as inference_optimized_ENHANCED.py)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # True for static image mode during data collection
    max_num_hands=1,
    min_detection_confidence=0.75,  # MATCHED: Same as inference
    min_tracking_confidence=0.6     # MATCHED: Same as inference
)
mp_drawing = mp.solutions.drawing_utils

# Define gestures and number of images to collect
gestures = ['open_palm', 'thumbs_up', 'peace_sign', 'random']
num_images = 300

print("="*80)
print("🎥 ENHANCED DATA COLLECTION - CAMERA MATCHED TO INFERENCE")
print("="*80)
print("Camera Settings:")
print("  Resolution: 640x480 (same as inference)")
print("  FPS Cap: 60 (same as inference)")
print("  MediaPipe Settings: Matched to inference_optimized_ENHANCED.py")
print("="*80 + "\n")

for gesture in gestures:
    # Create a folder for each gesture
    gesture_path = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

    print(f'\n📁 Collecting data for gesture: {gesture}')
    print(f'   Target: {num_images} images')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # ============================================
    # MATCHED CAMERA SETTINGS FROM INFERENCE
    # ============================================
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # MATCHED: Same resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # MATCHED: Same resolution
    cap.set(cv2.CAP_PROP_FPS, 60)             # MATCHED: Same FPS cap
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # MATCHED: Same buffer size

    # Prompt user to get ready
    print('\n   ⏳ Get ready... Press "s" to start capturing.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("   ❌ Error reading frame")
            break

        # Flip for display (mirror image)
        frame_display = cv2.flip(frame, 1)

        # Draw UI text
        cv2.putText(frame_display, f'Gesture: {gesture}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_display, 'Press "s" to start', (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_display, f'Target: {num_images} images', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_display, f'Resolution: 640x480 (Matched)', (10, 480-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Data Collection - Ready', frame_display)

        if cv2.waitKey(25) == ord('s'):
            break

    # Start capturing images
    counter = 0
    print(f"   🎬 Starting capture for '{gesture}' gesture...")

    while counter < num_images:
        ret, frame = cap.read()
        if not ret:
            print("   ❌ Error reading frame during capture")
            break

        # Flip for display (mirror image)
        frame_display = cv2.flip(frame, 1)

        # Display progress
        progress = f'{counter + 1}/{num_images}'
        percentage = (counter + 1) / num_images * 100

        cv2.putText(frame_display, f'Gesture: {gesture}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_display, progress, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_display, 'Capturing...', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_display, f'{percentage:.1f}%', (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame_display, f'Resolution: 640x480 (Matched)', (10, 480-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Data Collection - Recording', frame_display)

        # Small delay between captures
        cv2.waitKey(25)

        # Save the image (use original frame, not flipped)
        img_name = os.path.join(gesture_path, f'{gesture}_{time.time()}.jpg')
        cv2.imwrite(img_name, frame)

        counter += 1

        if counter % 10 == 0:  # Print progress every 10 images
            print(f'   ✓ {counter}/{num_images} images captured for {gesture}')

    cap.release()
    cv2.destroyAllWindows()

    print(f'\n   ✅ Completed: {gesture} ({num_images} images)')

print("\n" + "="*80)
print("✅ DATA COLLECTION COMPLETE FOR ALL GESTURES!")
print("="*80)
print("\nGestures collected:")
for gesture in gestures:
    gesture_path = os.path.join(DATA_DIR, gesture)
    num_files = len([f for f in os.listdir(gesture_path) if f.endswith('.jpg')])
    print(f"  • {gesture}: {num_files} images")
print("\nNotes:")
print("  ✓ Camera settings matched to inference_optimized_ENHANCED.py")
print("  ✓ Resolution: 640x480 (same as inference)")
print("  ✓ MediaPipe confidence thresholds matched")
print("  ✓ Ready for training with consistent data")
print("="*80)
