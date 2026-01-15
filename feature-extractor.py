import cv2
import mediapipe as mp
import numpy as np
import os
import glob

# --- CONFIGURATION ---
# The parent folder containing your 4 action subfolders
DATASET_ROOT = r"C:\Users\gutech\Desktop\atm-activity\roi_video_dataset"
OUTPUT_ROOT = r"C:\Users\gutech\Desktop\atm-activity\npy_data"

# Video Config
# 3 seconds * 30 fps = 90 frames (Covers your max length)
SEQUENCE_LENGTH = 90 

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- 1. Setup MediaPipe ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the task file (Make sure this file exists in your project folder)
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print(f"❌ Error: '{model_path}' not found. Please download it from MediaPipe.")
    exit()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.1, 
    min_tracking_confidence=0.1
)

# Contrast Enhancer (Helps detecting hands on metallic ATM surfaces)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def process_video(video_path, landmarker, global_offset):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    video_data = []
    
    last_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Unique timestamp for MediaPipe video mode
        timestamp_ms = global_offset + int(1000 * frame_idx / fps)
        last_timestamp = int(1000 * frame_idx / fps)

        # 1. Enhance Contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 2. Feed to MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_enhanced)
        
        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except ValueError:
            # Skip frame if MediaPipe complains about timestamp order
            frame_idx += 1
            continue

        # 3. Extract 3D Landmarks (x, y, z)
        # We only take the first hand detected [0]
        if result.hand_world_landmarks:
            frame_features = []
            for lm in result.hand_world_landmarks[0]:
                frame_features.extend([lm.x, lm.y, lm.z]) # 21 points * 3 = 63 values
            video_data.append(frame_features)
        
        frame_idx += 1

    cap.release()
    # Return data + duration so next video starts with correct timestamp offset
    return video_data, last_timestamp + 200

def pad_sequence(sequence, target_len):
    """Pads short videos with zeros, truncates long videos."""
    arr = np.array(sequence)
    
    # Handle empty videos (no hands found)
    if len(arr) == 0: 
        return np.zeros((target_len, 63))
    
    # Pad if shorter than 600 frames
    if len(arr) < target_len:
        padding = np.zeros((target_len - len(arr), 63))
        return np.vstack((arr, padding))
    
    # Truncate if longer than 600 frames
    else:
        return arr[:target_len]

# --- MAIN EXECUTION ---
# Exact folder names corresponding to your 4 classes
classes = ["a_card_in", "b_keypad", "c_ret_card", "d_ret_cash"]

# Offset counter to keep MediaPipe timestamps increasing globally
global_offset = 0 

print(f"🚀 Starting Extraction...")
print(f"   Target Shape: ({SEQUENCE_LENGTH}, 63)")
print(f"   Looking in: {DATASET_ROOT}")

with HandLandmarker.create_from_options(options) as landmarker:
    for class_name in classes:
        folder_path = os.path.join(DATASET_ROOT, class_name)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder not found: {folder_path}")
            continue

        video_files = glob.glob(os.path.join(folder_path, "*.avi")) + glob.glob(os.path.join(folder_path, "*.mp4"))
        
        print(f"\n--- Processing Class: {class_name.upper()} ({len(video_files)} videos) ---")
        class_dataset = []
        
        for i, video_path in enumerate(video_files):
            # Extract landmarks
            raw_seq, duration = process_video(video_path, landmarker, global_offset)
            global_offset += duration
            
            # Filter: Only keep video if we found a hand in at least 10 frames
            if len(raw_seq) > 10: 
                padded_seq = pad_sequence(raw_seq, SEQUENCE_LENGTH)
                class_dataset.append(padded_seq)
                print(f"   [{i+1}/{len(video_files)}] Success: Found {len(raw_seq)} frames")
            else:
                print(f"   [{i+1}/{len(video_files)}] Skipped: Hand not detected enough.")

        if len(class_dataset) > 0:
            # Save .NPY file
            npy_path = os.path.join(OUTPUT_ROOT, f"{class_name}.npy")
            final_data = np.array(class_dataset)
            np.save(npy_path, final_data)
            print(f"✅ Saved {class_name}.npy -> Shape: {final_data.shape}")
        else:
            print(f"❌ No valid data extracted for {class_name}")

print(f"\n🎉 All Done! NPY files saved to: {OUTPUT_ROOT}")