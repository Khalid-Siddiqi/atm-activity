import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# --- 1. Kalman Filter (Stabilizer) ---
class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array(np.eye(4, 8), dtype=np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.first_frame = True
        self.missed_frames = 0

    def update(self, box):
        x, y, w, h = box
        measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(w)], [np.float32(h)]])
        if self.first_frame:
            self.kf.statePre = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre
            self.first_frame = False
            return box
        self.kf.correct(measurement)
        self.missed_frames = 0
        prediction = self.kf.predict()
        return self._get_box(prediction)

    def predict_only(self):
        prediction = self.kf.predict()
        self.missed_frames += 1
        return self._get_box(prediction)

    def _get_box(self, state):
        return [int(state[0]), int(state[1]), int(state[2]), int(state[3])]

# --- 2. Configuration ---
yolo_cfg = "cross-hands-yolov4-tiny.cfg"
yolo_weights = "cross-hands-yolov4-tiny.weights"
task_path = "hand_landmarker.task"

if not all(os.path.exists(f) for f in [yolo_cfg, yolo_weights, task_path]):
    print("❌ Error: Missing configuration files.")
    sys.exit()

# Load YOLO (FORCE CPU TO PREVENT CRASH)
print("⚙️ Loading YOLO on CPU (Standard OpenCV detected)...")
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model_yolo = cv2.dnn_DetectionModel(net)
model_yolo.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=task_path), running_mode=VisionRunningMode.VIDEO, num_hands=1)

# --- 3. Video Processing ---
VIDEO_PATH = r'C:\Users\Yousuf Traders\Desktop\Projects\atm\atm_video\3.mp4'
OUTPUT_PATH = "output_final.avi" 

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Cannot open {VIDEO_PATH}")
    sys.exit()

# Get dimensions from first frame (Safest method)
ret, first_frame = cap.read()
if not ret: sys.exit()
height, width, _ = first_frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind

fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use MJPG Codec (Universal compatibility)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

tracker = KalmanTracker()
frame_idx = 0

print(f"🎬 Processing {total_frames} frames. Output will be: {OUTPUT_PATH}")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        base_timestamp_ms = int(1000 * frame_idx / fps)
        h, w, _ = frame.shape
        final_box = None

        # A. YOLO
        classes, scores, boxes = model_yolo.detect(frame, confThreshold=0.2, nmsThreshold=0.4)
        if len(boxes) > 0:
            best_idx = np.argmax(scores)
            final_box = tracker.update(boxes[best_idx])
        else:
            if not tracker.first_frame and tracker.missed_frames < 10:
                final_box = tracker.predict_only()

        # B. MediaPipe
        if final_box is not None:
            bx, by, bw, bh = final_box
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

            cx, cy = bx + bw // 2, by + bh // 2
            side = int(max(bw, bh) * 1.5)
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(w, x1 + side), min(h, y1 + side)
            
            hand_crop = frame[y1:y2, x1:x2]
            if hand_crop.size > 0:
                hand_crop_resized = cv2.resize(hand_crop, (256, 256))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_crop_resized)
                result = landmarker.detect_for_video(mp_image, base_timestamp_ms)

                # ... inside the loop ...
                if result.hand_landmarks:
                    print(f"✅ Hand Detected! Frame {frame_idx}") # <--- ADD THIS
                    
                    for n_landmarks in result.hand_landmarks:
                        # Print the first landmark (Wrist) coordinates to verify data
                        wrist = n_landmarks[0] 
                        print(f"   Wrist Coords -> X: {wrist.x:.4f}, Y: {wrist.y:.4f}, Z: {wrist.z:.4f}") # <--- ADD THIS

                        for nl in n_landmarks:
                            # (Existing drawing code...)
                            orig_x = int(nl.x * (x2 - x1) + x1)
                            orig_y = int(nl.y * (y2 - y1) + y1)
                            cv2.circle(frame, (orig_x, orig_y), 4, (0, 255, 0), -1)

        out.write(frame)
        
        if frame_idx % 30 == 0:
            print(f"Processing: {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)", end='\r')
            cv2.imshow('ATM Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Success! Video saved to {OUTPUT_PATH}")