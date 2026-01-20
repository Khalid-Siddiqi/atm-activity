import cv2
import numpy as np
import torch
import torch.nn as nn
import collections
import time
from ultralytics import YOLO
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
LSTM_MODEL_PATH = "atm_lstm_model_final.pth"
YOLO_MODEL_PATH = "atm.pt"
HAND_CFG = "cross-hands-yolov4-tiny.cfg"
HAND_WEIGHTS = "cross-hands-yolov4-tiny.weights"
VIDEO_SOURCE = r"C:\Users\gutech\Desktop\atm-activity\video\1.mp4"
OUTPUT_PATH = "atm_system_output.mp4"
TASK_FILE = "hand_landmarker.task"

# LSTM Config
SEQUENCE_LENGTH = 30
INPUT_SIZE = 63
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASSES = 4
LSTM_CLASSES = ["Insert", "PIN", "Take Card", "Cash"] 

# YOLOv11 Config (Used for Card/Money, but Keypad is manual now)
YOLO_CLASS_MAP = { 0: "Card", 1: "Keypad", 2: "Money" }

# Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# --- LSTM Model ---
class ATMSurveillanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ATMSurveillanceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        final_out = self.fc(lstm_out[:, -1, :])
        return final_out

# --- Helpers ---
def get_hand_box_yolov4(net, output_layers, frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    best_box = None
    max_conf = 0.0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                if confidence > max_conf:
                    max_conf = confidence
                    best_box = (x, y, x + w, y + h)
    return best_box

def is_overlapping(box1, box2, padding=50):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    b1_x1 = max(0, b1_x1 - padding)
    b1_y1 = max(0, b1_y1 - padding)
    b1_x2 += padding
    b1_y2 += padding
    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)
    if x_right < x_left or y_bottom < y_top: return False
    return True

# --- MAIN SYSTEM ---
def run_system():
    # 1. CHECK GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ GPU NOT DETECTED. Using CPU.")

    # 2. Load Models
    print("   🔹 Loading YOLOv11...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(device)

    print("   🔹 Loading YOLOv4-tiny...")
    hand_net = cv2.dnn.readNet(HAND_WEIGHTS, HAND_CFG)
    # CPU Mode for YOLOv4 to be safe
    hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    try:
        ln = hand_net.getLayerNames()
        output_layers = [ln[i - 1] for i in hand_net.getUnconnectedOutLayers()]
    except:
        ln = hand_net.getLayerNames()
        output_layers = [ln[i[0] - 1] for i in hand_net.getUnconnectedOutLayers()]

    print("   🔹 Loading LSTM...")
    lstm_model = ATMSurveillanceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm_model.eval()

    # 3. MediaPipe Setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_FILE),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    current_phase = 0 
    sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, test_frame = cap.read()
    if not ret: return
    h, w, _ = test_frame.shape
    
    # --- MANUAL KEYPAD SELECTION ---
    print("\n⚠️  ACTION REQUIRED: Draw a box around the Keypad and press ENTER or SPACE.")
    r = cv2.selectROI("SELECT KEYPAD", test_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("SELECT KEYPAD")
    
    # Save the manual box (x, y, w, h) -> (x1, y1, x2, y2)
    global_keypad_box = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3]))
    print(f"✅ Keypad Region Locked: {global_keypad_box}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    calc_timestamp_ms = 0 

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                calc_timestamp_ms += 33 

                # A. YOLOv11 (For Cards/Money only)
                yolo_results = yolo_model(frame, verbose=False, conf=0.4, device=device.index)[0]
                detected_objects = []
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    label = YOLO_CLASS_MAP.get(cls_id, "Unknown")
                    detected_objects.append(label)
                    
                    # Draw Object
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 165, 0), 2)
                    cv2.putText(frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,165,0), 2)

                # ALWAYS Draw Manual Keypad
                gx1, gy1, gx2, gy2 = global_keypad_box
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
                cv2.putText(frame, "Keypad (Manual)", (gx1, gy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

                # B. YOLOv4 (Hand)
                hand_box = get_hand_box_yolov4(hand_net, output_layers, frame)
                if hand_box:
                    x, y, x2, y2 = hand_box
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 2) 

                # C. MEDIAPIPE
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_result = landmarker.detect_for_video(mp_image, calc_timestamp_ms)
                
                if mp_result.hand_world_landmarks:
                    current_frame_features = []
                    for lm in mp_result.hand_world_landmarks[0]:
                        current_frame_features.extend([lm.x, lm.y, lm.z])
                    sequence_buffer.append(current_frame_features)
                    
                    if mp_result.hand_landmarks:
                        landmarks = mp_result.hand_landmarks[0]
                        for start_idx, end_idx in HAND_CONNECTIONS:
                            s = landmarks[start_idx]
                            e = landmarks[end_idx]
                            cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0, 255, 0), 2)
                else:
                    if len(sequence_buffer) > 0: sequence_buffer.clear()

                # D. LSTM SCORES
                probs = [0.0, 0.0, 0.0, 0.0]
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        lstm_out = lstm_model(input_tensor)
                        probs = torch.softmax(lstm_out, dim=1).cpu().numpy()[0]

                # E. VERIFIER LOGIC
                status_msg = f"Phase {current_phase}: Waiting..."
                overlap_detected = False
                
                if hand_box is not None:
                    overlap_detected = is_overlapping(global_keypad_box, hand_box, padding=50)

                CONF_THRESH = 0.40

                # Phase 1: Card
                if "Card" in detected_objects and current_phase in [0, 4]:
                    if probs[0] > CONF_THRESH:
                        current_phase = 1
                        status_msg = "✅ Phase 1: Card Inserted"

                # Phase 2: Keypad (Logic now relies on manual box)
                elif overlap_detected and current_phase >= 1:
                    if "Card" not in detected_objects:
                        if probs[1] > CONF_THRESH:
                            current_phase = 2
                            status_msg = "✅ Phase 2: Typing PIN"
                        elif current_phase == 2:
                            status_msg = "✅ Phase 2: Typing PIN (Holding)"

                # Phase 3: Card Out
                elif "Card" in detected_objects and current_phase >= 2:
                    if probs[2] > CONF_THRESH:
                        current_phase = 3
                        status_msg = "✅ Phase 3: Card Retrieved"

                # Phase 4: Cash
                elif "Money" in detected_objects:
                    if probs[3] > CONF_THRESH:
                        current_phase = 4
                        status_msg = "✅ Phase 4: Cash Retrieved"

                # F. RENDER
                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
                color = (0, 255, 0) if "✅" in status_msg else (0, 255, 255)
                cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.putText(frame, "RTX 3060 (Manual Keypad)", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                video_writer.write(frame)
                cv2.imshow("ATM Surveillance System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print("✅ Done!")

if __name__ == "__main__":
    run_system()