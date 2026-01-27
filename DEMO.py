import cv2
import numpy as np
import torch
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "atm.pt"
HAND_CFG = "cross-hands-yolov4-tiny.cfg"
HAND_WEIGHTS = "cross-hands-yolov4-tiny.weights"
VIDEO_SOURCE = r"C:\Users\gutech\Desktop\atm-activity\video\1.mp4"
# Using 'mp4v' is the safest native option for .mp4 without external DLLs
OUTPUT_PATH = "atm_client_demo_final.mp4" 
SIDEBAR_WIDTH = 350  

# Visual Colors (BGR Format)
COLORS = {
    "Card": (255, 100, 0),    # Blue
    "Keypad": (0, 165, 255),  # Orange
    "Money": (0, 255, 0),     # Green
    "Hand": (0, 255, 255)     # Yellow
}

YOLO_CLASS_MAP = { 0: "Card", 1: "Keypad", 2: "Money" }

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

def is_overlapping(box1, box2, padding=30):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    b1_x1 = max(0, b1_x1 - padding); b1_y1 = max(0, b1_y1 - padding)
    b1_x2 += padding; b1_y2 += padding
    x_left = max(b1_x1, b2_x1); y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2); y_bottom = min(b1_y2, b2_y2)
    return not (x_right < x_left or y_bottom < y_top)

# --- MAIN SYSTEM ---
def run_system():
    # Load Models
    yolo_model = YOLO(YOLO_MODEL_PATH)
    if torch.cuda.is_available(): yolo_model.to('cuda')

    hand_net = cv2.dnn.readNet(HAND_WEIGHTS, HAND_CFG)
    hand_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    hand_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    ln = hand_net.getLayerNames()
    output_layers = [ln[i - 1] for i in hand_net.getUnconnectedOutLayers()]

    # Tracking
    checklist = [False] * 4
    task_names = ["Card Insertion", "PIN Entry", "Card Retrieval", "Cash Withdrawal"]
    active_p1 = False
    active_p3 = False

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    ret, test_frame = cap.read()
    if not ret: return
    h, w, _ = test_frame.shape
    
    # Calibration
    cv2.namedWindow("CALIBRATION", cv2.WINDOW_NORMAL)
    r = cv2.selectROI("CALIBRATION", test_frame, False, False)
    cv2.destroyWindow("CALIBRATION")
    
    if r[2] == 0 or r[3] == 0:
        global_keypad_box = (0, 0, 0, 0)
    else:
        global_keypad_box = (int(r[0]), int(r[1]), int(r[0]+r[2]), int(r[1]+r[3]))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    new_width = w + SIDEBAR_WIDTH

    # --- CODEC: mp4v for WhatsApp compatibility ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (new_width, h))

    pin_frames = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. RUN YOLOv11 (GPU)
        results = yolo_model(frame, verbose=False, conf=0.4)[0]
        detected_objects = []

        # 2. DRAW YOLOv11 DETECTIONS (FOR CLIENTS)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = YOLO_CLASS_MAP.get(cls_id, "Unknown")
            conf = float(box.conf[0])
            detected_objects.append(label)
            
            # Draw Box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = COLORS.get(label, (255, 255, 255))
            
            # Don't draw "Keypad" detection box if you want to keep it totally clean,
            # but usually clients like seeing that the AI detects the keypad object itself.
            # I will keep object detection boxes, but I have removed the "ROI" box below.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label_text = f"{label} {conf:.2f}"
            (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 3. RUN & DRAW HAND DETECTION
        hand_box = get_hand_box_yolov4(hand_net, output_layers, frame)
        if hand_box:
            x, y, x2, y2 = hand_box
            cv2.rectangle(frame, (x, y), (x2, y2), COLORS["Hand"], 2)
            cv2.putText(frame, "Hand", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["Hand"], 2)

        # --- LOGIC GATES ---
        # Phase 1: Card Insertion
        if "Card" in detected_objects and not checklist[0] and not active_p1:
            active_p1 = True
        if active_p1 and "Card" not in detected_objects:
            checklist[0] = True; active_p1 = False

        # Phase 2: PIN Entry
        elif checklist[0] and not checklist[1]:
            if hand_box and is_overlapping(global_keypad_box, hand_box):
                pin_frames += 1
                if pin_frames > 15: checklist[1] = True

        # Phase 3: Card Retrieval
        elif checklist[1] and not checklist[2]:
            if "Card" in detected_objects and not active_p3: active_p3 = True
            # Per your request: Complete P3 when Money is seen (Phase 4 start)
            if "Money" in detected_objects: checklist[2] = True 

        # Phase 4: Cash Withdrawal
        if "Money" in detected_objects:
            checklist[3] = True
            if checklist[1]: checklist[2] = True 

        # --- DASHBOARD RENDER ---
        canvas = np.zeros((h, new_width, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame 

        # Sidebar Header
        cv2.putText(canvas, "ATM TRANSACTION LOG", (w + 20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv2.line(canvas, (w + 20, 65), (w + SIDEBAR_WIDTH - 20, 65), (100, 100, 100), 1)
        
        # NOTE: I have REMOVED the Keypad ROI drawing here as requested.
        # The orange box will NOT appear on the final video.

        for i, name in enumerate(task_names):
            y_offset = 110 + (i * 60)
            
            # Task Box
            cv2.rectangle(canvas, (w + 20, y_offset - 25), (w + SIDEBAR_WIDTH - 20, y_offset + 25), (30, 30, 30), -1)
            cv2.putText(canvas, name, (w + 40, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            
            # Status Dot
            if checklist[i]:
                cv2.circle(canvas, (w + SIDEBAR_WIDTH - 50, y_offset + 5), 8, (0, 255, 0), -1) # Green
            else:
                cv2.circle(canvas, (w + SIDEBAR_WIDTH - 50, y_offset + 5), 8, (80, 80, 80), 2) # Grey

        video_writer.write(canvas)
        cv2.imshow("ATM Client Demo", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Client Demo Video Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_system()