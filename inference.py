import cv2
import mediapipe as mp
# Fix for solutions error
try:
    from mediapipe import solutions
except ImportError:
    import mediapipe.python.solutions as solutions

import numpy as np
import torch
import torch.nn as nn
import collections

# --- CONFIGURATION ---
MODEL_PATH = "atm_lstm_model_final.pth"
VIDEO_PATH = r"C:\Users\gutech\Desktop\atm-activity\video\1.mp4" 
OUTPUT_PATH = "atm_output.mp4"  # <--- File to save

SEQUENCE_LENGTH = 30
INPUT_SIZE = 63
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASSES = 4

CLASS_NAMES = ["Insert Card", "Use Screen", "Take Card", "Take Cash"]

# --- Model Class ---
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
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 1. SETUP GPU ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("⚠️ GPU NOT FOUND. Running on CPU.")

# --- 2. LOAD MODEL ---
print("🚀 Loading model...")
model = ATMSurveillanceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 3. SETUP MEDIAPIPE ---
mp_hands = solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4) 
mp_draw = solutions.drawing_utils

sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(VIDEO_PATH)

# --- VIDEO WRITER SETUP ---
# Get video properties to match output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"🎬 Saving video to: {OUTPUT_PATH} ({frame_width}x{frame_height} @ {fps}fps)")

# 'mp4v' is the standard codec for .mp4 files (works in VLC)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

print("🎬 Starting Analysis...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    frame_features = []
    
    # Check for 'multi_hand_world_landmarks'
    if results.multi_hand_world_landmarks:
        
        # 1. Get 3D WORLD Landmarks (Meters) - FOR MODEL
        world_landmarks = results.multi_hand_world_landmarks[0]
        for lm in world_landmarks.landmark:
            frame_features.extend([lm.x, lm.y, lm.z])
            
        # 2. Get 2D SCREEN Landmarks - FOR DRAWING ONLY
        if results.multi_hand_landmarks:
            screen_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, screen_landmarks, mp_hands.HAND_CONNECTIONS)

        # Add to buffer
        sequence_buffer.append(frame_features)
    else:
        # Hand Lost: Clear Buffer
        if len(sequence_buffer) > 0:
            sequence_buffer.clear()

    # --- PREDICTION ---
    prediction_text = "Waiting for Hand..."
    color = (0, 165, 255) # Orange

    if len(sequence_buffer) == SEQUENCE_LENGTH:
        input_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Debug Print
            probs_list = probabilities[0].cpu().numpy()
            print(f"Scores: {probs_list} -> Best: {CLASS_NAMES[predicted_idx.item()]}")

            # Threshold Check
            if confidence.item() > 0.5: 
                action = CLASS_NAMES[predicted_idx.item()]
                prediction_text = f"{action} ({confidence.item()*100:.0f}%)"
                color = (0, 255, 0) # Green
            else:
                prediction_text = "Uncertain"

    # --- VISUALIZATION ---
    cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
    cv2.putText(frame, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Save the frame to video file
    out.write(frame)

    # Show live (optional, you can comment this out to run faster)
    cv2.imshow('ATM AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Clean up
cap.release()
out.release() # Stop saving
cv2.destroyAllWindows()
print("✅ Video saved successfully!")