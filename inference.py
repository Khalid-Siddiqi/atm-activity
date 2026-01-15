import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import collections

# --- CONFIGURATION ---
MODEL_PATH = "atm_lstm_model_final.pth"
VIDEO_PATH = r"C:\Users\gutech\Desktop\atm-activity\video\7.mp4" 
# VIDEO_PATH = 0  # Uncomment to use Webcam

# Must match training exactly
SEQUENCE_LENGTH = 30
INPUT_SIZE = 63
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_CLASSES = 4

CLASS_NAMES = ["Insert Card", "Type PIN", "Take Card", "Take Cash"]

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

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Loading model on {device}...")

model = ATMSurveillanceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Buffer to store the last 30 frames
sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. MediaPipe Extraction
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    frame_features = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for lm in hand_landmarks.landmark:
            frame_features.extend([lm.x, lm.y, lm.z])
    else:
        # If no hand, skip or pad (we skip to keep buffer clean)
        frame_features = [0] * 63

    # 2. Prediction Logic
    prediction_text = "Analyzing..."
    color = (0, 0, 255) # Red

    if len(frame_features) > 1:
        sequence_buffer.append(frame_features)

    if len(sequence_buffer) == SEQUENCE_LENGTH:
        input_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Confidence Threshold (e.g. 70%)
            if confidence.item() > 0.7:
                action = CLASS_NAMES[predicted_idx.item()]
                prediction_text = f"{action} ({confidence.item()*100:.0f}%)"
                color = (0, 255, 0) # Green
            else:
                prediction_text = "Uncertain"

    # 3. Visualization
    cv2.rectangle(frame, (0,0), (640, 50), (0,0,0), -1)
    cv2.putText(frame, prediction_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('ATM AI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()