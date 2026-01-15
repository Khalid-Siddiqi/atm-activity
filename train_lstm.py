import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\gutech\Desktop\atm-activity\npy_data"
MODEL_SAVE_PATH = "atm_lstm_model_final.pth"

# Model Hyperparameters
SEQUENCE_LENGTH = 30   # 1 Second (30 frames)
STRIDE = 5             # Slide 5 frames at a time (Heavy overlap = More data)
INPUT_SIZE = 63        # 21 landmarks * 3 coords
HIDDEN_SIZE = 64       
NUM_LAYERS = 1         
NUM_CLASSES = 4        
BATCH_SIZE = 16        
EPOCHS = 100           
LEARNING_RATE = 0.001

CLASS_NAMES = ["a_card_in", "b_keypad", "c_ret_card", "d_ret_cash"]

# --- Helper: Data Augmentation ---
def augment_data(X_data, y_data):
    augmented_X = []
    augmented_y = []
    
    print(f"   ✨ Augmenting {len(X_data)} samples...")
    for i in range(len(X_data)):
        seq, label = X_data[i], y_data[i]
        augmented_X.append(seq)
        augmented_y.append(label)
        
        # Jitter (Noise)
        noise = np.random.normal(0, 0.01, seq.shape)
        augmented_X.append(seq + noise)
        augmented_y.append(label)
        
    return np.array(augmented_X), np.array(augmented_y)

# --- Helper: Load Data ---
def load_and_prep_data():
    print(f"📥 Looking for data in: {INPUT_FOLDER}")
    
    all_X = []
    all_y = []

    for i, class_name in enumerate(CLASS_NAMES):
        file_path = os.path.join(INPUT_FOLDER, f"{class_name}.npy")
        
        # DEBUG: Check if file exists
        if not os.path.exists(file_path):
            print(f"   ❌ MISSING: Could not find {file_path}")
            continue
            
        data = np.load(file_path)
        chunked_data = []
        
        # Sliding Window Logic
        for seq in data:
            # If video is shorter than 30 frames, pad it
            if len(seq) < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - len(seq), 63))
                chunked_data.append(np.vstack((seq, padding)))
                continue

            # Standard Sliding Window
            # Range logic fixed: Use len(seq) - sequence_length + 1 to include the end
            num_windows = 0
            for start in range(0, len(seq) - SEQUENCE_LENGTH + 1, STRIDE):
                end = start + SEQUENCE_LENGTH
                chunk = seq[start:end]
                chunked_data.append(chunk)
                num_windows += 1

        if len(chunked_data) > 0:
            class_X = np.array(chunked_data)
            class_y = np.full(len(class_X), i)
            all_X.append(class_X)
            all_y.append(class_y)
            print(f"   🔹 Loaded '{class_name}': {len(data)} videos -> {len(class_X)} sequences")
        else:
            print(f"   ⚠️ Loaded '{class_name}' but generated 0 sequences (Check video lengths!)")

    # CRITICAL CHECK: Did we get any data?
    if len(all_X) == 0:
        print("\n❌ CRITICAL ERROR: No data loaded. Please check:")
        print("   1. Is the folder path correct?")
        print("   2. Did you run the feature extractor script first?")
        exit()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Augment
    X, y = augment_data(X, y)
    
    print(f"📦 Final Dataset Shape: {X.shape}")
    return X, y

# --- Dataset Class ---
class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Model ---
# --- Model ---
class ATMSurveillanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ATMSurveillanceLSTM, self).__init__()
        
        # --- MISSING LINES ADDED HERE ---
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # --------------------------------

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Now self.hidden_size and self.num_layers will work!
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Main ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Device: {device}")
    
    X, y = load_and_prep_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(HandSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HandSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = ATMSurveillanceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🔥 Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        if (epoch+1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += y_val.size(0)
                    test_correct += (predicted == y_val).sum().item()
            
            train_acc = 100 * correct / total
            test_acc = 100 * test_correct / test_total
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")