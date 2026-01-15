import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION (UPDATED) ---
INPUT_FOLDER = r"C:\Users\gutech\Desktop\atm-activity\npy_data"
MODEL_SAVE_PATH = "atm_lstm_model_fixed.pth"

# 1. OPTIMIZATION: Reduce Sequence Length
# 3 seconds * 30 fps = 90 frames (Focus on the action, cut the empty space)
SEQUENCE_LENGTH = 90  

# 2. OPTIMIZATION: Simplify Model
INPUT_SIZE = 63       
HIDDEN_SIZE = 64      # Reduced from 128 to prevent overfitting
NUM_LAYERS = 1        # Reduced from 2 to 1 (Simpler is better for small data)
NUM_CLASSES = 4
BATCH_SIZE = 8        # Smaller batch size for small dataset
EPOCHS = 100          # More epochs since we have more data now
LEARNING_RATE = 0.001

CLASS_NAMES = ["a_card_in", "b_keypad", "a_ret_card", "d_ret_cash"]

# --- Data Augmentation Functions ---
def augment_data(X_data, y_data):
    """ artificially increases dataset size by adding noise and scaling """
    augmented_X = []
    augmented_y = []

    print(f"   ✨ Augmenting data (Original size: {len(X_data)})...")

    for i in range(len(X_data)):
        sequence = X_data[i]
        label = y_data[i]
        
        # 1. Original
        augmented_X.append(sequence)
        augmented_y.append(label)

        # 2. Add Noise (Jitter)
        noise = np.random.normal(0, 0.02, sequence.shape)
        augmented_X.append(sequence + noise)
        augmented_y.append(label)

        # 3. Scaling (Simulate closer/further hand)
        scaler = np.random.uniform(0.9, 1.1)
        augmented_X.append(sequence * scaler)
        augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)

# --- Dataset Loader ---
class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_and_prep_data():
    print("📥 Loading and Processing ATM datasets...")
    all_X = []
    all_y = []

    for i, class_name in enumerate(CLASS_NAMES):
        file_path = os.path.join(INPUT_FOLDER, f"{class_name}.npy")
        if not os.path.exists(file_path): continue
            
        # Load raw (original 600 length)
        data = np.load(file_path)
        
        # TRIM sequence to new shorter length (90)
        # We take the MIDDLE 90 frames (where action usually happens)
        # If video is shorter, we pad.
        trimmed_data = []
        for seq in data:
            if len(seq) > SEQUENCE_LENGTH:
                # Take center crop of the action
                start = (len(seq) - SEQUENCE_LENGTH) // 2
                trimmed_data.append(seq[start : start + SEQUENCE_LENGTH])
            else:
                # Pad if too short
                padding = np.zeros((SEQUENCE_LENGTH - len(seq), 63))
                trimmed_data.append(np.vstack((seq, padding)))
        
        data = np.array(trimmed_data)
        labels = np.full(len(data), i)
        
        all_X.append(data)
        all_y.append(labels)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Apply Augmentation
    X, y = augment_data(X, y)
    
    print(f"📦 Final Augmented Dataset Shape: {X.shape}")
    return X, y

# --- Model ---
class ATMSurveillanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ATMSurveillanceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0) # No dropout for 1 layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Training ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Device: {device}")
    
    X, y = load_and_prep_data()
    
    # Stratify ensure we get examples of ALL classes in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(HandSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HandSequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = ATMSurveillanceLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🔥 Starting Training (Fixed)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        # Evaluation
        if (epoch+1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_val.size(0)
                    correct += (predicted == y_val).sum().item()
            
            train_acc = 100 * correct_train / total_train
            test_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Fixed Model saved to: {MODEL_SAVE_PATH}")