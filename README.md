# ATM Activity Recognition System

An ATM surveillance system that detects and tracks human actions in real time using YOLOv11, YOLOv4-tiny, and MediaPipe hand landmarks.

---

## Demo

<video src="atm_surveillance_whatsapp.mp4" controls width="100%"></video>

---

## How It Works

The system processes ATM footage and identifies a 4-step transaction sequence:

| Step | Action | Detection Method |
|------|--------|-----------------|
| 1 | Card Insertion | YOLOv11 detects card appearing then disappearing |
| 2 | PIN Entry | YOLOv4 hand overlaps with calibrated keypad ROI |
| 3 | Card Retrieval | YOLOv11 detects card reappearing |
| 4 | Cash Withdrawal | YOLOv11 detects money |

A sidebar dashboard shows a live transaction log with green/grey status dots for each step.

---

## Models

| File | Description |
|------|-------------|
| `atm.pt` | YOLOv11 — detects Card, Keypad, Money |
| `cross-hands-yolov4-tiny.cfg` / `.weights` | YOLOv4-tiny hand detector |
| `hand_landmarker.task` | MediaPipe hand landmark model |
| `atm_lstm_model_final.pth` | LSTM classifier for action sequences |

---

## Run with Docker (recommended)

**Requirements:** [Docker](https://docs.docker.com/get-docker/) installed.

The app opens an ROI calibration window so you can draw a box around the keypad before processing starts. Docker needs access to your screen for this — follow the one-time setup for your OS below.

### Linux

```bash
xhost +local:docker          # allow Docker to open windows on your display

git clone https://github.com/Khalid-Siddiqi/atm-activity.git
cd atm-activity
mkdir -p videos               # drop your input video here as videos/input.mp4

docker compose up --build
```

### Mac

Install [XQuartz](https://www.xquartz.org/), then:

```bash
open -a XQuartz               # start the X server
xhost +localhost              # allow connections

export DISPLAY=:0
git clone https://github.com/Khalid-Siddiqi/atm-activity.git
cd atm-activity
mkdir -p videos

docker compose up --build
```

### Windows

Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/). Launch it with **"Disable access control"** checked, then:

```cmd
set DISPLAY=<your-host-ip>:0.0

git clone https://github.com/Khalid-Siddiqi/atm-activity.git
cd atm-activity
mkdir videos

docker compose up --build
```

> To find your host IP on Windows run `ipconfig` and look for the Ethernet/Wi-Fi IPv4 address.

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Edit the VIDEO_SOURCE path in DEMO.py, then run
python DEMO.py
```

---

## Project Structure

```
atm-activity/
├── DEMO.py                        # Main demo — interactive GUI + video output
├── inference.py                   # Batch inference without GUI
├── train_lstm.py                  # Train the LSTM action classifier
├── feature-extractor.py           # Extract MediaPipe hand landmarks to .npy
├── atm.pt                         # Trained YOLOv11 model
├── atm_lstm_model_final.pth       # Trained LSTM model
├── hand_landmarker.task           # MediaPipe model
├── cross-hands-yolov4-tiny.cfg    # YOLOv4 config
├── cross-hands-yolov4-tiny.weights# YOLOv4 weights
├── requirements.txt
└── Dockerfile
```

---

## Training Your Own Model

```bash
# 1. Annotate ROI in videos
python draw_roi_video.py

# 2. Extract clips per action class
python extract_roi.py

# 3. Extract hand landmarks
python feature-extractor.py

# 4. Train LSTM
python train_lstm.py
```
