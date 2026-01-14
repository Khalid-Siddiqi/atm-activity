import cv2
import json
import os

# --- CONFIGURATION ---
JSON_PATH = "labels.json"
OUTPUT_FOLDER = "dataset_splits"
# ---------------------

def split_video_by_seconds():
    # 1. Load JSON
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    video_path = data['video_filename']
    segments = data['segments']

    # 2. Open Video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    # Get FPS to calculate frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"🎬 Video Loaded: {video_path}")
    print(f"   FPS: {fps:.2f} (Using this to calculate start/end frames)")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 3. Process Segments
    for seg in segments:
        action_name = seg['action']
        
        # --- THE MATH (Seconds -> Frames) ---
        start_sec = seg['start_sec']
        end_sec = seg['end_sec']
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        # ------------------------------------

        out_name = f"{OUTPUT_FOLDER}/{action_name}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

        print(f"   ✂️ Cutting '{action_name}' ({start_sec}s - {end_sec}s) -> Frames {start_frame}-{end_frame}")

        # Jump to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"      ⚠️ Video ended early at {current_frame}!")
                break
            
            out.write(frame)
            current_frame += 1

        out.release()
        print(f"      ✅ Saved to {out_name}")

    cap.release()
    print("\n🎉 Done! Check 'dataset_splits' folder.")

if __name__ == "__main__":
    split_video_by_seconds()