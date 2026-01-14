import cv2
import json
import os

# --- CONFIGURATION ---
JSON_PATH = "labels.json"
OUTPUT_ROOT = "dataset_final"
# ---------------------

def process_batch():
    # 1. Load the Master List
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        video_list = json.load(f)

    print(f"📂 Found {len(video_list)} videos in the batch list.")

    # 2. Loop through every video in the JSON
    for video_entry in video_list:
        video_path = video_entry['filename']
        segments = video_entry['segments']

        # Check if file exists
        if not os.path.exists(video_path):
            print(f"   ❌ Skipping: '{video_path}' not found.")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Extract clean name (e.g., "atm_01") for naming files
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n🎬 Processing: {base_name} (FPS: {fps:.2f})")

        # 3. Cut the segments for this specific video
        for seg in segments:
            action_name = seg['action']
            start_sec = seg['start']
            end_sec = seg['end']

            # Calculate frames
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            # Create folder: dataset_final/a_card_in/
            action_folder = os.path.join(OUTPUT_ROOT, action_name)
            os.makedirs(action_folder, exist_ok=True)

            # Output Name: dataset_final/a_card_in/atm_01_a_card_in.avi
            out_name = f"{action_folder}/{base_name}_{action_name}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            curr = start_frame
            while curr <= end_frame:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
                curr += 1
            
            out.release()
            print(f"   mapped -> {action_name} ({start_sec}s-{end_sec}s)")

        cap.release()

    print(f"\n✅ All 11 videos processed! Check '{OUTPUT_ROOT}' folder.")

if __name__ == "__main__":
    process_batch()