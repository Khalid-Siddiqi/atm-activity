import cv2
import json
import os

JSON_PATH = "video_label.json" 

def select_single_roi():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    # Load existing data
    with open(JSON_PATH, 'r') as f:
        video_list = json.load(f)

    updated_list = []

    print("ℹ️ INSTRUCTIONS:")
    print("1. Draw the WHOLE BODY ROI (Include person + ATM interface).")
    print("2. Press SPACE or ENTER to confirm.")
    print("3. Press 'c' to cancel/skip a video.\n")

    for entry in video_list:
        video_path = entry['filename']
        
        # Check if already done (skip if 'roi_body' already exists)
        if 'roi_body' in entry:
            print(f"⏩ Skipping {video_path} (ROI already exists)")
            updated_list.append(entry)
            continue

        if not os.path.exists(video_path):
            print(f"⚠️ Video not found: {video_path}")
            updated_list.append(entry)
            continue

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret: 
            updated_list.append(entry)
            continue

        print(f"🎨 Labeling: {video_path}")
        
        # --- SELECT SINGLE ROI ---
        # fromCenter=False means you drag from top-left to bottom-right
        r_body = cv2.selectROI("Draw WHOLE BODY ROI", frame, fromCenter=False)
        cv2.destroyWindow("Draw WHOLE BODY ROI")

        # Save to entry (x, y, w, h)
        # We only save 'roi_body' now.
        entry['roi_body'] = list(r_body) 
        
        updated_list.append(entry)

    # Save back to JSON
    with open(JSON_PATH, 'w') as f:
        json.dump(updated_list, f, indent=4)
    
    print(f"\n✅ All ROIs saved to {JSON_PATH}!")

if __name__ == "__main__":
    select_single_roi()