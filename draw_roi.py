import cv2
import json
import os

JSON_PATH = "label.json" # Your existing master list

def select_rois():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: {JSON_PATH} not found.")
        return

    # Load existing data
    with open(JSON_PATH, 'r') as f:
        video_list = json.load(f)

    updated_list = []

    print("ℹ️ INSTRUCTIONS:")
    print("1. First, draw the BODY ROI (Big box). Press SPACE/ENTER.")
    print("2. Next, draw the ATM ROI (Small box). Press SPACE/ENTER.")
    print("3. Press 'c' to cancel/skip a video.\n")

    for entry in video_list:
        video_path = entry['filename']
        
        # Check if already done (skip if you want to resume)
        if 'roi_body' in entry and 'roi_atm' in entry:
            print(f"⏩ Skipping {video_path} (ROIs already exist)")
            updated_list.append(entry)
            continue

        if not os.path.exists(video_path):
            print(f"⚠️ Video not found: {video_path}")
            updated_list.append(entry)
            continue

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret: continue

        print(f"🎨 Labeling: {video_path}")
        
        # 1. Select Body ROI
        print("   -> Draw BODY ROI...")
        r_body = cv2.selectROI("1. Draw BODY ROI", frame, fromCenter=False)
        cv2.destroyWindow("1. Draw BODY ROI")
        
        # 2. Select ATM ROI
        print("   -> Draw ATM ROI...")
        r_atm = cv2.selectROI("2. Draw ATM ROI", frame, fromCenter=False)
        cv2.destroyWindow("2. Draw ATM ROI")

        # Save to entry (x, y, w, h)
        entry['roi_body'] = list(r_body) # e.g., [100, 100, 300, 400]
        entry['roi_atm']  = list(r_atm)  # e.g., [150, 150, 50, 50]
        
        updated_list.append(entry)

    # Save back to JSON
    with open(JSON_PATH, 'w') as f:
        json.dump(updated_list, f, indent=4)
    
    print(f"\n✅ All ROIs saved to {JSON_PATH}!")

if __name__ == "__main__":
    select_rois()