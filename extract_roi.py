# import cv2
# import json
# import os
# import sys

# # --- CONFIGURATION ---
# JSON_PATH = "label.json"
# OUTPUT_ROOT = "dataset_extracted"  # Main output folder
# # ---------------------

# def extract_clips():
#     # 1. Load the JSON Label Data
#     if not os.path.exists(JSON_PATH):
#         print(f"❌ Error: '{JSON_PATH}' not found.")
#         return

#     with open(JSON_PATH, 'r') as f:
#         video_data = json.load(f)

#     print(f"📂 Found {len(video_data)} videos in JSON. Starting processing...")

#     # 2. Iterate through each video entry
#     for entry in video_data:
#         video_rel_path = entry['filename']  # e.g., "video/1.mp4"
        
#         # Handle Windows/Mac path differences automatically
#         video_path = os.path.normpath(video_rel_path)
        
#         if not os.path.exists(video_path):
#             print(f"⚠️ Skipping: '{video_path}' not found on disk.")
#             continue

#         # Open Video
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"⚠️ Error: Could not open '{video_path}'.")
#             continue

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         # Get ROI coordinates [x, y, w, h]
#         # Default to 0,0,0,0 if missing
#         bx, by, bw, bh = entry.get('roi_body', [0,0,0,0])
#         ax, ay, aw, ah = entry.get('roi_atm', [0,0,0,0])

#         # Validation: If ROI is empty, we can't crop
#         if bw <= 0 or bh <= 0 or aw <= 0 or ah <= 0:
#             print(f"⚠️ Skipping '{video_path}': Invalid ROI dimensions.")
#             cap.release()
#             continue

#         # Extract clean filename (e.g., "1")
#         base_name = os.path.splitext(os.path.basename(video_path))[0]
#         print(f"\n🎬 Processing Video: {base_name} (FPS: {fps:.2f})")

#         # 3. Process each segment (Action)
#         segments = entry.get('segments', [])
        
#         for seg in segments:
#             action = seg['action']
#             start_sec = seg['start']
#             end_sec = seg['end']

#             start_frame = int(start_sec * fps)
#             end_frame = int(end_sec * fps)

#             # Safety check for video length
#             if start_frame >= total_frames:
#                 print(f"   ⚠️ Skipping segment {action}: Start time exceeds video length.")
#                 continue

#             # --- PREPARE OUTPUT PATHS ---
#             # Structure: dataset/view_body/action_name/video_1_action.avi
            
#             # 1. Body View Folder
#             path_body = os.path.join(OUTPUT_ROOT, "view_body", action)
#             os.makedirs(path_body, exist_ok=True)
#             out_name_body = os.path.join(path_body, f"{base_name}_{action}.avi")

#             # 2. ATM View Folder
#             path_atm = os.path.join(OUTPUT_ROOT, "view_atm", action)
#             os.makedirs(path_atm, exist_ok=True)
#             out_name_atm = os.path.join(path_atm, f"{base_name}_{action}.avi")

#             # --- SETUP VIDEO WRITERS ---
#             # MJPG is safe for Windows. Size must be (WIDTH, HEIGHT) of the ROI
#             fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
#             out_b = cv2.VideoWriter(out_name_body, fourcc, fps, (bw, bh))
#             out_a = cv2.VideoWriter(out_name_atm, fourcc, fps, (aw, ah))

#             # --- EXTRACT FRAMES ---
#             cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
#             curr = start_frame
#             while curr <= end_frame:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # CROP BODY: frame[y:y+h, x:x+w]
#                 # We check bounds to ensure we don't crash if ROI is slightly off
#                 crop_b = frame[by : by+bh, bx : bx+bw]
#                 out_b.write(crop_b)

#                 # CROP ATM
#                 crop_a = frame[ay : ay+ah, ax : ax+aw]
#                 out_a.write(crop_a)

#                 curr += 1

#             # Release writers for this segment
#             out_b.release()
#             out_a.release()
#             print(f"   ✅ Extracted '{action}' ({start_sec}-{end_sec}s)")

#         cap.release()

#     print(f"\n🎉 Processing Complete! Files saved in: {os.path.abspath(OUTPUT_ROOT)}")

# if __name__ == "__main__":
#     extract_clips()



## extract 1 roi


import cv2
import json
import os
import sys

# --- CONFIGURATION ---
JSON_PATH = "video_label.json"
OUTPUT_ROOT = "roi_video_dataset"  # Main output folder
# ---------------------

def extract_clips():
    # 1. Load the JSON Label Data
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: '{JSON_PATH}' not found.")
        return

    with open(JSON_PATH, 'r') as f:
        video_data = json.load(f)

    print(f"📂 Found {len(video_data)} videos in JSON. Starting processing...")

    # 2. Iterate through each video entry
    for entry in video_data:
        video_rel_path = entry['filename']  # e.g., "video/1.mp4"
        
        # Handle Windows/Mac path differences automatically
        video_path = os.path.normpath(video_rel_path)
        
        if not os.path.exists(video_path):
            print(f"⚠️ Skipping: '{video_path}' not found on disk.")
            continue

        # Open Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Error: Could not open '{video_path}'.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get ROI coordinates [x, y, w, h]
        # Only looking for 'roi_body' now
        bx, by, bw, bh = entry.get('roi_body', [0,0,0,0])

        # Validation: If ROI is empty, we can't crop
        if bw <= 0 or bh <= 0:
            print(f"⚠️ Skipping '{video_path}': Invalid ROI dimensions (roi_body missing or zero).")
            cap.release()
            continue

        # Extract clean filename (e.g., "1")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n🎬 Processing Video: {base_name} (FPS: {fps:.2f})")

        # 3. Process each segment (Action)
        segments = entry.get('segments', [])
        
        for seg in segments:
            action = seg['action']
            start_sec = seg['start']
            end_sec = seg['end']

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            # Safety check for video length
            if start_frame >= total_frames:
                print(f"   ⚠️ Skipping segment {action}: Start time exceeds video length.")
                continue

            # --- PREPARE OUTPUT PATHS ---
            # Structure: dataset_extracted/action_name/video_1_action.avi
            
            # Create Action Folder
            path_action = os.path.join(OUTPUT_ROOT, action)
            os.makedirs(path_action, exist_ok=True)
            
            out_name = os.path.join(path_action, f"{base_name}_{action}.avi")

            # --- SETUP VIDEO WRITER ---
            # MJPG is safe for Windows. Size must be (WIDTH, HEIGHT) of the ROI
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            out_writer = cv2.VideoWriter(out_name, fourcc, fps, (bw, bh))

            # --- EXTRACT FRAMES ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            curr = start_frame
            while curr <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # CROP BODY: frame[y:y+h, x:x+w]
                crop_b = frame[by : by+bh, bx : bx+bw]
                out_writer.write(crop_b)

                curr += 1

            # Release writer for this segment
            out_writer.release()
            print(f"   ✅ Extracted '{action}' ({start_sec}-{end_sec}s)")

        cap.release()

    print(f"\n🎉 Processing Complete! Files saved in: {os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    extract_clips()