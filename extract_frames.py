import cv2
import os

# --- CONFIGURATION ---
SOURCE_FOLDER = "roi_image_dataset/view_body"   # Where your clips are
OUTPUT_FOLDER = "image_data1" # Single folder for ALL images

# 1 = Save every frame (Best for video AI)
# 10 = Save every 10th frame (Best for YOLO/Object Detection to save time)
FRAME_SKIP = 10
# ---------------------

def flatten_frames():
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ Error: Source folder '{SOURCE_FOLDER}' not found.")
        return

    # Create the single output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"📂 Scanning '{SOURCE_FOLDER}'...")
    print(f"📂 Output will be in '{OUTPUT_FOLDER}'")
    
    video_count = 0
    total_images = 0

    # Walk through all subfolders
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_count += 1
                video_path = os.path.join(root, file)
                
                # Get action name from parent folder name (e.g., "a_card_in")
                # This helps you know what action it is just by looking at the file
                action_name = os.path.basename(root)
                base_name = os.path.splitext(file)[0]
                
                print(f"   🎬 Processing: {file} ...")
                
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0
                saved_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    if frame_idx % FRAME_SKIP == 0:
                        # Unique Filename: video1_a_card_in_frame_001.jpg
                        # This prevents overwriting and keeps things organized
                        img_name = f"{base_name}_frame1_{saved_count:04d}.jpg"
                        out_path = os.path.join(OUTPUT_FOLDER, img_name)
                        
                        cv2.imwrite(out_path, frame)
                        saved_count += 1
                        total_images += 1
                    
                    frame_idx += 1

                cap.release()

    print(f"\n🎉 Done! Processed {video_count} videos.")
    print(f"🖼️  Saved {total_images} images to: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    flatten_frames()