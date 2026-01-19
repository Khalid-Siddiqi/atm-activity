import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\Yousuf Traders\Desktop\Projects\atm-activity\image_data\extra_data"
OUTPUT_FOLDER = r"C:\Users\Yousuf Traders\Desktop\Projects\atm-activity\image_data\augmented_dataset"

# How many new variations to create per image?
# 40 images * 10 = 400 images
AUGMENT_FACTOR = 10 
# ---------------------

def add_noise(image):
    """Adds grainy noise to simulate bad cameras"""
    row, col, ch = image.shape
    mean = 0
    var = 50
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.astype('float32') 
    # Clip values to stay between 0-255
    return np.clip(noisy, 0, 255).astype('uint8')

def adjust_brightness(image):
    """Randomly makes image darker or brighter"""
    value = np.random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Add offset to the Value (Brightness) channel
    # Using cv2.add ensures we don't wrap around (255+1 becomes 255, not 0)
    if value > 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, abs(value))
        
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def rotate_image(image):
    """Rotates image slightly (-10 to +10 degrees)"""
    angle = np.random.uniform(-10, 10)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def zoom_image(image):
    """Zooms in slightly"""
    zoom_factor = np.random.uniform(1.0, 1.3)
    h, w = image.shape[:2]
    
    # Crop center
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    cropped = image[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))

# --- MAIN EXECUTION ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Find all images (jpg, png, jpeg)
image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.*"))
print(f"📂 Found {len(image_files)} original images.")

count = 0

for img_path in image_files:
    # Read Image
    original = cv2.imread(img_path)
    if original is None: continue
    
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. Save Original First
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename}_orig.jpg"), original)
    
    # 2. Generate Variations
    for i in range(AUGMENT_FACTOR):
        aug_img = original.copy()
        
        # Randomly apply effects
        if np.random.rand() > 0.5: aug_img = adjust_brightness(aug_img)
        if np.random.rand() > 0.5: aug_img = add_noise(aug_img)
        if np.random.rand() > 0.5: aug_img = rotate_image(aug_img)
        if np.random.rand() > 0.5: aug_img = zoom_image(aug_img)
        
        # Save
        new_name = f"{filename}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, new_name), aug_img)
        count += 1

print(f"\n🎉 Success! Generated {count} images in '{OUTPUT_FOLDER}'")