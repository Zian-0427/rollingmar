import os
from PIL import Image
from tqdm import tqdm
import numpy as np

# --- Configuration ---
# TODO: Set your image size from args.img_size
# This is the 'args.img_size' you mentioned.
IMG_SIZE = 64 
# ---------------------

base_path = "/data/imagenet_50_private/"
source_dir = base_path + 'val'
target_dir = base_path + f'val_flat_{IMG_SIZE}'


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

print(f"Creating flattened, cropped directory at: {target_dir}")
print(f"All images will be center-cropped to {IMG_SIZE}x{IMG_SIZE}")

# Walk through the source directory
for root, _, files in os.walk(source_dir, followlinks=True):
    if root == source_dir:
        # Skip files directly in the 'val' root, if any
        continue
        
    # Get the class name (e.g., "n01729977")
    class_name = os.path.basename(root)
    
    print(f"\nProcessing class: {class_name}")
    
    # Use tqdm for a progress bar on the files
    for file in tqdm(files, desc=f"  Class {class_name}", leave=False):
        
        # Get the full path to the original file
        source_file_path = os.path.join(root, file)
        
        # Create the new "safe" filename
        new_filename = f"{class_name}_{file}"
        target_save_path = os.path.join(target_dir, new_filename)
        
        # Skip if this file has already been processed
        if os.path.exists(target_save_path):
            continue
            
        # --- This is the new block ---
        # Instead of symlinking, we load, crop, and save
        try:
            # 1. Open the image
            with Image.open(source_file_path) as img:
                # 2. Convert to RGB (handles palette/alpha issues)
                img_rgb = img.convert('RGB')
                
                # 3. Apply the center crop
                img_cropped = center_crop_arr(img_rgb, IMG_SIZE)
                
                # 4. Save the new, cropped image
                img_cropped.save(target_save_path)
                
        except Exception as e:
            # Print an error but continue with other files
            print(f"\nError processing {source_file_path}: {e}")
        # ---------------------------

print("\nDone.")