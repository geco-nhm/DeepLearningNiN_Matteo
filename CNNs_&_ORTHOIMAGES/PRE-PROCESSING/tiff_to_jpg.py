
"""

TO APPLY TO THE FOLDER CONTAINING ALREADY THE TRAIN/VAL SPLITS
"""


import os
import tifffile
from PIL import Image
import numpy as np

def convert_tif_to_jpg(tif_path, jpg_path):
    try:
        # Load the TIFF image
        tiff_image = tifffile.imread(tif_path)
        
        # Normalize and convert the TIFF image to uint8 if necessary
        if tiff_image.dtype != np.uint8:
            tiff_image = (255 * (tiff_image - tiff_image.min()) / (tiff_image.ptp() + 1e-8)).astype(np.uint8)
        
        # Convert the normalized image to a PIL Image
        pil_image = Image.fromarray(tiff_image)
        
        # Save the PIL Image as a JPG file
        pil_image.save(jpg_path, format='JPEG')
        
        print(f"Converted {tif_path} to {jpg_path}")
    except Exception as e:
        print(f"Failed to convert {tif_path}: {e}")

def convert_all_tif_in_directory(src_directory, dst_directory):
    for root, _, files in os.walk(src_directory):
        for filename in files:
            if filename.endswith('.tif'):
                src_path = os.path.join(root, filename)
                # Create the corresponding path in the destination directory
                relative_path = os.path.relpath(root, src_directory)
                dst_dir = os.path.join(dst_directory, relative_path)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, filename.replace('.tif', '.jpg'))
                convert_tif_to_jpg(src_path, dst_path)

root_path = '/Users/matteo_crespinjouan/Desktop/NHM/ORTHO_ENDEAVOUR/32_split'
train_path = os.path.join(root_path, 'train')
val_path = os.path.join(root_path, 'val')

train_jpg_path = os.path.join(root_path, 'train_jpg')
val_jpg_path = os.path.join(root_path, 'val_jpg')

convert_all_tif_in_directory(train_path, train_jpg_path)
convert_all_tif_in_directory(val_path, val_jpg_path)