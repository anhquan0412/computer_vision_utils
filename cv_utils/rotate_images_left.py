#!/usr/bin/env python3

import os
from PIL import Image
from tqdm import tqdm
import sys

def rotate_images_to_portrait(source_folder, destination_folder=None):
    # Step 1: Find all images (end with jpg or png) in the given directory
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if destination_folder is None:
        destination_folder=source_folder
    
    # Step 2: Use tqdm to track progress
    for filename in tqdm(image_files, desc="Processing images"):
        source_path = os.path.join(source_folder, filename)
        
        # Open the image
        with Image.open(source_path) as img:
            width, height = img.size
            
            # Check if the image is in landscape mode
            if width > height:
                # Rotate the image by 90 degrees to the left
                img_rotated = img.rotate(90, expand=True)
                
                # Determine the save path
                destination_path = os.path.join(destination_folder, filename)
                
                # Save the rotated image
                img_rotated.save(destination_path)

# if __name__ == "__main__":
#     # Check if sufficient arguments are provided
#     if len(sys.argv) < 2 or len(sys.argv) > 3:
#         print("Usage: python rotate_images.py <source_folder_path> [destination_folder_path]")
#         sys.exit(1)
    
#     source_folder = sys.argv[1]
#     destination_folder = sys.argv[2] if len(sys.argv) == 3 else None
    
#     if not os.path.isdir(source_folder):
#         print(f"The provided source path '{source_folder}' is not a directory.")
#         sys.exit(1)
    
#     # If a destination folder is provided, check if it exists
#     if destination_folder and not os.path.isdir(destination_folder):
#         print(f"The provided destination path '{destination_folder}' does not exist. Creating it.")
#         os.makedirs(destination_folder, exist_ok=True)
    
#     rotate_images_to_portrait(source_folder, destination_folder)
    

# example usage:
# python rotate_images_left.py NewCameraSetup-May24
# python rotate_images_left.py NewCameraSetup-May24 NewCameraSetup-May24-Rotated 