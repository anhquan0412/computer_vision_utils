import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial
import shutil
from tqdm import tqdm
import time
import json

def read_excel(path,sheet_int=0):
    return pd.read_excel(path,sheet_name=sheet_int)

def get_all_image_paths(directory):
    """
    Recursively gets all image file paths in a given directory.

    Args:
    directory (str): The directory to search for image files.

    Returns:
    list: A list of paths to image files found within the directory and its subdirectories.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}  # Use a set for faster membership testing
    image_paths = []

    def scan_directory(dir_path):
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(entry.path)
                elif entry.is_dir():
                    scan_directory(entry.path)

    scan_directory(directory)
    return image_paths

def read_json(path):
    path = Path(path)
    print(f'Reading JSON file from {path}')
    return json.loads(path.read_text())

def extract_images(absolute_paths,species,extracted_dir,extracted_folder = 'ExtractedSpecies',max_workers=2):
    
    def copy_file(file_path, destination_dir,keep_structure=True):
        file_path = Path(file_path)
        destination_dir = Path(destination_dir)
        if keep_structure:
            common_dir = file_path.relative_to(Path(*destination_dir.parts[:-2]))
            destination_dir = (destination_dir/common_dir).parent
            destination_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy(file_path, destination_dir)
    
    # For this function to work, there must be a relative path of each of "absolute path" to "extracted_dir"
    # In another word, "extracted_dir" must be within each of "absolute_path"
    assert len(absolute_paths)==len(species)
    
    extracted_dir = Path(extracted_dir)/(extracted_folder.strip())
    extracted_dir.mkdir(exist_ok=True)
    
    for spe in set(species):
        start_time = time.time()  # Start time
        print(f'Copying images for species: {spe}')
        species_dir = extracted_dir / spe.replace('/', '-')
        species_dir.mkdir(exist_ok=True)
        
        copy_to_dest_func = partial(copy_file, destination_dir=species_dir)
    
        start_time = time.time()  # Start time
        paths_filtered = [p for p,s in zip(absolute_paths,species) if s==spe]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(copy_to_dest_func, paths_filtered), total=len(paths_filtered)))
        
        end_time = time.time()  # End time
        print(f'===> Done copying {len(paths_filtered)} images. Time taken: {end_time - start_time:.2f} seconds.\n')

    print('Done copying all images.')