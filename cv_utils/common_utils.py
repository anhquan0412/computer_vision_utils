import pandas as pd
from pathlib import Path
import os
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
    image_extensions = {'.jpg', '.jpeg', '.png'}
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

def value_counts_both(inps):
    if not isinstance(inps,pd.Series):
        inps = pd.Series(inps)
    return pd.DataFrame({
        'count': inps.value_counts(),
        'percentage': inps.value_counts(normalize=True)
        })


