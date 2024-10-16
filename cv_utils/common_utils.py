import pandas as pd
from pathlib import Path
import os
import json
import numpy as np
import math

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


def truncate_float_array(xs, precision=3):
    """
    Vectorized version of truncate_float(...), truncates the fractional portion of each
    floating-point value to a specific number of floating-point digits.

    Credit to Dan Morris's MegaDetector ct_utils.py
    Args:
        xs (list): list of floats to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1            
            
    Returns:
        list: list of truncated floats
    """

    return [truncate_float(x, precision=precision) for x in xs]


def truncate_float(x, precision=3):
    """
    Truncates the fractional portion of a floating-point value to a specific number of 
    floating-point digits.
    
    For example: 
        
        truncate_float(0.0003214884) --> 0.000321
        truncate_float(1.0003214884) --> 1.000321
    
    This function is primarily used to achieve a certain float representation
    before exporting to JSON.

    Args:
        x (float): scalar to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1
        
    Returns:
        float: truncated version of [x]
    """

    assert precision > 0

    if np.isclose(x, 0):
        
        return 0
    
    elif (x > 1):
        
        fractional_component = x - 1.0
        return 1 + truncate_float(fractional_component)
    
    else:
        
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit.
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        
        # Shift decimal point by multiplication with factor, flooring, and
        # division by factor.
        return math.floor(x * factor)/factor
