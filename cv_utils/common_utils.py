import pandas as pd
from pathlib import Path
import os
import json
from multiprocessing import Pool, cpu_count

def dataframe_apply_parallel(dfGrouped, func,n_workers=None):
    # https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    if n_workers is None: n_workers=min(16,cpu_count())
    with Pool(n_workers) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return ret_list

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


def check_and_fix_http_path(path):
    if 'https:/' in path:
        h_str = 'https:/'
    elif 'http:/' in path:
        h_str = 'http:/'
    else:
        return path
    path = path.split(h_str)[1]
    return f"{h_str}/{path}"