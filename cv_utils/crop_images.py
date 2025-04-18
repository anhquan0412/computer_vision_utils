import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
from pandas.api.types import is_integer_dtype,is_float_dtype
from tqdm import tqdm
from PIL import Image, ImageOps
import warnings
from .common_utils import read_json
from .awc_utils import mdv5_json_to_df

def crop_and_save_image(img_path,img_dir,cropped_dir,bbox_coord,square_crop,bbox_rank,postfix,force,return_relative_path=True,error_log=[]):  
    """
    Crop images based on bounding box coordinates and save the cropped images.

    img_dir (str): The absolute path to the directory containing the raw images.

    img_path (str): The relative path to the image file.

    cropped_dir (str): The absolute path to the directory where the cropped images will be saved.

    bbox_coord (tuple): The normalized bounding box coordinates

    square_crop (bool): Whether to add black padding to make the cropped image a square. Default is True.

    bbox_rank (int): The ranking (order) of the bbox to the image. For image with multiple bboxes, this is used to distinguish the result

    postfix (str): Postfix to be added to the cropped image's file name. Default is an empty string.

    return_relative_path (bool): Whether to return the relative path of the cropped image. Default is False.

    force (bool): Whether to overwrite existing cropped images. Default is False.

    Returns:
    The path to the cropped image
    """

    img_path = Path(img_path)
    img_dir = Path(img_dir)
    cropped_dir = Path(cropped_dir)


    dest_absolute_path = cropped_dir/img_path
    dest_absolute_path.parent.mkdir(exist_ok=True,parents=True)
    if postfix.strip()!="": postfix=f"_{postfix}"
    dest_fname = f"{dest_absolute_path.stem}___crop{bbox_rank:>02d}{postfix}{dest_absolute_path.suffix}"
    dest_absolute_path = dest_absolute_path.parent / dest_fname

    # skip cropping if the cropped images already exists
    if dest_absolute_path.exists() and not force:
        if not return_relative_path:
            return dest_absolute_path.as_posix()
        return (img_path.parent/dest_fname).as_posix()

    try:
        # load local image
        with Image.open(img_dir/img_path) as img:
            img.load()
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')

        # start cropping
        img_w, img_h = img.size
        xmin = int(bbox_coord[0] * img_w)
        ymin = int(bbox_coord[1] * img_h)
        box_w = int(bbox_coord[2] * img_w)
        box_h = int(bbox_coord[3] * img_h)
        if square_crop:
            box_size = max(box_w, box_h)
            xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
            ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
            box_w = min(img_w, box_size)
            box_h = min(img_h, box_size)
    
        if box_w == 0 or box_h == 0:
            error_log.append([img_path.as_posix(), bbox_coord, 'Box coordinate error'])
            return None
    
        crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
        if square_crop and (box_w != box_h):
            crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
        crop.save(dest_absolute_path)

        if not return_relative_path:
            return dest_absolute_path.as_posix()
        return (img_path.parent/dest_fname).as_posix()
    except Exception as e:
        exception_type = type(e).__name__
        error_log.append([img_path.as_posix(), bbox_coord, f'{exception_type}: {e}'])
        return None

def crop_images_from_df(df,img_dir,cropped_dir,square_crop=True,postfix="",crop_cat=['1'],max_workers=1,logdir='.',detection_csv='detection.csv',force=False):
    """
    Crop images based on bounding box coordinates provided in a DataFrame and save the cropped images.
    Also save the new csv file with the cropped image paths, and log any error that occurs during the process.

    Parameters:
    df (pd.DataFrame): The DataFrame containing image file paths and bounding box coordinates.
        The DataFrame must have 3 columns: 'file', 'detection_bbox', 'bbox_rank'.
        The DataFrame can have 'detection_category' column if the images are to be filtered by category.

    img_dir (str): The absolute path to the directory containing the raw images.

    cropped_dir (str): The absolute path to the directory where the cropped images will be saved.

    square_crop (bool): Whether to add black padding to make the cropped image a square. Default is True.

    postfix (str): Postfix to be added to the cropped image's file name. Default is an empty string.

    crop_cat (list): A list of categories to be cropped. Contain STRING. Default is '1' for animal category.

    max_workers (int): Number of workers for parallelization. Default is 1.

    logdir (str): Relative path to the log directory. Default is the current directory.

    force (bool): Whether to overwrite existing cropped images. Default is False.

    Returns:
    None
    """
    error_log = []
    
    def wrapper(args):
        return crop_and_save_image(*args,error_log=error_log)

    org_columns=df.columns.tolist()
    if 'cropped_file' in org_columns:
        org_columns.remove('cropped_file')

    assert "file" in org_columns, "There must be a column called 'file' containing relative paths of images"
    assert "detection_bbox" in org_columns, "There must be a column called 'detection_bbox' containing the normalized bbox coordinates as lists"
    assert "bbox_rank" in org_columns, "There must be a column called 'bbox_rank' containing the ranking of the bbox to the image.\n For image with multiple bboxes, this is used to distinguish the result"
    
    df_no_bbox = df[df.detection_bbox.isna()].copy()
    df_no_bbox['cropped_file'] = None
    
    df = df.dropna(subset=['detection_bbox']).copy().reset_index(drop=True)
    if df.detection_bbox.dtype == "object" and df.shape[0]>0:
        if not isinstance(df.detection_bbox.values[0],(tuple,list)):
            df.detection_bbox = df.detection_bbox.apply(lambda x: tuple(ast.literal_eval(x)))
    df.bbox_rank = df.bbox_rank.astype(int)
    

    df_no_crop = pd.DataFrame(columns=org_columns)
    if 'detection_category' in org_columns:
        # if detection_category is an int or float, convert to int, then to str
        if is_integer_dtype(df.detection_category) or is_float_dtype(df.detection_category):
            df.detection_category = df.detection_category.astype(int)
        df.detection_category = df.detection_category.astype(str)
        df_no_crop = df[~df.detection_category.isin(crop_cat)].copy().reset_index(drop=True)
        df_no_crop['cropped_file'] = None
        df = df[df.detection_category.isin(crop_cat)].copy().reset_index(drop=True)
    else:
        warnings.warn("No 'detection_category' column found. All detections will be cropped.")

    Path(cropped_dir).mkdir(exist_ok=True,parents=True)

    df['_img_dir'] = str(img_dir)
    df['_cropped_dir'] = str(cropped_dir)
    df['_square_crop'] = square_crop
    df['_postfix'] = postfix
    df['_force'] = force
    args_list = df[['file','_img_dir','_cropped_dir','detection_bbox','_square_crop','bbox_rank','_postfix','_force']].values.tolist()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        cropped_paths = list(tqdm(executor.map(wrapper, args_list),total=len(args_list),desc="Cropping images"))
        
    assert len(cropped_paths)==df.shape[0]
    df['cropped_file'] = cropped_paths
    df = df[org_columns+['cropped_file']].copy()
    df = pd.concat([df,df_no_crop,df_no_bbox],ignore_index=True)
    df.to_csv(str(Path(detection_csv)),index=False)

    # Write the error log to a CSV file
    if len(error_log):
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True,parents=True)
        error_log_path = Path(logdir) / f"cropping_errors_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(error_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["img_path", "detection_bbox", "error"])
            writer.writerows(error_log)


def crop_images_from_csv(detection_csv,img_dir,cropped_dir,square_crop=True,postfix="",crop_cat=['1'],max_workers=1,logdir='.',force=False):
    """
    Crop images based on bounding box coordinates provided in a CSV file and save the cropped images.
    Also save the new csv file with the cropped image paths, and log any error that occurs during the process.

    Parameters:
    detection_csv (str): The absolute path to the CSV file containing image file paths and bounding box coordinates.
                         The CSV file must have 3 columns: 'file', 'detection_bbox', 'bbox_rank'.

    img_dir (str): The absolute path to the directory containing the raw images.

    cropped_dir (str): The absolute path to the directory where the cropped images will be saved.

    square_crop (bool): Whether to add black padding to make the cropped image a square. Default is True.

    postfix (str): Postfix to be added to the cropped image's file name. Default is an empty string.

    crop_cat (list): A list of categories to be cropped. Contain STRING. Default is '1' for animal category.

    max_workers (int): Number of workers for parallelization. Default is 1.

    logdir (str): Relative path to the log directory. Default is the current directory.

    force (bool): Whether to overwrite existing cropped images. Default is False.

    Returns:
    None
    """

    df = pd.read_csv(Path(detection_csv))
    crop_images_from_df(df,img_dir,cropped_dir,square_crop,postfix,crop_cat,max_workers,logdir,detection_csv,force=force)
    

def crop_images_from_md_json(md_json,img_dir,cropped_dir,square_crop=True,postfix="",crop_cat=['1'],max_workers=1,logdir='.',force=False):
    """
    Crop images based on bounding box coordinates provided in a JSON file and save the cropped images.
    Also save the new csv file with the cropped image paths in the same location as the json file, 
    and log any error that occurs during the process.

    Parameters:
    md_json (str): The absolute path to the JSON file containing image file paths and bounding box coordinates.

    img_dir (str): The absolute path to the directory containing the raw images.

    cropped_dir (str): The absolute path to the directory where the cropped images will be saved.

    square_crop (bool): Whether to add black padding to make the cropped image a square. Default is True.

    postfix (str): Postfix to be added to the cropped image's file name. Default is an empty string.

    crop_cat (list): A list of categories to be cropped. Contain STRING. Default is '1' for animal category.

    max_workers (int): Number of workers for parallelization. Default is 1.

    logdir (str): Relative path to the log directory. Default is the current directory.

    force (bool): Whether to overwrite existing cropped images. Default is False.

    Returns:
    None
    """

    json_file = read_json(md_json)
    df = mdv5_json_to_df(json_file)
    csv_file = Path(md_json).with_suffix('.csv')
    df.to_csv(csv_file,index=False)
    crop_images_from_df(df,img_dir,cropped_dir,square_crop,postfix,crop_cat,max_workers,logdir,csv_file,force=force)

    

def main():
    parser = argparse.ArgumentParser(description="Crop images based on bounding box coordinates provided in a CSV file.")
    parser.add_argument('detection_file', type=str, help="The absolute path to the file containing image file paths and bounding box coordinates. Can be a csv or a JSON file")
    parser.add_argument('img_dir', type=str, help="The absolute path to the directory containing the raw images.")
    parser.add_argument('cropped_dir', type=str, help="The absolute path to the directory where the cropped images will be saved.")
    parser.add_argument('--square_crop', type=bool, default=True, help="Whether to add black padding to make the cropped image a square. Default is True.")
    parser.add_argument('--postfix', type=str, default="", help="Postfix to be added to the cropped image's file name. Default is an empty string.")
    parser.add_argument('--crop_cat',  help="A list of categories (treated as string) to be cropped, separated by comma. Default is 1 for animal category", type=str, default='1')
    parser.add_argument('--max_workers', type=int, default=1, help="Number of workers for parallelization. Default is 1.")
    parser.add_argument('--logdir', type=str, default='.', help="Relative path to the log directory. Default is the current directory.")
    parser.add_argument('--force', type=bool, default=False, help="Whether to overwrite existing cropped images. Default is False.")
    
    args = parser.parse_args()
    args.crop_cat = args.crop_cat.strip().split(',')
    if args.detection_file.endswith('.csv'):
        _func = crop_images_from_csv
    elif args.detection_file.endswith('.json'):
        _func = crop_images_from_md_json
    else:
        raise ValueError("Detection file must be either a CSV or a JSON file.")
    _func(args.detection_file,
          img_dir=args.img_dir,
          cropped_dir=args.cropped_dir,
          square_crop=args.square_crop,
          postfix=args.postfix,
          crop_cat=args.crop_cat,
          max_workers=args.max_workers,
          logdir=args.logdir,
          force=args.force
          )
    

if __name__ == "__main__":
    main()

# Example Usage:
# crop-images-from-file path/to/csv_or_json_file path/to/image_dir path/to/cropped_dir --square_crop True --postfix mv5b --max_workers 4 --crop_cat 1 --logdir path/to/log_dir --force False