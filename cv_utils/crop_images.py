import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps
from datetime import datetime
import csv
import pandas as pd
from tqdm import tqdm

def crop_images_from_file(detection_csv,img_dir,cropped_dir,square_crop=True,postfix="",max_workers=1,logdir='.'):
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

    max_workers (int): Number of workers for parallelization. Default is 1.

    logdir (str): Relative path to the log directory. Default is the current directory.

    Returns:
    None
    """

    error_log = []
    def crop_and_save_image(img_path,img_dir,cropped_dir,bbox_coord,square_crop,bbox_rank,postfix):
        img_path = Path(img_path)
        img_dir = Path(img_dir)
        cropped_dir = Path(cropped_dir)
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
                error_log.append((img_path.as_posix(), bbox_coord, 'Box coordinate error'))
                return None
        
            crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
            if square_crop and (box_w != box_h):
                crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)


            dest_absolute_path = cropped_dir/img_path
            dest_absolute_path.parent.mkdir(exist_ok=True,parents=True)
            if postfix.strip()!="": postfix=f"_{postfix}"
            dest_fname = f"{dest_absolute_path.stem}___crop{bbox_rank:>02d}{postfix}{dest_absolute_path.suffix}"
            dest_absolute_path = dest_absolute_path.parent / dest_fname
            crop.save(dest_absolute_path)
            return (img_path.parent/dest_fname).as_posix()
        except Exception as e:
            exception_type = type(e).__name__
            error_log.append((img_path.as_posix(), bbox_coord, f'{exception_type}: {e}'))
            return None

    def wrapper(args):
        return crop_and_save_image(*args)
    
    df = pd.read_csv(detection_csv)
    org_columns=df.columns.tolist()

    assert "file" in df.columns.values, "There must be a column called 'file' containing relative paths of images"
    assert "detection_bbox" in df.columns.values, "There must be a column called 'detection_bbox' containing the normalized bbox coordinates as lists"
    assert "bbox_rank" in df.columns.values, "There must be a column called 'bbox_rank' containing the ranking of the bbox to the image.\n For image with multiple bboxes, this is used to distinguish the result"
    
    if df.detection_bbox.dtype == "object":
        df.detection_bbox = df.detection_bbox.apply(lambda x: tuple(ast.literal_eval(x)))
    df.bbox_rank = df.bbox_rank.astype(int)
    
    Path(cropped_dir).mkdir(exist_ok=True,parents=True)

    df['img_dir'] = str(img_dir)
    df['cropped_dir'] = str(cropped_dir)
    df['square_crop'] = square_crop
    df['postfix'] = postfix
    args_list = df[['file','img_dir','cropped_dir','detection_bbox','square_crop','bbox_rank','postfix']].values.tolist()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        cropped_paths = list(tqdm(executor.map(wrapper, args_list)))
        
    assert len(cropped_paths)==df.shape[0]
    df['cropped_file'] = cropped_paths
    df[org_columns+['cropped_file']].to_csv(str(Path(detection_csv)),index=False)

    # Write the error log to a CSV file
    if len(error_log):
        logdir = Path(logdir)
        logdir.mkdir(exist_ok=True,parents=True)
        error_log_path = Path(logdir) / f"cropping_errors_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(error_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["img_path", "detection_bbox", "error"])
            writer.writerows(error_log)


def main():
    parser = argparse.ArgumentParser(description="Crop images based on bounding box coordinates provided in a CSV file.")
    parser.add_argument('detection_csv', type=str, help="The absolute path to the CSV file containing image file paths and bounding box coordinates.")
    parser.add_argument('img_dir', type=str, help="The absolute path to the directory containing the raw images.")
    parser.add_argument('cropped_dir', type=str, help="The absolute path to the directory where the cropped images will be saved.")
    parser.add_argument('--square_crop', type=bool, default=True, help="Whether to add black padding to make the cropped image a square. Default is True.")
    parser.add_argument('--postfix', type=str, default="", help="Postfix to be added to the cropped image's file name. Default is an empty string.")
    parser.add_argument('--max_workers', type=int, default=1, help="Number of workers for parallelization. Default is 1.")
    parser.add_argument('--logdir', type=str, default='.', help="Relative path to the log directory. Default is the current directory.")
    
    args = parser.parse_args()
    
    crop_images_from_file(
        detection_csv=args.detection_csv,
        img_dir=args.img_dir,
        cropped_dir=args.cropped_dir,
        square_crop=args.square_crop,
        postfix=args.postfix,
        max_workers=args.max_workers,
        logdir=args.logdir
    )

if __name__ == "__main__":
    main()

# Example Usage:
# crop-images-from-file C:\Users\testing.csv D:\image D:\image\cropped --square_crop True --postfix mv5b --max_workers 4 --logdir C:\Users\cropping_logs