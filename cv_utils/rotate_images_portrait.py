from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from .common_utils import get_all_image_paths

def rotate_image(img_path, source_dir, destination_dir=None, rotate_left=True):
    if destination_dir is None:
        destination_dir = source_dir
    img_path = Path(img_path)
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    
    with Image.open(img_path) as img:
        width, height = img.size
        if width > height:
            img = img.rotate(90 if rotate_left else -90, expand=True)
            relative_path = img_path.relative_to(source_dir)
            dest_path = destination_dir/relative_path
            dest_path.parent.mkdir(exist_ok=True,parents=True)
        img.save(dest_path)

def rotate_images_to_portrait(source_folder, destination_folder=None, rotate_left=True, max_workers=1):
    image_paths = get_all_image_paths(source_folder)
    
    if destination_folder is None:
        destination_folder = source_folder
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda f: rotate_image(f, source_folder,destination_folder, rotate_left), image_paths), total=len(image_paths), desc="Processing images"))

def main():
    parser = argparse.ArgumentParser(description="Rotate images to portrait orientation.")
    parser.add_argument('source_folder', type=str, help="The absolute path to the source folder containing images.")
    parser.add_argument('destination_folder', type=str, nargs='?', default=None, help="The absolute path to the destination folder for rotated images. If not provided, images will be overwritten.")
    parser.add_argument('--rotate_left', type=bool, default=True, help="Whether to rotate the images to the left or right. Default is True.")
    parser.add_argument('--max_workers', type=int, default=1, help="Number of workers for parallelization. Default is 1.")
    
    args = parser.parse_args()
    
    rotate_images_to_portrait(
        source_folder=args.source_folder,
        destination_folder=args.destination_folder,
        rotate_left=args.rotate_left,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()

# Example usage:
# rotate-images-portrait /absolute_path/to/source/folder
# rotate-images-portrait /absolute_path/to/source/folder /absolute_path/to/destination/folder --rotate_left=False --max_workers=4
