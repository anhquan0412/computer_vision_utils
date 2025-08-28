from typing import Sequence, BinaryIO, Optional, Any, Dict, List
from PIL import Image, ImageOps
from megadetector.visualization import visualization_utils as md_viz
from io import BytesIO
from typing import Union
import io
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import logging
import collections
from datetime import timedelta, datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
azure_logger = logging.getLogger("azure.core.pipeline.policies")
azure_logger.setLevel(logging.WARNING)


def load_local_image(img_path: str |  BinaryIO) -> Optional[Image.Image]:
    """Attempts to load an image from a local path."""
    try:
        with Image.open(img_path) as img:
            img.load()
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        return img
    except OSError as e:  # PIL.UnidentifiedImageError is a subclass of OSError
        exception_type = type(e).__name__
        print(f'Unable to load {img_path}. {exception_type}: {e}.')
    return None


def load_image_general(input_file: Union[str, BytesIO]) -> Image.Image:
    """Loads the image at input_file as a PIL Image into memory.
    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.
    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes
    Returns: PIL.Image.Image, in RGB mode
    """
    image = md_viz.open_image(input_file)
    image.load()
    return image

def download_img(img_file,input_container_client,ignore_exif_rotation=True,load_img=True):
    use_url = img_file.startswith(('http://', 'https://'))
    if not use_url and input_container_client is not None:
        downloader = input_container_client.download_blob(img_file)
        img_file = io.BytesIO()
        blob_props = downloader.download_to_stream(img_file)

    img = md_viz.open_image(img_file,ignore_exif_rotation=ignore_exif_rotation)
    if load_img:
        img.load()
    return img

def yolo_to_crop_format(bbox_yolo):
    """
    Convert from YOLO format [x_center, y_center, width, height] to 
    crop format [x_min, y_min, width, height]
    """
    x_center, y_center, width, height = bbox_yolo
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    return [x_min, y_min, width, height]

def crop_image(img: Image.Image, bbox_norm: Sequence[float], square_crop: bool) -> Image.Image:
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    
    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        raise ValueError(f'Skipping size-0 crop (w={box_w}, h={box_h})')

    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    return crop



def is_color_image(
    image_path: str,
    saturation_percentile_threshold: int = 20, # Threshold for the 99th percentile saturation
    channel_diff_threshold: int = 10,
    saturation_percentile: int = 99 # Which percentile to check (99 ignores top 1% outliers)
) -> bool | None:
    """
    Checks if an image is color or grayscale/black and white using a combined
    check of channel similarity AND high-percentile saturation.

    An image is classified as B&W only if BOTH the average channel difference
    AND the specified high percentile (e.g., 99th) of saturation values are
    below their respective thresholds. Otherwise, it's considered color.
    This approach is more robust to noise/outlier pixels than using max saturation.

    Args:
        image_path: Path to the image file.
        saturation_percentile_threshold: Threshold for the high percentile saturation value.
                                         (Range 0-255). Adjust based on testing (e.g., 15-35).
        channel_diff_threshold: Threshold for the average absolute difference
                                between color channels. (Range 0-255)
                                Adjust based on testing (e.g., 5-15).
        saturation_percentile: The percentile of saturation values to check (e.g., 99).

    Returns:
        True if the image is likely color.
        False if the image is likely grayscale/black and white.
        None if the image cannot be loaded or processed.
    """
    if not (0 < saturation_percentile <= 100):
        logging.error("saturation_percentile must be between 0 and 100.")
        return None # Invalid input

    try:
        img = cv2.imread(image_path)

        if img is None:
            logging.warning(f"Could not load image: {image_path}")
            return None

        # If the image has fewer than 3 channels, it's definitely grayscale
        if len(img.shape) < 3 or img.shape[2] < 3:
            return False # Classified as B&W

        # 1. Channel Similarity
        # Use float32 for calculations to avoid potential intermediate overflows and precision issues
        b_channel = img[:, :, 0].astype(np.float32)
        g_channel = img[:, :, 1].astype(np.float32)
        r_channel = img[:, :, 2].astype(np.float32)
        # Calculate mean absolute differences
        diff_bg = np.mean(np.abs(b_channel - g_channel))
        diff_gr = np.mean(np.abs(g_channel - r_channel))
        mean_channel_diff = (diff_bg + diff_gr) / 2

        # 2. Saturation (using percentile)
        # Convert the image from BGR to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Extract the Saturation channel (channel index 1)
        saturation_channel = hsv_img[:, :, 1]
        # Calculate the specified percentile of saturation values
        # This ignores outlier pixels (e.g., top 1% if percentile=99)
        # Flatten the channel array for percentile calculation
        saturation_value_percentile = np.percentile(saturation_channel.flatten(), saturation_percentile)

        # --- Combined Check ---
        # Classify as B&W ONLY if BOTH channel difference AND high-percentile saturation are low
        if mean_channel_diff < channel_diff_threshold and saturation_value_percentile < saturation_percentile_threshold:
            # This condition catches true grayscale and typical IR images more reliably
            return False # Classified as B&W
        else:
            # If either channel difference is significant OR the bulk of saturation values are significant,
            # it's likely a color image (even if muted or low-light color).
            return True # Classified as Color

    except Exception as e:
        # Log the error
        logging.error(f"Error processing image {image_path}: {e}", exc_info=False) # Set exc_info=True for full traceback
        return None

def process_colorcheck_parallel(
    image_paths: list[str],
    saturation_percentile_threshold: int = 20, # Renamed threshold parameter
    channel_diff_threshold: int = 10,
    saturation_percentile: int = 99, # Added percentile parameter
    max_workers: int = 8
):
    """
    Processes a list of images in parallel to check if they are color using percentile saturation.

    Args:
        image_paths: A list of paths to image files.
        saturation_percentile_threshold: The saturation percentile threshold.
        channel_diff_threshold: The channel difference threshold.
        saturation_percentile: The percentile to use for saturation check (e.g., 99).
        max_workers: The maximum number of threads to use for parallel processing.

    Returns:
        A dictionary mapping image paths to their color status (True=color, False=BW, None=Error).
    """
    results = {}
    start_time = time.time()
    total_images = len(image_paths)
    logging.info(f"Starting parallel processing for {total_images} images using up to {max_workers} workers.")

    # Use ThreadPoolExecutor for parallel I/O bound tasks (like reading files)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for each image processing task, passing all thresholds/params
        future_to_path = {
            executor.submit(
                is_color_image,
                path,
                saturation_percentile_threshold,
                channel_diff_threshold,
                saturation_percentile # Pass the percentile value
            ): path for path in image_paths
        }

        # Use standard tqdm for a console/text-based progress bar
        for future in tqdm(as_completed(future_to_path), total=total_images, desc="Processing Images"):
            path = future_to_path[future]
            try:
                result = future.result()
                results[path] = result
            except Exception as exc:
                # Log the exception that might occur during future.result() call itself
                logging.error(f'{path} generated an exception during future retrieval: {exc}', exc_info=False)
                results[path] = None # Mark as error

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info(f"Finished processing {len(results)} images in {processing_time:.2f} seconds.")
    return results

# # Usage of color check function
# CHANNEL_DIFF_THRESHOLD = 10       # If mean channel diff is BELOW this AND...
# SATURATION_PERCENTILE_THRESHOLD = 20 # 99th percentile saturation is BELOW this -> classify as B&W.
# SATURATION_PERCENTILE = 99        # Use the 99th percentile (ignores top 1% outliers)
# NUM_WORKERS = 1 # os.cpu_count()
# results_dict = process_colorcheck_parallel(
#     image_paths=df.abs_file.tolist(),
#     saturation_percentile_threshold=SATURATION_PERCENTILE_THRESHOLD,
#     channel_diff_threshold=CHANNEL_DIFF_THRESHOLD,
#     saturation_percentile=SATURATION_PERCENTILE,
#     max_workers=NUM_WORKERS
# )
# with open(f'color_check.json', 'w') as f: 
#     json.dump(results_dict, f)


def sequence_assignment(
    image_data: List[Dict[str, Any]], 
    time_gap: int = 3
) -> Dict[str, List[Dict[str, str]]]:
    """
    Assigns sequence IDs to camera trap images based on their location and timestamp.

    This function groups images by their parent directory, sorts them chronologically,
    and then segments them into sequences. A new sequence is started if the time
    difference between an image and its predecessor exceeds the specified time_gap.

    Args:
        image_data: A list of dictionaries, where each dictionary represents an
                    image and should contain 'file' and 'datetime' keys.
        time_gap: The maximum time in seconds allowed between consecutive
                  images in the same sequence. Defaults to 3.

    Returns:
        A dictionary with a single key 'images', containing a list of
        dictionaries, each with 'file_name' and its assigned 'seq_id'.
        
    Note:
        Image records that are missing the 'file' or 'datetime' key, or have
        a malformed datetime string, will be silently ignored.
    """
    grouped_by_dir = collections.defaultdict(list)
    for image_info in image_data:
        try:
            filepath_str = image_info['file']
            datetime_str = image_info['datetime']
            filepath = Path(filepath_str)
            parent_dir = str(filepath.parent)
            
            datetime_obj = datetime.strptime(
                datetime_str, '%Y:%m:%d %H:%M:%S'
            )
            
            grouped_by_dir[parent_dir].append({
                'file': filepath_str, 
                'datetime_obj': datetime_obj
            })
        except (KeyError, ValueError):
            # If required keys are missing or datetime format is invalid,
            # silently ignore this record and continue to the next.
            continue

    final_results = []
    time_delta_gap = timedelta(seconds=time_gap)

    for parent_dir, images_in_group in grouped_by_dir.items():
        if not images_in_group:
            continue
            
        sorted_images = sorted(
            images_in_group, 
            key=lambda x: (x['file'],x['datetime_obj'])
        )


        sequence_counter = 1
        
        # The first image always starts the first sequence for its directory.
        first_image = sorted_images[0]
        seq_id = f"{parent_dir}/sequence_{sequence_counter}"
        final_results.append({'file_name': first_image['file'], 'seq_id': seq_id})

        for i in range(1, len(sorted_images)):
            current_image = sorted_images[i]
            previous_image = sorted_images[i - 1]

            # If the gap is too large, start a new sequence.
            if current_image['datetime_obj'] - previous_image['datetime_obj'] > time_delta_gap:
                sequence_counter += 1
            
            seq_id = f"{parent_dir}/sequence_{sequence_counter}"
            final_results.append({
                'file_name': current_image['file'], 
                'seq_id': seq_id
            })
            
    return {'images': final_results}