from typing import Sequence, Any, BinaryIO, Optional
from PIL import Image, ImageOps
import requests
from io import BytesIO
from typing import Union


def convert_yolo_to_xywh(yolo_box):
    """
    Converts a YOLO format bounding box to [x_min, y_min, width_of_box, height_of_box].

    Args:
        yolo_box (list): bounding box of format [x_center, y_center, width_of_box, height_of_box]

    Returns:
        list: bbox with coordinates represented as [x_min, y_min, width_of_box, height_of_box]
    """
    
    x_center, y_center, width_of_box, height_of_box = yolo_box
    x_min = x_center - width_of_box / 2.0
    y_min = y_center - height_of_box / 2.0
    return [x_min, y_min, width_of_box, height_of_box]

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


def open_image(input_file: Union[str, BytesIO]) -> Image:
    """Opens an image in binary format using PIL.Image and converts to RGB mode.
    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.
    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes
    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        response = requests.get(input_file)
        image = Image.open(BytesIO(response.content))
        try:
            response = requests.get(input_file)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise
    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError(f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    try:
        IMAGE_ROTATIONS = {
            3: 180,
            6: 270,
            8: 90
        }
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image

def load_image_general(input_file: Union[str, BytesIO]) -> Image.Image:
    """Loads the image at input_file as a PIL Image into memory.
    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.
    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes
    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image

    
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