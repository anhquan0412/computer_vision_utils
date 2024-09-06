from typing import Sequence, Any, BinaryIO, Optional
from PIL import Image, ImageOps

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