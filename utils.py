from .src.utils import color_by_id
from pathlib import Path

def imcrops(image, boxes):
    """
    Crops multiple regions from the image based on bounding boxes.

    Parameters:
    - image: numpy array representing the image (H, W, C)
    - boxes: list or array of bounding boxes, each in the format (x1, y1, x2, y2)

    Returns:
    - crops: list of cropped image regions as numpy arrays
    """
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        crops.append(crop)
    return crops

def version():
    v = Path(__file__).parent / "VERSION"
    return v.read_text().strip()