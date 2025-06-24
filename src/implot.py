import cv2
import numpy as np
from pathlib import Path
from norfair.drawing.drawer import Drawer
from .utils import color_by_id

def plot_image_grid(imgs, col, row, tags=None):
    """
    Arranges multiple images in a grid format (rows and columns) with large tags on top.

    Args:
        imgs (list of numpy arrays): List of images to arrange.
        col (int): Number of columns in the grid.
        row (int): Number of rows in the grid.
        tags (list of str, optional): List of tags for each image.

    Returns:
        numpy array: Concatenated grid image with larger tags.
    """
    assert len(imgs) <= col * row, "Not enough rows and columns for all images."

    # Convert grayscale images to 3-channel BGR
    imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in imgs]

    # Find the maximum height and width
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)

    # Resize all images to match the largest one
    resized_imgs = [cv2.resize(img, (max_w, max_h)) for img in imgs]

    # Load the fallback image
    none_img_path = Path(__file__).parents[1] / 'assets/hello_opencv.png'
    none_img = cv2.imread(str(none_img_path)) if none_img_path.exists() else np.zeros((max_h, max_w, 3), dtype=np.uint8)

    # Fill missing images with placeholders
    while len(resized_imgs) < col * row:
        resized_imgs.append(cv2.resize(none_img, (max_w, max_h)))

    # Default tags if none provided
    if tags is None:
        grid_rows = [
            np.hstack(resized_imgs[i * col:(i + 1) * col])
            for i in range(row)
        ]
        concatenated = np.vstack(grid_rows)
        return concatenated
    elif len(tags) < len(resized_imgs):
        tags.extend(["Unknown"] * (len(resized_imgs) - len(tags)))  # Fill missing tags

    # Tag settings for bigger size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max_h * 0.65 / 480  # Increased font size
    font_thickness = round(max_h * 1 / 480)  # Thicker text
    text_color = (0, 0, 0)  # White text
    bg_color = (255, 255, 255)  # Black background for the tag

    text_height = round(max_h * 30 / 480)  # Bigger space for text
    
    labeled_imgs = []
    for img, tag in zip(resized_imgs, tags):
        # Create a black rectangle for the text background
        text_img = np.full((text_height, max_w, 3), bg_color, dtype=np.uint8)

        # Put text at the center of the black rectangle
        text_size = cv2.getTextSize(tag, font, font_scale, font_thickness)[0]
        text_x = (max_w - text_size[0]) // 2
        text_y = (text_height + text_size[1]) // 2
        cv2.putText(text_img, tag, (text_x, text_y), font, font_scale, text_color, font_thickness)

        # Stack text and image vertically
        labeled_img = np.vstack((text_img, img))
        labeled_imgs.append(labeled_img)

    # Split images into rows
    grid_rows = [
        np.hstack(labeled_imgs[i * col:(i + 1) * col])
        for i in range(row)
    ]

    # Stack rows vertically
    concatenated = np.vstack(grid_rows)

    return concatenated

def draw_boxes(image, 
               boxes, 
               ids=[], 
               labels=[], 
               default_color=(0,255,0),
               size=None,
               thickness=None,):
    plot = image.copy()

    if thickness is None:
        thickness = int(max(plot.shape) / 500)
    if size is None:
        size = min(max(max(plot.shape) / 4000, 0.5), 1.5)

    # Default color if ids not provided
    default_color = default_color

    if len(ids) == len(boxes):
        colors = [color_by_id(int(i)) for i in ids]
    else:
        colors = [default_color] * len(boxes)

    for ibox, box in enumerate(boxes):
        box = np.array(box, dtype=np.int32).reshape(2, 2)
        color = colors[ibox]

        text_anchor = (
            box[0, 0] - thickness // 2,
            box[0, 1] - thickness // 2 - 1,
        )

        # Drawing rectangle
        Drawer.rectangle(frame=plot, points=box, color=color, thickness=thickness)

        text = ''

        # Drawing id if available
        if ibox < len(ids):
            text += f'{ids[ibox]}'

        # Drawing label if available
        if labels and ibox < len(labels):
            text += f' {labels[ibox]}'
        
        if text != '':
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness,)
            text_anchor = (text_anchor[0], 
                          text_anchor[1] + text_height 
                          if text_anchor[1] - text_height < 0
                          else text_anchor[1])
            Drawer.text(frame=plot, text=text, position=text_anchor, thickness=thickness, color=color)
            
    return plot

def plot_image_grid_optimize(imgs, col, row):
    """
    Combine a list of images into a grid with col columns and row rows.
    If images have different sizes, resize them to match the smallest dimensions.
    If there are fewer images than needed, pad with black images.

    Args:
        imgs: List of images as numpy arrays
        col: Number of columns in the output grid
        row: Number of rows in the output grid

    Returns:
        Combined image as a numpy array
    """
    if not imgs:
        raise ValueError("Image list cannot be empty")

    # Find the smallest height and width among all images
    min_h = min(img.shape[0] for img in imgs)
    min_w = min(img.shape[1] for img in imgs)

    # Determine number of channels (handle both grayscale and color images)
    channels = 1
    for img in imgs:
        if len(img.shape) == 3:
            channels = max(channels, img.shape[2])

    # Resize all images to the smallest dimensions
    resized_images = []
    for img in imgs:
        if len(img.shape) == 2:  # Grayscale image
            img_resized = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
            if channels > 1:
                img_resized = np.stack([img_resized]*channels, axis=-1)
        else:  # Color image
            img_resized = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_AREA)
            if img_resized.shape[2] < channels:  # Handle case where some images have fewer channels
                img_resized = np.pad(img_resized, ((0,0),(0,0),(0,channels-img_resized.shape[2])), 
                                   mode='constant', constant_values=0)
        resized_images.append(img_resized)

    # Create a black image for padding if needed
    if channels == 1:
        black_image = np.zeros((min_h, min_w), dtype=resized_images[0].dtype)
    else:
        black_image = np.zeros((min_h, min_w, channels), dtype=resized_images[0].dtype)

    # Calculate how many images we need in total
    total_images = col * row
    current_images = len(resized_images)

    # Pad the image list with black images if necessary
    if current_images < total_images:
        resized_images = resized_images + [black_image] * (total_images - current_images)

    # Combine images into rows first
    rows = []
    for row in range(row):
        row_start = row * col
        row_end = row_start + col
        row_images = resized_images[row_start:row_end]

        # Combine images horizontally
        row_combined = np.hstack(row_images)
        rows.append(row_combined)

    # Combine rows vertically
    combined = np.vstack(rows)

    return combined