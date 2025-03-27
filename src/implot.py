import cv2
import numpy as np
from pathlib import Path

def implots(imgs, cal, row, tags=None):
    """
    Arranges multiple images in a grid format (rows and columns) with large tags on top.

    Args:
        imgs (list of numpy arrays): List of images to arrange.
        cal (int): Number of columns in the grid.
        row (int): Number of rows in the grid.
        tags (list of str, optional): List of tags for each image.

    Returns:
        numpy array: Concatenated grid image with larger tags.
    """
    assert len(imgs) <= cal * row, "Not enough rows and columns for all images."

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
    while len(resized_imgs) < cal * row:
        resized_imgs.append(cv2.resize(none_img, (max_w, max_h)))

    # Default tags if none provided
    if tags is None:
        grid_rows = [
            np.hstack(resized_imgs[i * cal:(i + 1) * cal])
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
        np.hstack(labeled_imgs[i * cal:(i + 1) * cal])
        for i in range(row)
    ]

    # Stack rows vertically
    concatenated = np.vstack(grid_rows)

    return concatenated