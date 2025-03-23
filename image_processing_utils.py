from typing import List
import numpy as np



def is_face_wide_enough(image: np.ndarray, bbox: dict, min_width: int) -> bool:
    """
    Checks if the face in the bounding box meets the minimum width requirement.

    Parameters
    ----------
    image : np.ndarray
        The image containing the face.
    bbox : dict
        A relative bounding box with keys 'xmin', 'ymin', 'width', 'height'.
    min_width : int
        The minimum width (in pixels) for the face to be considered valid.

    Returns
    -------
    bool
        True if the face is wide enough, False otherwise.
    """
    # Get the dimensions of the image
    h, w, _ = image.shape

    # Convert the relative bounding box width to absolute width
    bbox_width = int(bbox.width * w)

    # Check if the face meets the minimum width requirement
    return bbox_width >= min_width


# NOTE: This is a custom function NOT from frankenface
def get_average_face(images : List[np.ndarray]) -> np.ndarray:
    """
    Accepts a list of paths to some images and returns an average image.

    Parameters
    ----------
    images : List[np.ndarray]
        A list of images that will be averaged.

    Returns
    -------
    np.ndarray
        An averaged image.
    """
    # Initialize variables to store the sum of images and the count
    image_sum = None
    num_images = 0

    # Iterate over each image file
    for image in images:

        # Check if the image was loaded successfully
        if image is None:
            print(f"Warning: Could not load image . Skipping.")
            continue

        # Convert the image to float32 for accurate summation
        image = image.astype(np.float32)

        # Initialize the sum if this is the first image
        if image_sum is None:
            image_sum = np.zeros_like(image, dtype=np.float32)

        # Add the image to the sum
        image_sum += image
        num_images += 1

    # Check if any images were processed
    if num_images == 0:
        raise ValueError("No valid images found in the directory.")

    # Compute the average image
    averaged_image = image_sum / num_images

    # Convert back to uint8 for saving/displaying
    averaged_image = averaged_image.astype(np.uint8)

    return averaged_image


# NOTE: this is taken from the repo `face_yourself`
def simple_crop_face(image, bbox, margin_fraction):
    h, w, _ = image.shape

    # Calculate the bounding box dimensions
    bbox_width = int(bbox.width * w)
    bbox_height = int(bbox.height * h)

    # Calculate margins based on the required square size
    if bbox_width > bbox_height:
        diff = bbox_width - bbox_height
        margin_y = int((bbox_height * margin_fraction) + (diff / 2))
        margin_x = int(bbox_width * margin_fraction)
    else:
        diff = bbox_height - bbox_width
        margin_x = int((bbox_width * margin_fraction) + (diff / 2))
        margin_y = int(bbox_height * margin_fraction)

    # Calculate the coordinates for cropping
    x_min = max(int(bbox.xmin * w) - margin_x, 0)
    y_min = max(int(bbox.ymin * h) - margin_y, 0)
    x_max = min(int((bbox.xmin + bbox.width) * w) + margin_x, w)
    y_max = min(int((bbox.ymin + bbox.height) * h) + margin_y, h)

    # Make sure the crop is square
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width > crop_height:
        diff = crop_width - crop_height
        y_min = max(y_min - diff // 2, 0)
        y_max = min(y_max + diff // 2, h)
    elif crop_height > crop_width:
        diff = crop_height - crop_width
        x_min = max(x_min - diff // 2, 0)
        x_max = min(x_max + diff // 2, w)

    # Return the cropped square image
    return image[y_min:y_max, x_min:x_max]


