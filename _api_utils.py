import requests
import cv2
import numpy as np
import base64
from typing import List


def morph_combine_images_api(api_url : str,
                             image_1_path : str,
                             image_2_path : str):
    """
    Sends two images to the morph-combine API to create a composite
    image.
    NOTE: order of the images does not matter.

    Parameters
    ----------
    api_url : str
        The URL of the morph-combine API endpoint.
    image_path_1 : str
        Path to the first image.
    image_path_2 : str
        Path to the second image.

    Returns
    -------
    np.ndarray
        A 3D array with shape (height, width, 3)
        representing an RGB image.
    """
    # Send the images to the API
    with open(image_1_path, "rb") as image_file_1, \
         open(image_2_path, "rb") as image_file_2:

        files = {"image_1": image_file_1,
                 "image_2": image_file_2}

        response = requests.post(api_url, files=files)


def image_to_video_api(api_url : str,
                       image : np.ndarray) -> List[np.ndarray]:
    """
    Sends an image to the video generation API and returns the
    animated video frames based on this image.

    Parameters
    ----------
    api_url : str
        The URL of the video generation API endpoint.
    image : np.ndarray
        The image file that will be sent to the API

    Returns
    -------
    List[numpy.ndarray]
        A list of numpy arrays, where each array represents a frame of
        the generated video. Each frame is a 3D array with shape
        (height, width, 3), representing an RGB image.

    Notes
    -----
    - The function sends the image to the API using a POST
    request with the image file in the payload.
    - The API is expected to return a JSON response containing
    base64-encoded JPEG frames.
    - The function decodes these frames into numpy arrays using
    OpenCV's `imdecode` function.

    Examples
    --------
    >>> api_url = "https://example.com/video-generation-api"
    >>> image_path = "input_image.jpg"
    >>> frames = image_to_video_api(api_url, image_path)
    >>> for frame in frames:
    ...     cv2.imshow("Frame", frame)
    ...     cv2.waitKey(0)
    """
    # Write to a jpg for easier sending
    cv2.imwrite("tmp_avg_face.jpg", image)

    # Send the image to the API
    with open("tmp_avg_face.jpg", "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(api_url, files=files)

    # Print the status code
    print("Status Code:", response.status_code)

    # Check if the request was successful
    if response.status_code != 200:
        print("Error:", response.status_code)
        return None

    try:
        # Parse the response as JSON
        response_json = response.json()

    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON response")
        return None

    # Decode the base64-encoded JPEG frames into numpy arrays
    frames = []

    for frame_base64 in response_json.get("frames", []):
        # Decode base64 string into bytes
        frame_bytes = base64.b64decode(frame_base64.encode("utf-8"))

        # Convert bytes to numpy array and decode JPEG
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)

        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        frames.append(frame)

    return frames



# Example usage
if __name__ == "__main__":

    # Get the frames from the API.
    frames = image_to_video_api(
        api_url="http://127.0.0.1:5000/animate-image",
        image_path="cam.jpg"
    )

    # Show the frames. Press a key to cycle through the frames.
    if frames:
        print(f"Received {len(frames)} frames.")
        for i, frame in enumerate(frames):
            cv2.imshow(f"Frame {i}", frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
