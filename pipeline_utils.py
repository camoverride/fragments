import cv2
import mediapipe as mp
import numpy as np

from image_processing_utils import get_average_face, simple_crop_face, is_face_wide_enough
from utils_from_ff import crop_align_image_based_on_eyes, \
    is_face_looking_forward, get_face_landmarks, get_additional_landmarks, \
    get_triangulation_indexes_for_landmarks, morph_align_face, get_average_landmarks



# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Initialize detection.
face_detection = \
    mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)

# Initialize image capture.
cap = cv2.VideoCapture(0)
cv2.waitKey(1000) # pause so first frame isn't dark


def get_faces_from_webcam(debug : bool):
    """
    If a face is detected, a tuple (np.ndarray, list) is
    returned, where the first element is a frame from the
    webcam, and the second element is a list of all the
    bounding boxes surrounding faces. If no faces are
    deteceted, a tuple of (False, False) is returned.

    NOTE: These bounding boxes can then be utilized with
    the function `simple_crop_face`

    Parameters
    ----------
    debug : bool
        Shows intermediate processing steps as images
        which pause execution until a key is pressed.

    Returns
    -------
    tuple
        A tuple of (False, False) if no faces are detected.
        A tuple of (np.ndarray, list) there the first element
        is a frame from the webcam, and the second element
        is a list of bounding boxes that contain faces.
    """
    # Get a frame from the webcam.
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to capture image from webcam.")

    if debug:
        cv2.imshow("Image from webcam", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Look for faces in the frame
    frame_data = face_detection.process(frame)

    # If there are no faces, return False.
    if not frame_data.detections:
        print("No faces detected!")
        return False, False

    # If there are faces, return the frame and listf of bounding boxes.
    relative_bbs = [detection.location_data.relative_bounding_box \
            for detection in frame_data.detections]

    return frame, relative_bbs


def face_processing_pipeline(image : np.ndarray,
                             bb : dict,
                             min_width : int,
                             margin_fraction : float,
                             height_output : int,
                             width_output : int,
                             l : float,
                             r : float,
                             t : float,
                             b : float,
                             debug : bool):
    """
    This function takes an `image` with a face contained by a bounding
    box `bb` and performs the following processing steps:
        - checks if the image is large enough
        - crops the image to the bounding box with a `margin_fraction`
        - checks if the face is looking forward. TODO: change sensitivity.
        - crops and rotates the face so the pupils are horizontal and aligned.
        - resizes the image to the desired output dimensions.
    If any of these steps fails, the function returns False.
    If the pipeline succeeds, a cropped and aligned face is returned.

    Parameters
    ----------
    image : np.ndarray
        An image from the webcam containing a face.
    bb : TODO: what type????
        A relative bounding box.
    min_width : int
        Returns False if the image is not at least `min_width`
    margin_fraction : float
        The fractional margin added to all sides during the initial
        `simple_crop_face`. NOTE: This is needed or else face_mesh
        won't detect a face in following steps.
    height_output : int
        How tall the final output image should be.
    width_output : int
        How wide the final output image should be.
    l : float
        The left margin for `crop_align_image_based_on_eyes`
    r : float
        The right margin for `crop_align_image_based_on_eyes`
    t : float
        The top margin for `crop_align_image_based_on_eyes`
    b : float
        The bottom margin for `crop_align_image_based_on_eyes`
    debug : bool
        Shows intermediate processing steps as images
        which pause execution until a key is pressed.

    Returns
    -------
    np.ndarray
        A forward-looking face that has been cropped, rotated,
        and pupil-aligned.
    """
    # Check if the face is wide enough.
    if not is_face_wide_enough(image=image,
                               bbox=bb,
                               min_width=min_width):
        print("Face is too small!")
        return False

    # Simple crop the image to the bounding box.
    face_cropped = simple_crop_face(image=image,
                                    bbox=bb,
                                    margin_fraction=margin_fraction)

    if debug:
        cv2.imshow("Simple cropped face", face_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Get the face_mesh landmarks.
    results = face_mesh.process(face_cropped)
    
    # If there are no results, return False
    if not results:
        return False

    if not results.multi_face_landmarks:
        return False

    # Get the landmarks.
    landmarks = results.multi_face_landmarks[0]
    face_height, face_width, _ = face_cropped.shape

    if not landmarks:
        print("No landmarks detected!")
        return False

    # Check if it's looking forward.
    face_forward = is_face_looking_forward(face_landmarks=landmarks,
                                           image_height=face_height,
                                           image_width=face_width)

    # If it's not looking forward, return False
    if not face_forward:
        print("Face isn't looking forward")
        return False

    if debug:    
        # Annotate the frame.
        face_annotated = np.copy(face_cropped)
        cv2.putText(face_cropped, f"{face_forward}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
        cv2.imshow("Forward face", face_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Process the images with face_mesh to rotate and align the pupils.
    face_cropped_rotated = crop_align_image_based_on_eyes(image=face_cropped,
                                                          l=l,
                                                          r=r,
                                                          t=t,
                                                          b=b)

    if debug:
        cv2.imshow("Face, cropped and pupil-aligned", face_cropped_rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Resize the image
    face_cropped_rotated_resized = cv2.resize(face_cropped_rotated, 
                                                 (width_output, height_output))
    
    if debug:
        cv2.imshow("Face, resized", face_cropped_rotated_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return face_cropped_rotated_resized
        

def morph_faces(face_1 : np.ndarray,
                face_2 : np.ndarray,
                triangulation_indexes_ : list,
                debug : bool) -> np.ndarray:
    """
    This is another processing pipeline. It accepts two images of
    faces that have already been processed with `face_processing_pipeline`
    and returns an averaged composite face.

    NOTE: both faces must be processed with the same arguments: margins,
    output dimensions, etc. or this function will throw an error.

    These are the processing steps:
    - Collect landmarks for both faces + additional landmarks.
    - Collect triangulation indexes for faces. TODO: pre-compute.
    - Morph both faces onto intermediate landmarks.
    - Average these faces together.

    Paramaters
    ----------
    face_1 : np.ndarray
        An image of a face. Must have been processed with the same
        arguments as `face_2`.
    face_2 : np.ndarray
        An image of a face. Must have been processed with the same
        arguments as `face_1`.
    triangulation_indexes : list
        A list of mappings between the points of the face, used when
        morphing.
    debug : bool
        Shows intermediate processing steps as images
        which pause execution until a key is pressed.

    Returns
    -------
    np.ndarray
        A face which is a composite created by morphing both faces
        to an intermediate landmark space, then alpha-blending them
        together (where alpha = 0.5, half and half)
    """
    # Collect the landmarks for both faces here.
    all_landmarks_both_faces = []

    for face in [face_1, face_2]:
        face_landmarks = get_face_landmarks(face)
        face_additional_landmarks = \
            get_additional_landmarks(image_height=face.shape[0],
                                    image_width=face.shape[1])
        face_all_landmarks = face_landmarks + face_additional_landmarks

        all_landmarks_both_faces.append(face_all_landmarks)
    
    face_1_landmarks, face_2_landmarks = all_landmarks_both_faces

    # Get triangulation indexes. NOTE: either face can be used.
    triangulation_indexes = \
    get_triangulation_indexes_for_landmarks(landmarks=face_1_landmarks,
                                            image_height=face_1.shape[0],
                                            image_width=face_1.shape[1])

    # Get the average landmarks
    average_landmarks = np.mean(all_landmarks_both_faces, axis=0).astype(int).tolist()

    # Morph each face onto the average points between them.
    morphed_faces = []
    for face in [face_1, face_2]:
        face_morphed = morph_align_face(source_face=face,
                                        target_face_all_landmarks=average_landmarks,
                                        triangulation_indexes=triangulation_indexes)
        morphed_faces.append(face_morphed)

    face_1_morphed, face_2_morphed = morphed_faces

    if debug:
        cv2.imshow("Face 1, morphed", face_1_morphed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Face 2, morphed", face_2_morphed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Average together the faces.
    averaged_face = get_average_face([face_1_morphed,
                                      face_2_morphed])

    if debug:
        cv2.imshow("Averaged face", averaged_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return averaged_face


def is_face_centered(relative_bb) -> bool:
    """
    Determines whether the center of a bounding box is within the middle
    square of a 3x3 grid (like a tic-tac-toe board) of the image.

    Parameters
    ----------
    relative_bb : object
        A bounding box object with attributes `xmin`, `ymin`, `width`, and `height`
        (all normalized to the image dimensions, i.e., in the range [0, 1]).

    Returns
    -------
    bool
        True if the center of the bounding box is within the middle square
        of the 3x3 grid, False otherwise.
    """
    # Extract normalized bounding box coordinates
    xmin = relative_bb.xmin
    ymin = relative_bb.ymin
    width = relative_bb.width
    height = relative_bb.height

    # Calculate the center of the bounding box
    center_x = xmin + (width / 2)
    center_y = ymin + (height / 2)

    # Define the boundaries of the middle square in the 3x3 grid
    middle_x_min = 1 / 3
    middle_x_max = 2 / 3
    middle_y_min = 1 / 3
    middle_y_max = 2 / 3

    # Check if the center of the bounding box is within the middle square
    is_centered = (middle_x_min <= center_x < middle_x_max) and \
                  (middle_y_min <= center_y < middle_y_max)

    return is_centered
