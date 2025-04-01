import copy
import logging
import os
import sys
import threading
import yaml

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

from _image_processing_utils import simple_crop_face, quantify_blur, is_face_wide_enough, \
is_face_centered, get_faces_from_camera, get_face_landmarks, get_additional_landmarks, \
    morph_align_face, is_face_looking_forward, crop_align_image_based_on_eyes, \
        get_average_face



# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Initialize detection.
face_detection = \
    mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)

# Globals for tracking images.
processed_faces = []
processed_face_landmarks = []
recent_embeddings = []

# Initialize a global lock
memory_lock = threading.Lock()


def collect_faces(camera_type : str,
                  blur_threshold : float,
                  face_memory : int,
                  embedding_memory : int,
                  tolerance : float,
                  min_width : int,
                  margin_fraction : float, # TODO: work on this!
                  height_output : int,
                  width_output : int,
                  l : float,
                  r : float,
                  t : float,
                  b : float,
                  triangulation_indexes : list, # TODO: add this!
                  check_centering : bool,
                  check_forward : bool,
                  debug_images : bool) -> bool: # TODO: remove debug! 
    """
    This function gets faces from the webcam and applies a processing
    pipeline. After this pipeline executes, if a new face has been
    detected, it is added to a collage.

    These are the steps of the processing pipeline. If any of the conditions
    in the pipeline aren't met, it returns False:
        - Get an image from the webcam along with bounding boxes aroung faces.
        - Check if the face is centered in the image, so that when cropping
            occurs, there will be space for a wide crop.
        - Check if the face is too blurry using a tight crop around the face.
        - Check if the image of the face is sufficiently wide (in pixels).
        - Check if the face is looking forward.
        - Crop and rotate the face.
        - Resize the face.
        - Embed the face and check if the face has already been detected.
        - Save the processed face and record it, its path, and its landmarks
            in some global variables to be shared with the `dispaly` function.

    After the processing pipeline, a display image is created. If any of the
    conditions in this process aren't met (such as not enough faces) the
    process returns False:
        - Get the two most recent faces.
        - Average out the landmarks from these faces.
        - Align both faces to the averaged "target" landmarks.
        - Average these faces together to form a single image.
        - Use the API to created an animated video from the image.
    
    Parameters
    ----------
    camera_type : str
        Whether we're using the `picam` or `webcam`.
    blur_threshold : float
        The maximum blurriness allowed in a face image. The face image
        should be cropped tightly to the face. The blurriness is calculated
        using the Laplacian. Faces that exceed this value will be skipped.
        NOTE: this must be manually tested with every camera and location
        setup (because of lighting conditions, etc.)
    face_memory : int
        How many faces/embeddings/landmarks should we keep in memory? If
        too many accumulate, the machine can crash! NOTE: in the default
        mode, only the two most recent faces are used.
    embedding_memory : int
        How many embeddings are remembered? NOTE: if `embedding_memory` <
        `face_memory`, then repeated faces are allowed inton the collage.
    tolerance : float
        The face recognition tolerance. NOTE: 0.6 is standard in the industry.
        Higher values mean that when comparing two embeddings, the algorithm
        is more likely to decide they are different. Lower tolerance means
        that the algorithm is more likely to say that the embeddings match.
    min_width : int
        The minimum width of the detected face in pixels. Faces smaller than
        this value will be skipped.
    margin_fraction : float
        For the simple cropping algorithm, how much width/height should be
        added to the face as a margin, in terms of the existing width/height
        of the face bounding box.
    height_output : int
        The height that the output image is resized to. Resizing does not
        preserve aspect ratio.
    width_output : int
        The width that the output image is resized to. Resizing does not
        preserve aspect ratio.
    l : float
        The left margin around the face bounding box in terms of K, where
        K is the distance between the pupils.
    r : float
        The right margin around the face bounding box in terms of K, where
        K is the distance between the pupils.
    t : float
        The top margin around the face bounding box in terms of K, where
        K is the distance between the pupils.
    b : float
        The bottom margin around the face bounding box in terms of K, where
        K is the distance between the pupils.
    triangulation_indexes : list or None
        The indexes of the triangles that connect all the landmarks in the
        face image. This is used for morphine (affine transformations).
        NOTE: this should be pre-computed.
        TODO: precompute this.
    check_centering : bool
        Check if the face is centered.
    check_forward : bool
        Check if the face is looking forward
    debug_images : bool
        If True, images of intermediate processing steps are displayed.
        This should always be False when used in production mode.

    Returns
    -------
    None or False
        False if one of the processing conditions wasn't met, such
        as a face not looking forward. Or False after the processing
        steps if there were insufficient images to form an animation
        (or if this animation has already been formed!).

        If processing conditions are met, images are written to 
        `images/faces`.
        
        If collage conditions are met, .npz archices of animations
        are written to `images/collages`.
    """
    # Declare all the globals
    global processed_faces, processed_face_landmarks, recent_embeddings

    # Wrap everything in a giant try/except
    try:
        # Get an image from the webcam along with face bounding boxes.
        frame, bbs = get_faces_from_camera(camera_type=camera_type,
                                           debug=debug_images)

        # If bbs exists, then faces have been detected.
        if not bbs:
            logging.debug("No faces detected!!!")
            return False

        # There might be multiple faces in the image.
        for bb in bbs:

            # Check if face is too far from the center.
            if check_centering:
                if not is_face_centered(bb):
                    logging.debug("Face is not centered!!!")
                    return False

            # Get a simple-cropped face with tight margins for blur detection.
            simple_cropped_face_tight_margins = simple_crop_face(frame,
                                                                 bb,
                                                                 margin_fraction=0)

            # Test if the image is too blurry.
            if quantify_blur(simple_cropped_face_tight_margins) > blur_threshold:
                logging.info("Face is blurry!!!")
                return False

            if not is_face_wide_enough(image=frame,
                                       bbox=bb,
                                       min_width=min_width):
                logging.info("Face is too small!")
                return False

            # Simple crop the image to the bounding box.
            # NOTE: this might capture multiple faces and introduce bugs.
            simple_cropped_face_with_margin = \
                simple_crop_face(image=frame,
                                 bbox=bb,
                                 margin_fraction=margin_fraction)

            if debug_images:
                cv2.imshow("Simple cropped face with margin",
                           simple_cropped_face_with_margin)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Get the face_mesh landmarks.
            face_mesh_results = face_mesh.process(simple_cropped_face_with_margin)
            
            # If there are no results, return False
            if not face_mesh_results:
                return False
            if not face_mesh_results.multi_face_landmarks:
                return False

            # Get the landmarks.
            landmarks = face_mesh_results.multi_face_landmarks[0]

            if not landmarks:
                logging.info("No landmarks detected!")
                return False

            # Check if it's looking forward.
            face_height, face_width, _ = simple_cropped_face_with_margin.shape
            if check_forward:
                face_forward = is_face_looking_forward(face_landmarks=landmarks,
                                                    image_height=face_height,
                                                    image_width=face_width)

                # If it's not looking forward, return False
                if not face_forward:
                    logging.debug("Face isn't looking forward")
                    return False

                if debug_images:    
                    # Annotate the frame.
                    face_annotated = np.copy(simple_cropped_face_with_margin)
                    cv2.putText(simple_cropped_face_with_margin,
                                f"{face_forward}",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA)
                    cv2.imshow("Forward face", face_annotated)
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()

            # Process the images with face_mesh to rotate and align the pupils.
            try:
                face_cropped_rotated = \
                    crop_align_image_based_on_eyes(image=frame,
                                                   bb=bb,
                                                   l=l,
                                                   r=r,
                                                   t=t,
                                                   b=b)
            except ValueError as e:
                logging.warning("Error face crop/rotate!")
                logging.warning(e)
                return False

            if debug_images:
                cv2.imshow("Face, cropped and pupil-aligned", face_cropped_rotated)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Resize the image
            current_face = cv2.resize(face_cropped_rotated, 
                                      (width_output, height_output))
            
            if debug_images:
                cv2.imshow("Face, resized", current_face)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Check if the face has been seen before.
            # NOTE: use the face that has not been rotated and resized, as face_recognition
            # often fails with rotated images (unsure why!)
            simple_cropped_face_with_margin = cv2.cvtColor(simple_cropped_face_with_margin,
                                                           cv2.COLOR_BGR2RGB)
            current_face_embedding = \
                face_recognition.face_encodings(simple_cropped_face_with_margin)

            # If the face was able to be successfully embedded
            if not current_face_embedding:
                logging.warning("Face could not be embedded!!!")
                return False

            # Get the face embedding.
            # TODO: why index with [0] - probably because multiple embeddings might
            # be returned if the image has multiple faces...
            # NOTE: this is a potential source of bugs.
            current_face_embedding = current_face_embedding[0]

            # Check if the face has been recently embedded (recognized).
            results = face_recognition.compare_faces(recent_embeddings,
                                                     current_face_embedding,
                                                     tolerance=tolerance)

            if any(results):
                logging.info("The face has been seen before!!!")
                return False

            # Get all the face landmarks for later morphing.
            face_landmarks = get_face_landmarks(current_face)
            if not face_landmarks:
                logging.info("Could not get landmarks!!!")
                return False

            additional_landmarks = \
                get_additional_landmarks(image_height=current_face.shape[0],
                                         image_width=current_face.shape[1])
            current_face_all_landmarks = face_landmarks + additional_landmarks

            # If faces have aready been analyzed
            if processed_faces and processed_face_landmarks:
                # Get a copy of the global landmarks and current landmark.
                _all_landmarks = copy.deepcopy(processed_face_landmarks)
                _all_landmarks.extend([current_face_all_landmarks])

                # Get a copy of the global faces and current face.
                _processed_faces = copy.deepcopy(processed_faces)
                _processed_faces.extend([current_face])

                print(_all_landmarks)
                # Average the landmarks together.
                average_landmarks = np.mean(_all_landmarks, 
                                            axis=0).astype(int).tolist()

                # Morph-align the faces to the averaged landmarks.
                morph_aligned_faces = []

                for face, landmarks in zip(_processed_faces, _all_landmarks):
                    morphed_face = \
                        morph_align_face(source_face=face,
                                        source_face_landmarks=landmarks,
                                        target_face_landmarks=average_landmarks,
                                        triangulation_indexes=None)
                    
                    morph_aligned_faces.append(morphed_face)

                # Create an average face image for this dataset.
                average_face = get_average_face(morph_aligned_faces)

                # Finally, Display it!
                cv2.imshow("Display", average_face)
                cv2.waitKey(30)

            # This is the first run
            else:
                cv2.imshow("Display", current_face)
                cv2.waitKey(30)

            # Track everything!
            with memory_lock:
                logging.info("Adding face to memory!!!")

                # Add the current face details.
                processed_faces.insert(0, current_face)
                processed_face_landmarks.insert(0, current_face_all_landmarks)
                recent_embeddings.insert(0, current_face_embedding)

                # Make sure this list doesn't get too long!
                processed_faces = processed_faces[:face_memory]
                processed_face_landmarks = processed_face_landmarks[:face_memory]
                recent_embeddings = recent_embeddings[:embedding_memory]




            # # Average the landmarks together.
            # average_landmarks = np.mean(processed_face_landmarks, 
            #                             axis=0).astype(int).tolist()

            # # Morph-align the faces to the averaged landmarks.
            # morph_aligned_faces = []

            # for face, landmarks in zip(processed_faces, processed_face_landmarks):
            #     morphed_face = \
            #         morph_align_face(source_face=face,
            #                         source_face_landmarks=landmarks,
            #                         target_face_landmarks=average_landmarks,
            #                         triangulation_indexes=None)
                
            #     morph_aligned_faces.append(morphed_face)

            # # Create an average face image for this dataset.
            # average_face = get_average_face(morph_aligned_faces)

            # # Finally, Display it!
            # cv2.imshow("Display", average_face)
            # cv2.waitKey(30)


    except Exception as e:
        logging.warning("TOP LEVEL ERROR! %s", exc_info=True)
        return False


if __name__ == "__main__":
    # Get environment in SH mode
    os.environ["DISPLAY"] = ":0"



    """
    Copy from Mona Lisa Override!
    """


    # Load the YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    while True:
        try:
            collect_faces(camera_type=config["camera_type"],
                        blur_threshold=config["blur_threshold"],
                        face_memory=config["face_memory"],
                        embedding_memory=config["embedding_memory"],
                        tolerance=config["tolerance"],
                        min_width=config["min_width"],
                        margin_fraction=config["margin_fraction"],
                        height_output=config["height_output"],
                        width_output=config["width_output"],
                        l=config["l"],
                        r=config["r"],
                        t=config["t"],
                        b=config["b"],
                        triangulation_indexes=config["triangulation_indexes"],
                        check_centering=config["check_centering"],
                        check_forward=config["check_forward"],
                        debug_images=config["debug_images"])
        except KeyboardInterrupt:
            print("Keyboard Interrupt!!! Exiting!!!")
            break
        except Exception:
            pass
