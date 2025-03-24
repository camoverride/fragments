from collections import Counter
from datetime import datetime
import threading
import uuid
import yaml
import cv2
import face_recognition
import numpy as np
import os
import mediapipe as mp

from _image_processing_utils import simple_crop_face, quantify_blur, is_face_wide_enough, \
is_face_centered, get_faces_from_webcam, get_face_landmarks, get_additional_landmarks, morph_align_face, \
    create_composite_image, is_face_looking_forward, crop_align_image_based_on_eyes
from _database_utils import insert_embedding, read_face_list, \
query_recent_landmarks, get_recent_embeddings, insert_face_mapping


# Initialize MediaPipe FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, 
    min_detection_confidence=0.5
)

# Initialize detection.
face_detection = \
    mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)



def collect_faces(embeddings_db : str,
                  mappings_db : str,
                  min_num_faces_in_collage : int,
                  max_num_faces_in_collage : int,
                  num_frames_in_collage_animation : int,
                  blur_threshold : float,
                  face_memory : int,
                  tolerance : float,
                  min_width : int,
                  margin_fraction : float,
                  height_output : int,
                  width_output : int,
                  l : float,
                  r : float,
                  t : float,
                  b : float,
                  triangulation_indexes : list,
                  debug_images : bool) -> bool:
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
        - Save the processed face and record its path, embedding, and
            landmarks to a sqlite database.

    After the processing pipeline, a collage is formed. If any of the conditions
    in this process aren't met (such as not enough faces) the process returns
    False:
        - Query the embeddings database for some faces.
        - Average out the landmarks from these faces.
        - Align every face to the averaged "target" landmarks.
        - Collage these images into frames, and save the frames.
        - Save the processed collage and faces used to generate the
            collage to a sqlite database.
    
    Parameters
    ----------
    embeddings_db : str
        The path to a database of face embeddings with the schema:
        | ID | face_path | landmarks | embedding |
    mappings_db : str
        The path to a database of collages and the faces used to
        create them with the schema:
        | ID  | avg_face_path | animated_face_path | face_list |
        TODO: implement average face TODO: change to `collage_path`
    min_num_faces_in_collage : int
        The min number of faces that will be used to create a collage.
    max_num_faces_in_collage : int
        The max number of faces that will be used to create a collage.
    num_frames_in_collage_animation : int
        The number of frames that will be in the collage animation,
        ultimately saves as a .npz archive to images/collages.
    blur_threshold : float
        The maximum blurriness allowed in a face image. The face image
        should be cropped tightly to the face. The blurriness is calculated
        using the Laplacian. Faces that exceed this value will be skipped.
        NOTE: this must be manually tested with every camera and location
        setup (because of lighting conditions, etc.)
    face_memory : int
        When querying the embeddings databases for recent faces, how
        many faces should be considered? (using the most recent faces)
    tolerance : float
        The face recognition tolerance. 0.6 is standard in the industry.
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
    debug_images : bool
        If True, images of intermediate processing steps are displayed.
        This should always be False when used in production mode.

    Returns
    -------
    None or False
        False if one of the processing conditions wasn't met, such
        as a face not looking forward. Or False after the processing
        steps if there were insufficient images to form a new collage
        (or if a collage was already formed with existing images).

        If processing conditions are met, images are written to 
        `images/faces`.
        
        If collage conditions are met, .npz archices of animations
        are written to `images/collages`.
    """
    # Wrap everything in a giant try/except
    try:
        # Get an image from the webcam along with face bounding boxes.
        frame, bbs = get_faces_from_webcam(debug=debug_images)

        # If bbs exists, then faces have been detected.
        if not bbs:
            print("No faces detected!!!")
            return False

        # There might be multiple faces in the image.
        for bb in bbs:

            # Check if face is too far from the center. TODO: test this
            if not is_face_centered(bb):
                print("Face is not centered!!!")
                return False

            # Get a simple-cropped face with tight margins for blur detection.
            simple_cropped_face_tight_margins = simple_crop_face(frame, bb, margin_fraction=0)

            # Test if the image is too blurry.
            if quantify_blur(simple_cropped_face_tight_margins) > blur_threshold:
                print("Face is blurry!!!")
                return False

            if not is_face_wide_enough(image=frame,
                                    bbox=bb,
                                    min_width=min_width):
                print("Face is too small!")
                return False

            # Simple crop the image to the bounding box.
            # TODO: this will have to be large to accomodate the face mesh crop.
            # NOTE: this might capture multiple faces and introduce bugs.
            simple_cropped_face_with_margin = simple_crop_face(image=frame,
                                                            bbox=bb,
                                                            margin_fraction=margin_fraction)

            if debug_images:
                cv2.imshow("Simple cropped face with margin", simple_cropped_face_with_margin)
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
                print("No landmarks detected!")
                return False

            # Check if it's looking forward.
            face_height, face_width, _ = simple_cropped_face_with_margin.shape
            face_forward = is_face_looking_forward(face_landmarks=landmarks,
                                                image_height=face_height,
                                                image_width=face_width)

            # If it's not looking forward, return False
            if not face_forward:
                print("Face isn't looking forward")
                return False

            if debug_images:    
                # Annotate the frame.
                face_annotated = np.copy(simple_cropped_face_with_margin)
                cv2.putText(simple_cropped_face_with_margin, f"{face_forward}",
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
                    crop_align_image_based_on_eyes(image=simple_cropped_face_with_margin,
                                                l=l,
                                                r=r,
                                                t=t,
                                                b=b)
            except ValueError as e:
                print("Error face crop/rotate!")
                print(e)
                return False

            if debug_images:
                cv2.imshow("Face, cropped and pupil-aligned", face_cropped_rotated)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Resize the image
            face_cropped_rotated_resized = cv2.resize(face_cropped_rotated, 
                                                    (width_output, height_output))
            
            if debug_images:
                cv2.imshow("Face, resized", face_cropped_rotated_resized)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Check if the face has been seen before.
            # NOTE: use the face that has not been rotated and resized, as face_recognition
            # often fails with rotated images (unsure why!)
            simple_cropped_face_with_margin = cv2.cvtColor(simple_cropped_face_with_margin,
                                                           cv2.COLOR_BGR2RGB)
            new_face_embedding = \
                face_recognition.face_encodings(simple_cropped_face_with_margin)

            # If the face was able to be successfully embedded
            if not new_face_embedding:
                print("Face could not be embedded!!!")
                return False

            # Get the face embedding.
            # TODO: why index with [0] - probably because multiple embeddings might
            # be returned if the image has multiple faces...
            # NOTE: this is a potential source of bugs.
            new_face_embedding = new_face_embedding[0]

            # Get recent embeddings.
            recent_embeddings = get_recent_embeddings(db_path=embeddings_db,
                                                      num_embeddings=face_memory)

            # Check if the face has been recently recognized (embedded).
            results = face_recognition.compare_faces(recent_embeddings,
                                                     new_face_embedding,
                                                     tolerance=tolerance)

            if any(results):
                print("The face has been seen before!!!")
                return False

            # Save the processed face.
            embedding_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
                "_" + str(uuid.uuid4())[:8] + ".jpg"
            embedding_filepath = f"images/faces/{embedding_filename}"
            cv2.imwrite(embedding_filepath, face_cropped_rotated_resized)

            # Get all the face landmarks for later morphing.
            face_landmarks = get_face_landmarks(face_cropped_rotated_resized)
            additional_landmarks = \
                get_additional_landmarks(image_height=face_cropped_rotated_resized.shape[0],
                                        image_width=face_cropped_rotated_resized.shape[1])
            all_landmarks = face_landmarks + additional_landmarks

            # Insert the image path and embedding into the database.
            insert_embedding(db_path=embeddings_db,
                            face_path=embedding_filepath,
                            landmarks=all_landmarks,
                            embedding=new_face_embedding)

            ########################################################
            ################### Animation phase! ###################
            ########################################################

            # Collect the most recently processed faces from the database.
            face_paths, face_landmarks = \
                query_recent_landmarks(db_path=embeddings_db,
                                       n=max_num_faces_in_collage)

            if len(face_paths) < min_num_faces_in_collage:
                print("Not enough faces to collage!!!")
                return False

            # Read the current face_paths into a Counter where order doesn't matter but repeats do
            current_face_counter = Counter(face_paths)

            # Read the face_mapping database's `faces` column into a list of counters.
            face_lists = read_face_list(db_path=mappings_db)
            face_counters = [Counter(l) for l in face_lists]

            # Go through all the previously analyzed faces.
            for faces in face_counters:

                # Check if the faces have already been analyzed.
                if current_face_counter == faces:
                    print("Faces have already been averaged!!!")
                    return False

            # Get the average landmarks.
            average_landmarks = np.mean(face_landmarks, 
                                        axis=0).astype(int).tolist()

            # Morph-align all the faces to the averaged landmarks.
            morph_aligned_faces = []

            for source_face_path, source_face_landmarks in zip(face_paths, face_landmarks):
                source_face = cv2.imread(source_face_path)
                morphed_face = morph_align_face(source_face=source_face,
                                                source_face_landmarks=source_face_landmarks,
                                                target_face_landmarks=average_landmarks,
                                                triangulation_indexes=None)
                
                morph_aligned_faces.append(morphed_face)

            if morph_aligned_faces and debug_images:
                cv2.imshow("Morph-aligned face", morph_aligned_faces[0])
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Create an average face image for this dataset.
            average_face = np.mean(morph_aligned_faces, axis=0).astype(np.uint8)

            # Save the averaged image.
            averaged_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
                "_" + str(uuid.uuid4())[:8] + ".jpg"
            averaged_filepath = f"images/averages/{averaged_filename}"
            cv2.imwrite(averaged_filepath, average_face)

            # Create collages from these morphs.
            collage_frames = []

            for _ in range(num_frames_in_collage_animation):
                composite = create_composite_image(image_list=morph_aligned_faces,
                                                num_squares_height=90)

                collage_frames.append(composite)

            if collage_frames and debug_images:
                cv2.imshow("MAIN: Collaged face", morph_aligned_faces[0])
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            # Save the collage frames
            collage_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
                "_" + str(uuid.uuid4())[:8] + ".npz"
            collage_filepath = f"images/collages/{collage_filename}"
            np.savez(collage_filepath, *collage_frames)

            # Save to database.
            insert_face_mapping(db_path=mappings_db,
                                avg_face_path=averaged_filepath,
                                collage_filepath=collage_filepath,
                                face_list=face_paths)
            
    except Exception as e:
        print("TOP LEVEL ERROR!", e)
        return False



def run_animation_loop(animation_dir : str) -> None:
    """
    Looks for the most recent image (.jpg) or archive (.npz)
    in the `animation_dir`. If this has changed, load the file(s)
    to `current_play_files` as np.ndarray images. Otherwise continue
    playing files from `current_play_files`.

    Parameters
    ----------
    animation_dir : str
        The path to the directory containing a .jpg or .npz
        to be displayed.
    Returns
    -------
    None
        Displays image frames.
    """
    # The path to the current file.
    current_play_file_path = None

    # The file(s) unpacked into np.ndarray images.
    current_play_files = []

    while True:
        # Get paths to all the files.
        display_file_paths = [os.path.join(root, file) for root, _, files 
                              in os.walk(animation_dir) for file in files 
                              if file.endswith(".npz")
                              or file.endswith(".jpg")]
        display_file_paths = sorted(display_file_paths, key=os.path.getmtime)

        if not display_file_paths:
            print("No files to display!")

        else:
            # Check if the most recent video has changed.
            # Note: `None == [] # False`
            if current_play_file_path != display_file_paths[-1]:
                current_play_file_path = display_file_paths[-1]

                # If we're playing an animation, unpack them.
                if current_play_file_path.endswith(".npz"):
                    with np.load(current_play_file_path) as data:
                        current_play_files = [data[key] for key in data]

                # if we're playing a jpg, use it.
                elif current_play_file_path.endswith(".jpg"):
                    # This should only ever contain 1 image.
                    current_play_files = [cv2.imread(current_play_file_path)]

        # If files have been stored here, play them.
        if current_play_files:
            for image in current_play_files:
                cv2.imshow("Collage or Average", image)
                cv2.waitKey(100)
                

if __name__ == "__main__":

    # Load the YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    def collect_faces_loop():
        while True:
            collect_faces(embeddings_db=config["embeddings_db"],
                          mappings_db=config["mappings_db"],
                          min_num_faces_in_collage=config["min_num_faces_in_collage"],
                          max_num_faces_in_collage=config["max_num_faces_in_collage"],
                          num_frames_in_collage_animation=config["num_frames_in_collage_animation"],
                          blur_threshold=config["blur_threshold"],
                          face_memory=config["face_memory"],
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
                          debug_images=config["debug_images"])


    # Create thread for collecting faces.
    collect_faces_thread = threading.Thread(target=collect_faces_loop)
    collect_faces_thread.start()

    # This will continue forever.
    run_animation_loop(config["animation_dir"])
