from collections import Counter
from datetime import datetime
import threading
import uuid
import yaml
import cv2
import face_recognition
import numpy as np
import os

from pipeline_utils import get_faces_from_webcam, face_processing_pipeline, \
is_face_centered
from database_utils import insert_embedding, read_face_list, \
query_recent_landmarks, get_recent_embeddings
from utils_from_ff import get_face_landmarks, get_additional_landmarks, morph_align_face, \
    create_composite_image
from image_processing_utils import simple_crop_face, quantify_blur



def collect_faces(embeddings_db : str,
                  mappings_db : str,
                  blur_threshold : float,
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
                  debug_images : bool):
    """

    """
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

        simple_cropped_face = simple_crop_face(frame, bb, margin_fraction=0)

        # TODO: must test this threshold with every new camera!
        if quantify_blur(simple_cropped_face) > blur_threshold:
            print("-- Face is blurry!!!")
            return False

        # Try/except because sometimes there are processing errors.
        try:
            face = face_processing_pipeline(image=frame,
                                            bb=bb,
                                            min_width=min_width,
                                            margin_fraction=margin_fraction,
                                            height_output=height_output,
                                            width_output=width_output,
                                            l=l,
                                            r=r,
                                            t=t,
                                            b=b,
                                            debug=debug_images)

        except ValueError:
            print("There was an error in the pipeline!!!")
            return False

        if face is False:
            print("The face could not be processed!")
            return False
        
        if debug_images:
            cv2.imshow("Processed face", face)
            cv2.waitKey(5000)

        # Check if the face has been seen before.
        new_face_embedding = face_recognition.face_encodings(face)

        # If the face was able to be successfully embedded
        if not new_face_embedding:
            print("Face could not be embedded!!!")
            return False

        # Get the face embedding.
        new_face_embedding = new_face_embedding[0] # TODO: why index with [0]?

        # Get recent embeddings.
        recent_embeddings = get_recent_embeddings(db_path="face_embeddings.db",
                                                  num_embeddings=6)

        # Check if the face has been recently recognized (embedded).
        results = face_recognition.compare_faces(recent_embeddings,
                                                 new_face_embedding,
                                                 tolerance=tolerance)

        if any(results):
            print("The face has been seen before!!!")
            return False

        # Save the processed face
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
            "_" + str(uuid.uuid4())[:8] + ".jpg"
        filepath = f"images/faces/{filename}"
        cv2.imwrite(filepath, face)

        # Get all the face landmarks for later morphing
        face_landmarks = get_face_landmarks(face)
        additional_landmarks = \
            get_additional_landmarks(image_height=face.shape[0],
                                     image_width=face.shape[1])
        all_landmarks = face_landmarks + additional_landmarks

        # Insert the image path and embedding into the database.
        insert_embedding(db_path=embeddings_db,
                         face_path=filepath,
                         landmarks=all_landmarks,
                         embedding=new_face_embedding)


        #### Animation phase! #####

        # Collect the 10 most recently processed faces from the database.
        face_paths, face_landmarks = \
            query_recent_landmarks(db_path="face_embeddings.db",
                                   n=2)

        if len(face_paths) < 2:
            print("Not enough faces to collage!!!")
            return False

        # Check if the faces have already been averaged.
        faces_already_averaged = False

        # Read the current face_paths into a Counter where order doesn't matter but repeats do
        current_face_counter = Counter(face_paths)

        # Read the face_mapping database's `faces` column into a list of counters.
        face_lists = read_face_list(db_path=mappings_db)
        face_counters = [Counter(l) for l in face_lists]

        # Go through all the previously analyzed faces.
        for faces in face_counters:

            # Check if the faces have already been analyzed.
            if current_face_counter == faces:
                faces_already_averaged = True

        # If the faces haven't been analyzed yet, proceed.
        if faces_already_averaged == True:
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
            cv2.waitKey(5000)

        # Create collages from these morphs.
        collage_frames = []

        for _ in range(50):
            composite = create_composite_image(image_list=morph_aligned_faces,
                                               num_squares_height=90)

            collage_frames.append(composite)

        if collage_frames and debug_images:
            cv2.imshow("Collaged face", morph_aligned_faces[0])
            cv2.waitKey(5000)

        # Save the frames
        collage_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
            "_" + str(uuid.uuid4())[:8] + ".jpg"
        filepath = f"images/collages/{collage_filename}.npz"
        np.savez(filepath, *collage_frames)



def run_animation_loop() -> None:
    """
    
    """
    current_video_file_path = None
    frames = None

    while True:
        # Check if the most recent file has changed.
        video_file_paths = [os.path.join(root, file) for root, _, files 
                           in os.walk("images/collages") for file in files 
                           if file.endswith(".npz")]
        video_file_paths = sorted(video_file_paths, key=os.path.getmtime)

        # Check if the most recent video has changed. NOTE: `None != []`
        # This will create a small pause when reading the file.

        if current_video_file_path != video_file_paths[-1]:
            current_video_file_path = video_file_paths[-1]
            with np.load(current_video_file_path) as data:
                frames = [data[key] for key in data]

        if frames:
            for frame in frames:
                cv2.imshow("Collage", frame)
                cv2.waitKey(100)


                

if __name__ == "__main__":

    # print("starting animation loop")
    # run_animation_loop()

    print("Starting college faces")
    while True:
        collect_faces(embeddings_db="face_embeddings.db",
                      mappings_db="face_mappings.db",
                      blur_threshold=180, # definitely check on this!
                      tolerance=0.6, # 0.6 seems good
                      min_width=200, # probably increase above 500
                      margin_fraction=1.5, # this might need to be even wider
                      height_output=600, # depends on monitor
                      width_output=500, # depends on monitor
                      l=1.5, # 1.5
                      r=1.5, # 1.5
                      t=2.0, # 1.5
                      b=3.5, # 3.0
                      triangulation_indexes=None,
                      debug_images=False)
