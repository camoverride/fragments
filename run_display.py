from collections import Counter
from datetime import datetime
import os
import threading
import uuid
import yaml
import cv2
import face_recognition
import numpy as np

from pipeline_utils import get_faces_from_webcam, face_processing_pipeline, morph_faces, \
is_face_centered
from database_utils import query_embeddings_in_chunks, insert_embedding, read_face_list, \
insert_face_mapping, get_most_recent_row, query_recent_landmarks
from utils_from_ff import get_face_landmarks, get_additional_landmarks, morph_align_face, \
    create_composite_image



def collect_faces(embeddings_db : str,
                           mappings_db : str,
                           api_url : str,
                           chunk_size : int,
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
                           debug : bool):
    while True:
        # Get an image from the webcam along with face bounding boxes.
        frame, bbs = get_faces_from_webcam(debug=debug)

        # If bbs exists, then faces have been detected.
        if bbs:
            # There might be multiple faces in the image.
            # NOTE: only go through at most 2 faces.




            for bb in bbs[:2]:
                # Check if face is near the center of the image.
                if is_face_centered(bb):
                    print("CENTERED")
                
                
                # Check for blur using Laplacian Variance




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
                                                    debug=debug)

                except ValueError:
                    print("There was an error in the pipeline!")
                    face = False

                # If a face was returned by the pipeline
                if face is not False:

                    # Check if the face has been seen before.
                    new_face_embedding = face_recognition.face_encodings(face)

                    # If the face was able to be successfully embedded
                    if new_face_embedding:
                        # Get the face embedding.
                        # TODO: why index with [0]?
                        new_face_embedding = new_face_embedding[0]

                        face_seen_before = False

                        # Check if the face has been seen before.
                        for embeddings_chunk in \
                            query_embeddings_in_chunks(db_path=embeddings_db,
                                                       chunk_size=chunk_size):
                            results = face_recognition.compare_faces(embeddings_chunk,
                                                                     new_face_embedding,
                                                                     tolerance=tolerance)
                            if any(results):
                                face_seen_before = True

                        # If the face has not been seen before, proceed.
                        if face_seen_before == False:
                            print("New face detected!")

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

                        else:
                            print("face already detected")
                            pass

                    else:
                        print("No encoding detected! Skipping")




def create_animation(embeddings_db : str,
                     mappings_db: str):
    """
    
    """
    # Collect the 10 most recently processed faces from the database.
    face_paths, face_landmarks = \
        query_recent_landmarks(db_path="face_embeddings.db",
                               n=10)

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
    if faces_already_averaged == False:
        average_landmarks = np.mean(face_landmarks, axis=0).astype(int).tolist()

        print(face_paths)

        # Morph-align all the faces to the averaged landmarks.
        morph_aligned_faces = []

        for source_face_path, source_face_landmarks in zip(face_paths, face_landmarks):
            source_face = cv2.imread(source_face_path)
            morphed_face = morph_align_face(source_face=source_face,
                                            source_face_landmarks=source_face_landmarks,
                                            target_face_landmarks=average_landmarks,
                                            triangulation_indexes=None)
            
            morph_aligned_faces.append(morphed_face)
    
        # Create collages from these morphs.
        collage_frames = []

        for _ in range(50):
            composite = create_composite_image(image_list=morph_aligned_faces,
                                               num_squares_height=90)
            
            collage_frames.append(composite)
        
        for image in collage_frames:
            cv2.imshow("College", image)
            cv2.waitKey(100)
    
        cv2.destroyAllWindows()


        


if __name__ == "__main__":
    # create_animation(embeddings_db="face_embeddings.db",
    #                  mappings_db="face_mappings.db")
    

    collect_faces(embeddings_db="face_embeddings.db",
                           mappings_db="face_mappings.db",
                           api_url=None,
                           chunk_size=50,
                           tolerance=0.6,
                           min_width=200,
                           margin_fraction=1, # this might need to be even wider
                           height_output=600,
                           width_output=500,
                           l=1.5,
                           r=1.5,
                           t=2.5,
                           b=2.5,
                           triangulation_indexes=None,
                           debug=False)







#         # At least two faces must have been accumulated for these functions to work.
#         processed_faces = [os.path.join(root, file) for root, _, files 
#                            in os.walk("images/faces") for file in files 
#                            if file.endswith(".jpg")]
        
#         if len(processed_faces) >= 2:
#             # Find the two most recently processed faces.
#             processed_faces = sorted(processed_faces, key=os.path.getmtime)
#             face_1 = processed_faces[-1]
#             face_2 = processed_faces[-2]

#             # Check if the faces have already been averaged.
#             faces_already_averaged = False

#             # Read the current faces into a Counter where order doesn't matter but repeats do
#             current_face_counter = Counter([face_1, face_2])

#             # Read the face_mapping database's `faces` column into a list of counters.
#             face_lists = read_face_list(db_path=mappings_db)
#             face_counters = [Counter(l) for l in face_lists]

#             # Go through all the previously analyzed faces.
#             for faces in face_counters:

#                 # Check if the faces have already been analyzed.
#                 if current_face_counter == faces:
#                     faces_already_averaged = True

#             # If the faces haven't been analyzed yet, proceed.
#             if faces_already_averaged == False:

#                 # Average the most recent two faces together.
#                 averaged_face = morph_faces(face_1=cv2.imread(face_1),
#                                             face_2=cv2.imread(face_2),
#                                             triangulation_indexes_ = False,
#                                             debug = debug)
                
#                 # # Swap the faces.
#                 # print("SWAPPING")
#                 # swapped_face = swap_faces(source_image=averaged_face,
#                 #                         target_image=cv2.imread("images/passport.jpg"))

#                 # Save the averaged face.
#                 averaged_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
#                     "_" + str(uuid.uuid4())[:8] + ".jpg"

#                 averaged_filepath = f"images/averages/{averaged_filename}"

#                 cv2.imwrite(averaged_filepath, averaged_face)
                
#                 # Get the frames from the API.
#                 print("Animating!")
#                 animation_frames = \
#                     image_to_video_api(api_url=api_url,
#                                        image_path=averaged_filepath)
                
#                 # Save the frames as a .npz object
#                 animation_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + \
#                     "_" + str(uuid.uuid4())[:8] + ".npz"

#                 animation_filepath = f"images/animations/{animation_filename}"

#                 np.savez_compressed(animation_filepath, *animation_frames)
                
#                 # Write everything to the face_mappings database
#                 insert_face_mapping(db_path=mappings_db,
#                                     avg_face_path=averaged_filepath,
#                                     animated_face_path=animation_filepath,
#                                     face_list=[face_1, face_2])



# def run_animation_loop(mappings_db : str,
#                        main_display_id : str,
#                        left_display_id : str,
#                        right_display_id : str) -> None:

#     while True:
#         # Check that there are entries in the database. If not, it will pause here.
#         avg_face_path, animation_filepath, face_list = \
#             get_most_recent_row(db_path=mappings_db)


#         # Enter the main event loop for displaying frames once the db is populated.
#         while animation_filepath is not None:

#             # Track the filepath for changes.
#             prev_animation_filepath = None

#             # Query the database for the most recently processed images.
#             avg_face_path, animation_filepath, face_list = \
#                 get_most_recent_row(db_path=mappings_db)

#             # Check if there is a change.
#             if animation_filepath != prev_animation_filepath:

#                 # If the file has changed, load the new data.
#                 prev_animation_filepath = animation_filepath

#                 with np.load(animation_filepath) as data:
#                     frames = [data[key] for key in data]

#             for frame in frames:
#                 # TODO: specify which monitor these belong on!
#                 cv2.imshow(f"Animated Face", frame)

#                 cv2.imshow("Face 1", cv2.imread(face_list[0]))
#                 cv2.moveWindow("Face 1", 1000, 0)

#                 cv2.imshow("Face 2", cv2.imread(face_list[1]))
#                 cv2.moveWindow("Face 2", 500, 300)

#                 # Wait for key press and check if 'Esc' is pressed
#                 key = cv2.waitKey(40)

#                 if key == 27:  # 27 is the ASCII code for the 'Esc' key
#                     print("Exiting animation.")
#                     cv2.destroyAllWindows()  # Close the window before exiting
#                     return  # Exit the function and stop the animation
                


# if __name__ == "__main__":
#     # Load the config file
#     with open("config.yaml", "r") as file:
#         config = yaml.safe_load(file)

#     # Create the animations thread!
#     create_animations_thread = threading.Thread(
#     target=create_face_animations,  # The function to run
#     kwargs={
#         "embeddings_db": config["embeddings_db"],
#         "mappings_db": config["face_mappings_db"],
#         "api_url": config["api_url"],
#         "chunk_size": config["chunk_size"],
#         "tolerance": config["tolerance"],
#         "min_width": config["min_width"],
#         "margin_fraction": config["margin_fraction"],
#         "height_output": config["height_output"],
#         "width_output": config["width_output"],
#         "l": config["l"],
#         "r": config["r"],
#         "t": config["t"],
#         "b": config["b"],
#         "triangulation_indexes": config["triangulation_indexes"],
#         "debug": config["debug"]
#     })

#     # Start the thread!
#     create_animations_thread.start()


#     # NOTE: this one fails when run as a thread.
#     run_animation_loop(mappings_db=config["face_mappings_db"],
#                        main_display_id=config["main_display_id"],
#                        left_display_id=config["left_display_id"],
#                        right_display_id=config["right_display_id"])
