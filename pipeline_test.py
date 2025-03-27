import cv2
import sys
import numpy as np
import face_recognition

from _image_processing_utils import *



TEST_IMAGE_PATH = "cam.jpg"
TEST_IMAGE_PATH = "people.jpg"
TEST_IMAGE_PATH = "office_people.jpg"
TEST_IMAGE_PATH = "less_people.jpg"



BLUR_THRESHOLD = 200
MIN_WIDTH = 250
MARGIN_FRACTION = 0.5
TOLERANCE = 0.6 #

L = 1.5
R = 1.5
T = 1.5
B = 3.5

WIDTH_OUTPUT = 600
HEIGHT_OUTPUT = 1000


# Initialize detection.
face_detection = \
    mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.9)



if __name__ == "__main__":
    # Get an image to analyze.
    frame = cv2.imread(TEST_IMAGE_PATH)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 480))

    # Process the frame.
    frame_data = face_detection.process(frame_rgb)

    print(frame_data.detections)

    # If there are no faces, return False.
    if not frame_data.detections:
        print("No faces detected!")
        sys.exit()

    # If there are faces, return the frame and listf of bounding boxes.
    bbs = [detection.location_data.relative_bounding_box \
            for detection in frame_data.detections]

    cv2.imshow("Frame from webcam", frame)
    # TODO: draw bb around frame
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    # If bbs exists, then faces have been detected.
    if not bbs:
        print("No faces detected!!!")
        sys.exit()

    # There might be multiple faces in the image.
    for bb in bbs:

        # Check if face is too far from the center. TODO: test this
        # if not is_face_centered(bb):
        #     print("Face is not centered!!!")
        #     sys.exit()

        # Get a simple-cropped face with tight margins for blur detection.
        simple_cropped_face_tight_margins = simple_crop_face(frame,
                                                             bb,
                                                             margin_fraction=0)
        cv2.imshow("Simple cropped face with no margins, for blur detection",
                   simple_cropped_face_tight_margins)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


        # Test if the image is too blurry.
        # if quantify_blur(simple_cropped_face_tight_margins) > BLUR_THRESHOLD:
        #     print("Face is blurry!!!")
        #     sys.exit()

        # if not is_face_wide_enough(image=frame,
        #                             bbox=bb,
        #                             min_width=MIN_WIDTH):
        #     print("Face is too small!")
        #     sys.exit()

        # Simple crop the image to the bounding box.
        # NOTE: this should add a small margin for face_recognition and face_mesh
        simple_cropped_face_with_margin = \
            simple_crop_face(image=frame,
                             bbox=bb,
                             margin_fraction=MARGIN_FRACTION)

        cv2.imshow("Simple cropped face with a margin, fore face_recotnition and face_mesh",
                    simple_cropped_face_with_margin)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        # Get the face_mesh landmarks for the face with a margin.
        face_mesh_results = face_mesh.process(simple_cropped_face_with_margin)
        
        # If there are no results, return False
        if not face_mesh_results:
            print("Could not get face_mesh landmarks!")
            sys.exit()
        if not face_mesh_results.multi_face_landmarks:
            print("Could not get face_mesh landmarks!")
            sys.exit()

        # Get the landmarks. We can safely assume only one face is in the image because
        # the margin is small.
        simple_cropped_face_landmarks = face_mesh_results.multi_face_landmarks[0]

        if not simple_cropped_face_landmarks:
            print("No landmarks detected!")
            sys.exit()

        # Check if it's looking forward.
        face_height, face_width, _ = simple_cropped_face_with_margin.shape
        face_forward = is_face_looking_forward(face_landmarks=simple_cropped_face_landmarks,
                                               image_height=face_height,
                                               image_width=face_width)

        # If it's not looking forward, return False
        if not face_forward:
            print("Face isn't looking forward")
            sys.exit()

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
        face_cropped_rotated = \
            crop_align_image_based_on_eyes(image=frame,
                                           bb=bb,
                                           l=L,
                                           r=R,
                                           t=T,
                                           b=B)
        # if not any(face_cropped_rotated):
        #     print("Face coult not be crop-rotated!")
        #     sys.exit()

        cv2.imshow("Face, cropped and pupil-aligned", face_cropped_rotated)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        # Resize the image
        face_cropped_rotated_resized = cv2.resize(face_cropped_rotated, 
                                                 (WIDTH_OUTPUT, HEIGHT_OUTPUT))
        
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
            sys.exit()

        # Check if the face has been recently recognized (embedded).
        RECENT_EMBEDDINGS = []
        results = face_recognition.compare_faces(RECENT_EMBEDDINGS,
                                                 new_face_embedding,
                                                 tolerance=TOLERANCE)

        if any(results):
            print("The face has been seen before!!!")
            sys.exit()

        # Get all the face landmarks for later morphing.
        face_landmarks = get_face_landmarks(face_cropped_rotated_resized)
        additional_landmarks = \
            get_additional_landmarks(image_height=face_cropped_rotated_resized.shape[0],
                                        image_width=face_cropped_rotated_resized.shape[1])
        all_landmarks = face_landmarks + additional_landmarks

        # Create a copy to draw on
        display_img = face_cropped_rotated_resized.copy()

        # Plot all landmarks (green dots)
        for landmark in all_landmarks:
            x = int(landmark[0])
            y = int(landmark[1])
            cv2.circle(display_img, (x, y), 2, (0, 255, 0), -1)  # Green dot

        # Display the result
        cv2.imshow("Landmarks", display_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
