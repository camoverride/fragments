from typing import List, Tuple
import numpy as np
import cv2
import mediapipe as mp
import random
import numpy as np
import yaml
import logging



# Set up basic logging
logging.basicConfig(
    level=logging.INFO,  # Minimum level to log
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize image capture.
cap = cv2.VideoCapture(0)
cv2.waitKey(1000) # pause so first frame isn't dark

# Initialize detection.
face_detection = \
    mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                            max_num_faces=1,
                                            refine_landmarks=True,
                                            min_detection_confidence=0.5)

# Load the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if config["camera_type"] == "picam":
    from picamera2 import Picamera2 # type: ignore

    # Initialize the picamera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        # main={"size": (1920, 1080), "format": "RGB888"},
        # lores={"size": (640, 480)},
        # main={"format": "RGB888"},
        main={"size": (3280, 2464), "format": "RGB888"},
        display="main"))
    
    picam2.start()


def get_face_landmarks(image: np.ndarray) -> List[List[int]]:
    """
    Returns facial landmarks with guaranteed valid coordinates within image bounds.
    
    Args:
        image: BGR image (OpenCV format) with exactly one face
        
    Returns:
        List of [x,y] coordinates, all guaranteed within image dimensions
    """
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with face mesh
    results = face_mesh.process(rgb_image)
    
    # Initialize with empty list
    facial_landmarks = []
    
    if results.multi_face_landmarks:
        height, width = image.shape[:2]
        
        # Safely convert each landmark
        for landmark in results.multi_face_landmarks[0].landmark:
            # Clamp normalized coordinates to [0,1] range first
            x_norm = max(0.0, min(1.0, landmark.x))
            y_norm = max(0.0, min(1.0, landmark.y))
            
            # Convert to pixels with rounding and bounds checking
            x = int(round(x_norm * (width - 1)))
            y = int(round(y_norm * (height - 1)))
            
            # Final safety check (shouldn't be needed but protects against math errors)
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            facial_landmarks.append([x, y])
    
    return facial_landmarks


def get_additional_landmarks(image_height : int,
                             image_width : int) -> List[List[int]]:
    """
    Adds additional landmarks to an image. These landmarks are
    around the edges of the image. This helps with morphing so
    that the entire image can be tiled with delauney triangles.

    Parameters
    ----------
    image_height : int
        The height of the image in pixels.
    image_width : int
        The width of the image in pixels.

    Returns
    -------
    List[List[int]]
        A list of lists, where each sub-list is an additional landmark.
    """
    # subdiv.insert() cannot handle max values for edges, so add a small offset.
    # TODO: why???
    offset = 0.0001

    # New coordinates to add to the landmarks
    new_coords = [
        # Corners of the image
        [0, 0],
        [image_width - offset, 0],
        [image_width - offset, image_height - offset],
        [0, image_height - offset],

        # Middle of the top, bottom, left, right sides
        [(image_width - offset) / 2, 0],
        [(image_width - offset) / 2, image_height - offset],
        [0, (image_height - offset) / 2],
        [image_width - offset, image_height / 2],
    ]

    int_coords = [(int(x), int(y)) for (x, y) in new_coords]

    return int_coords


def _select_face_by_overlap(image, results, bb) -> int:
    """
    Selects face mesh with maximum overlap with bounding box.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    results : mediapipe face mesh results
    bb : mediapipe.framework.formats.location_data.RelativeBoundingBox
        MediaPipe's bounding box format
    
    Returns
    -------
    int
        Index of best matching face
    """
    # Extract bounding box coordinates
    bb_x = bb.xmin * image.shape[1]
    bb_y = bb.ymin * image.shape[0]
    bb_w = bb.width * image.shape[1]
    bb_h = bb.height * image.shape[0]
    bb_area = bb_w * bb_h
    
    max_overlap = -1
    best_idx = 0
    
    for i, face_landmarks in enumerate(results.multi_face_landmarks):
        # Get face mesh bounding box
        xs = [lm.x * image.shape[1] for lm in face_landmarks.landmark]
        ys = [lm.y * image.shape[0] for lm in face_landmarks.landmark]
        face_x1, face_x2 = min(xs), max(xs)
        face_y1, face_y2 = min(ys), max(ys)
        
        # Calculate intersection
        x_overlap = max(0, min(bb_x + bb_w, face_x2) - max(bb_x, face_x1))
        y_overlap = max(0, min(bb_y + bb_h, face_y2) - max(bb_y, face_y1))
        overlap_area = x_overlap * y_overlap
        
        # Normalize by target BB area
        overlap_ratio = overlap_area / bb_area
        
        if overlap_ratio > max_overlap:
            max_overlap = overlap_ratio
            best_idx = i
    
    return best_idx


def align_eyes_horizontally(image : np.ndarray,
                            bb : tuple) -> tuple:
    """
    Rotate an image so that the eyes are positioned horizontally.
    This makes it much more straightforward for subsequent cropping.
    This function also returns all the landmarks, appropriately
    rotated.

    Parameters
    ----------
    image : np.ndarray
        An image that should contain a face.
    bb : tuple
        A bounding box containing the face we care about.
    
    Returns
    -------
    tuple
        A tuple of two items.
        The first is a np.ndarray rotated image.
        The second is the rotated landmarks from mediapipe.
        TODO: what type exactly are the mediapipe landmarks?
    """
    # Read the image.
    image_rgb = image #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        raise ValueError("Alignment phase: no face detected.")
    
    # Find face mesh with maximum overlap with bb
    best_face_idx = _select_face_by_overlap(image_rgb, results, bb)
    landmarks = results.multi_face_landmarks[best_face_idx].landmark
    


        # Convert landmarks to image coordinates
    h, w = image.shape[:2]
    landmark_points = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    
    # Calculate rotation angle
    left_eye = landmark_points[33]
    right_eye = landmark_points[263]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Create rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Rotate landmarks (add homogeneous coordinate)
    homogeneous_landmarks = np.column_stack([landmark_points, np.ones(len(landmark_points))])
    rotated_points = (rotation_matrix @ homogeneous_landmarks.T).T
    
    # Convert back to MediaPipe landmark format
    rotated_landmarks = []
    for i, (x, y) in enumerate(rotated_points):
        landmark = results.multi_face_landmarks[best_face_idx].landmark[i]
        rotated_landmark = type(landmark)()
        rotated_landmark.x = x / w
        rotated_landmark.y = y / h
        rotated_landmark.z = landmark.z  # Z remains unchanged in 2D rotation
        rotated_landmarks.append(rotated_landmark)
    
    return rotated_image, rotated_landmarks


def crop_image_based_on_eyes(image : np.ndarray,
                             landmarks : np.ndarray,
                             l : float,
                             r : float,
                             t : float,
                             b : float) -> np.ndarray:
    """
    Crops the image based on the relative position of the eyes.
    K is the distance between pupils. Starting from the pupils,
    the value K is used to calculate the margins. For instance,
    if K is 200 pixels, then a left margin of 1.5 will mean that
    the left margin is 1.5 * 200 = 300 pixels, starting from the
    eyeball on the left side of the image.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face that has been rotated so that the
        eyes are on a horizontal plain.
    landmarks : TODO: what is the mediapipe type exactly?
        Landmarks returned by mediapipe and rotated.
    l : float
        The left margin, calculated as a fraction of K.
    r : float
        The right margin, calculated as a fraction of K.
    t : float
        The top margin, calculated as a fraction of K.
    b : float
        The bottom margin, calculated as a fraction of K.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    # Iris landmarks (468-472 for right eye, 473-477 for left eye)
    try:
        left_iris_landmarks = [landmarks[468],
                               landmarks[469],
                               landmarks[470],
                               landmarks[471],
                               landmarks[472]]
        right_iris_landmarks = [landmarks[473],
                                landmarks[474],
                                landmarks[475],
                                landmarks[476],
                                landmarks[477]]
        
    # If this exception gets hit too much, set refine_landmarks=True
    except IndexError:
        raise ValueError("Iris landmarks not available.")
    
    # Calculate the center of the left and right iris.
    left_iris_center = np.mean([(int(lm.x * image.shape[1]),
                                 int(lm.y * image.shape[0])) \
                                    for lm in left_iris_landmarks],
                                 axis=0)
    right_iris_center = np.mean([(int(lm.x * image.shape[1]),
                                  int(lm.y * image.shape[0])) \
                                    for lm in right_iris_landmarks],
                                  axis=0)
    
    # Calculate the distance between the eyes.
    K = right_iris_center[0] - left_iris_center[0]
    
    # Calculate crop coordinates
    x1 = int(left_iris_center[0] - l * K)
    x2 = int(right_iris_center[0] + r * K)
    y1 = int(left_iris_center[1] - t * K)
    y2 = int(left_iris_center[1] + b * K)
    
    # Create a blank image (black) of the desired crop size
    crop_height = y2 - y1
    crop_width = x2 - x1
    cropped_image = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    
    # Calculate the valid region of the original image to copy
    src_x1 = max(x1, 0)
    src_x2 = min(x2, image.shape[1])
    src_y1 = max(y1, 0)
    src_y2 = min(y2, image.shape[0])
    
    # Calculate the destination region in the blank image
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # Copy the valid region from the original image to the blank image
    if src_x2 > src_x1 and src_y2 > src_y1:
        cropped_image[dst_y1:dst_y2, dst_x1:dst_x2] = \
            image[src_y1:src_y2, src_x1:src_x2]
    
    return cropped_image


def crop_align_image_based_on_eyes(image : np.ndarray,
                                   bb : list,
                                   l : float,
                                   r : float,
                                   t : float,
                                   b : float) -> np.ndarray:
    """
    Wrapper function for the rotate and crop functions.
    NOTE: these should eventually be combined.

    Parameters
    ----------
    image : np.ndarray
        An image containing a face.
    l : float
        The left margin, calculated as a fraction of K.
    r : float
        The right margin, calculated as a fraction of K.
    t : float
        The top margin, calculated as a fraction of K.
    b : float
        The bottom margin, calculated as a fraction of K.

    Returns
    -------
    np.ndarray
        The cropped image.
    """
    rotated_image, rotated_landmarks = align_eyes_horizontally(image, bb)
    cropped_image = crop_image_based_on_eyes(image=rotated_image,
                                             landmarks=rotated_landmarks,
                                             l=l,
                                             r=r,
                                             t=t,
                                             b=b)

    return cropped_image


def is_face_looking_forward(face_landmarks: List[int],
                            image_height : int,
                            image_width : int
                            ) -> bool:
    """
    Analyzes the landmarks from a face and returns True if the
    face is looking forward (like a passport photo), otherwise
    it returns False.

    Parameters
    ----------
    face_landmarks : List[int]
        A list of all the landmarks returned from mediapipe's face mesh.
    image_height : int
        The height of the image in pixels.
    image_width : int
        The width of the image in pixelv.
    
    Returns
    -------
    bool
        True if the face is looking forward.
        False if the face is looking in another direction.
    """
    # Collect the 2D and 3D landmarks.
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
            # if idx == 1:
            #     nose_2d = (lm.x * image_width,lm.y * image_height)
            #     nose_3d = (lm.x * image_width,lm.y * image_height,lm.z * 3000)
            x, y = int(lm.x * image_width),int(lm.y * image_height)

            face_2d.append([x,y])
            face_3d.append(([x,y,lm.z]))

    # Get 2D coordinates
    face_2d = np.array(face_2d, dtype=np.float64)

    # Get 3D coordinates
    face_3d = np.array(face_3d,dtype=np.float64)

    # Calculate the orientation of the face.
    focal_length = 1 * image_width
    cam_matrix = np.array([[focal_length,0,image_height/2],
                        [0,focal_length,image_width/2],
                        [0,0,1]])
    distortion_matrix = np.zeros((4,1),dtype=np.float64)

    _, rotation_vec, _ = \
        cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

    # Get the rotational vector of the face.
    rmat, _ = cv2.Rodrigues(rotation_vec)

    angles, _, _ ,_, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Check which way the face is oriented.
    if y < -6: # Looking Left
        looking_forward = False

    elif y > 6: # Looking Right
        looking_forward = False

    elif x < -6: # Looking Down
        looking_forward = False

    elif x > 10: # Looking Up
        looking_forward = False

    else: # Looking Forward
        looking_forward = True

    return looking_forward


def get_delauney_triangles(image_width : int,
                           image_height : int,
                           landmark_coordinates : List[List[int]]) \
                            -> np.ndarray:
    """
    Accepts an image along with landmark coordinates, which are a
    list of tuples. The landmarks can be just the face landmarks or
    all the landmarks, which will include points along the edge of
    the image, not just the face.

    Returns a list of lists, where every element of the list is
    6 long and contains the three coordinate pairs of every
    delauney triangle:
        [ [[x1, x2, y1, y2, z1, z2], ... ]

    NOTE: there will be more delauney triangles than points.

    Parameters
    ----------
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.
    landmark_coordinates : List[List[int]]
        A list of all the landmark coordiantes.
    
    Returns
    -------
    np.ndarray
        A NumPy array of shape (N, 6), where each row contains the 
        coordinates of a Delaunay triangle: [x1, y1, x2, y2, x3, y3].
    """
    # Rectangle to be used with Subdiv2D
    rect = (0, 0, image_width, image_height)

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in landmark_coordinates:
        subdiv.insert(p)

    return subdiv.getTriangleList()


def get_triangulation_indexes_for_landmarks(landmarks : List[List[int]],
                                            image_height : int,
                                            image_width : int) -> List:
    """
    Connect together all the landmarks into delauney triangles that
    span the image.

    Parameters
    ----------
    landmarks : list
        A list of coordinate pairs for every landmark.
    image_height : int
        The height of the image.
    image_width : int
        The width of the image. 
    
    Returns
    -------
    List[List[int]]
        The triangulation indexes. A list containing triplets:
        [[458, 274, 459], [465, 417, 464], ... ]
    """
    # Get the delauney triangles based off the landmarks.
    delauney_triangles = get_delauney_triangles(image_width,
                                                image_height,
                                                landmarks)

    # Convert these points into indexes.
    enumerated_rows = {}
    for index, row in enumerate(landmarks):
        enumerated_rows[str(list(row))] = index

    triangulation_indexes = []

    for x1, x2, y1, y2, z1, z2 in delauney_triangles:
        x = str(list([int(x1), int(x2)]))
        y = str(list([int(y1), int(y2)]))
        z = str(list([int(z1), int(z2)]))

        index_x = enumerated_rows[x]
        index_y = enumerated_rows[y]
        index_z = enumerated_rows[z]

        triangulation_indexes.append([index_x, index_y, index_z])

    return triangulation_indexes

def applyAffineTransform(src: np.ndarray, 
                         srcTri: List[List[int]], 
                         dstTri: List[List[int]], 
                         size: Tuple[int, int]) -> np.ndarray:
    """
    Applies an affine transformation to an image region based on 
    corresponding triangle vertices.

    Given a source image region and a pair of corresponding triangles (one 
    in the source image and one in the destination image), this function 
    computes the affine transformation matrix and applies it to warp the 
    source patch onto the destination.

    Parameters
    ----------
    src : np.ndarray
        The source image patch that will be transformed.
    
    srcTri : List[List[int]]
        A list of three coordinate pairs representing the triangle in the 
        source image. Format: [[x1, y1], [x2, y2], [x3, y3]].
    
    dstTri : List[List[int]]
        A list of three coordinate pairs representing the corresponding 
        triangle in the destination image. Format: [[x1, y1], [x2, y2], [x3, y3]].
    
    size : Tuple[int, int]
        The dimensions (width, height) of the output image patch.

    Returns
    -------
    np.ndarray
        The warped image patch with the same dimensions as `size`.

    Notes
    -----
    - The transformation is computed using `cv2.getAffineTransform()`, which 
      finds a 2x3 matrix mapping `srcTri` to `dstTri`.
    - The transformation is applied using `cv2.warpAffine()` with bilinear 
      interpolation and border reflection to handle edge pixels.
    """
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src,
                         warpMat,
                         (size[0], size[1]),
                         None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def morph_align_face(source_face : np.ndarray,
                     source_face_landmarks : List[list[int]],
                     target_face_landmarks : List[List[int]],
                     triangulation_indexes: List) -> np.ndarray:
    """
    Accepts two images of the same dimensions containing faces.
    The features of the `source_face` are morphed so that they
    align with the landmarks of the `target_face`. Returns a morphed
    version of the `source_face`.

    This is done by extracting the landmarks from both faces and
    performing many affine transformations to change portions of
    the `source_face` to align with the `target_face`. These affine
    transformations assume a triangulation index, which is a division
    of all the landmarks into triangles, which are easy to mutate.

    Parameters
    ----------
    source_face : np.ndarray
        The face that will be morphed, having its features changed.
        Must be the same dimensions as `target_face`.

    target_face_all_landmarks : List[List[int]]
        The "skeleton" landmarks onto which the source face will be mutated.
        Must be the same dimensions as `source_face`.

    triangulation_indexes : list
        The list of triangles that span the entire image, used for
        morphing. Can be pre-computed, as it is not related to a
        specific face. It must have the same dimensions as `source_face`
        and `target_face`.

    Returns
    -------
    np.ndarray
        The "skin" from `source_face` morphed onto the landarks
        ("skeleton") of `target_face`.
    """
    # Get the triangulation indexes for the target face.
    # NOTE: the image height/width is the same in all images, so it's taken
    # # from the source, even though the landmarks are from the target.
    if not triangulation_indexes:
        triangulation_indexes = get_triangulation_indexes_for_landmarks(
                                            image_height=source_face.shape[0],
                                            image_width=source_face.shape[1],
                                            landmarks=target_face_landmarks)

    else:
        # Load the triangulation indexes.
        # TODO: implement this, save file to git
        pass

    # Leave space for final output
    morphed_face = np.zeros(source_face.shape, dtype=source_face.dtype)

    # Main event loop to morph triangles.
    for line in triangulation_indexes:
        # ID's of the triangulation points
        x = line[0]
        y = line[1]
        z = line[2]

        # Coordinate pairs
        t1 = [target_face_landmarks[x],
                target_face_landmarks[y],
                target_face_landmarks[z]]
        t2 = [source_face_landmarks[x],
                source_face_landmarks[y],
                source_face_landmarks[z]]

        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t1]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []

        for i in range(0, 3):
            tRect.append(((t1[i][0] - r[0]), (t1[i][1] - r[1])))
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Get the mask by filling triangles
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

        # Apply to small rectangular patches
        img2Rect = source_face[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warped_image = applyAffineTransform(src=img2Rect,
                                            srcTri=t2Rect,
                                            dstTri=tRect,
                                            size=size)

        # Copy triangular region of the rectangular patch to the output image
        morphed_face[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
            morphed_face[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * \
                    (1 - mask) + warped_image * mask

    return morphed_face


def get_average_landmarks(target_landmarks_paths : list) -> List[List[int]]:
    """
    Accepts a list of image paths and extracts the landmarks from each
    image, averaging them together.

    Parameters
    ----------
    target_landmarks_paths : list
        A list of the paths to every face image. This can simply
        be a list of one path, or some subset of the dataset.

    Returns
    -------
    List[List[int]]
        The averaged landmarks.
    """
    # Collect all the landmarks here.
    all_landmarks = []

    # Read all the images.
    for face_path in target_landmarks_paths:
        face_image = cv2.imread(face_path)

        # Get all the landmarks.
        landmarks = get_face_landmarks(face_image)
        if landmarks is not None:
            all_landmarks.append(np.array(landmarks, dtype=np.float32))

    # Compute the average of all landmarks.
    average_landmarks = np.mean(all_landmarks, axis=0).astype(int)

    # Convert to a list
    average_landmarks_list = average_landmarks.tolist()

    return average_landmarks_list


def create_composite_image(image_list : List[np.ndarray],
                           num_squares_height : int) -> np.ndarray :
    """
    Accepts a list of images and desired number of squares
    (along the vertical margin) and creates a composite image
    from them.

    Parameters
    ----------
    image_list : List[np.ndarray]
        A list of images encoded as numpy arrays.
    num_squares_height : int,
        The number of squares to tile the vertical of the image.
    
    Returns
    -------
    np.ndarray
        A composite image encoded as a numpy array.
    """
    # Get the height/width from a random image.
    image_height = image_list[0].shape[0]
    image_width = image_list[0].shape[1]

    # Check that the images all have the same shape
    if len(set((img.shape for img in image_list))) != 1:
        raise ValueError("All images must have the same dimensions.")

    # Set the image dimension info.
    crop_width = image_width - (image_width % num_squares_height)
    crop_height = image_height - (image_height % num_squares_height)
    image_list = [img[:crop_height, :crop_width] for img in image_list]

    square_size = crop_height // num_squares_height
    num_squares_width = crop_width // square_size

    # Generate the individual squares.
    # TODO: this should also be a memmap
    squares = [[[] for _ in range(num_squares_width)] for _ in range(num_squares_height)]
    for img in image_list:
        for i in range(num_squares_height):
            for j in range(num_squares_width):
                top = i * square_size
                left = j * square_size
                square = img[top:top + square_size, left:left + square_size]
                squares[i][j].append(square)

    # Combine the squares into an image.
    composite_image = np.zeros_like(image_list[0][:crop_height, :crop_width])
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            selected_square = random.choice(squares[i][j])
            top = i * square_size
            left = j * square_size
            composite_image[top:top + square_size, left:left + square_size] = selected_square

    return composite_image


def create_composite_image(image_list: List[np.ndarray], num_squares_height: int) -> np.ndarray:
    """
    Accepts a list of images and desired number of squares
    (along the vertical margin) and creates a composite image
    from them. The composite image will have the same dimensions
    as the input images.

    Parameters
    ----------
    image_list : List[np.ndarray]
        A list of images encoded as numpy arrays.
    num_squares_height : int,
        The number of squares to tile the vertical of the image.
    
    Returns
    -------
    np.ndarray
        A composite image encoded as a numpy array with the same
        dimensions as the input images.
    """
    # Get the height/width from a random image.
    image_height = image_list[0].shape[0]
    image_width = image_list[0].shape[1]

    # Check that the images all have the same shape
    if len(set((img.shape for img in image_list))) != 1:
        raise ValueError("All images must have the same dimensions.")

    # Calculate the square size based on the height
    square_size = image_height // num_squares_height

    # Calculate the number of squares along the width
    num_squares_width = image_width // square_size

    # Initialize the composite image with the same dimensions as the input images
    composite_image = np.zeros_like(image_list[0])

    # Generate the individual squares.
    for i in range(num_squares_height):
        for j in range(num_squares_width):
            # Calculate the top-left corner of the square
            top = i * square_size
            left = j * square_size

            # Randomly select an image to take the square from
            selected_image = random.choice(image_list)

            # Extract the square from the selected image
            square = selected_image[top:top + square_size, left:left + square_size]

            # Place the square into the composite image
            composite_image[top:top + square_size, left:left + square_size] = square

    return composite_image













def get_faces_from_camera(camera_type : str,
                          debug : bool):
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
    # Give camera time to warm up.

    cv2.waitKey(30)
    if camera_type == "webcam":
        # Get a frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to capture image from webcam.")
    elif camera_type == "picam":
        frame = picam2.capture_array()

    if debug:
        cv2.namedWindow("Camera cap", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Camera cap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Camera cap", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    # Look for faces in the frame
    frame_data = face_detection.process(frame)

    # If there are no faces, return False.
    if not frame_data.detections:
        return False, False

    # If there are faces, return the frame and listf of bounding boxes.
    relative_bbs = [detection.location_data.relative_bounding_box \
            for detection in frame_data.detections]

    return frame, relative_bbs


def quantify_blur(image):
    """
    Determine if an image is blurry using the variance of the Laplacian.

    Parameters:
    - image: Input image as a NumPy array (np.ndarray).
    - threshold: A threshold value to determine blurriness. Lower values are more sensitive to blur.

    Returns:
    - True if the image is blurry, False otherwise.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:  # Check if the image is in color (3 channels)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # Already grayscale

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian
    laplacian_var = np.var(laplacian)

    return laplacian_var


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
            logging.info(f"Could not load image. Skipping.")
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



def is_face_well_positioned(relative_bb, K) -> bool:
    """
    Determines whether a face bounding box is not too close to the edge of the image.

    A bounding box is considered too close if any part of it is within K (a fraction of the image size) 
    from the left, right, top, or bottom edges of the image.

    Parameters
    ----------
    relative_bb : object
        A bounding box object with attributes `xmin`, `ymin`, `width`, and `height`
        (all normalized to the image dimensions, i.e., in the range [0, 1]).
    K : float, optional
        A fraction of the image dimensions defining the safe margin from edges (default is 1.0).

    Returns
    -------
    bool
        True if the bounding box is not too close to the edges, False otherwise.
    """
    # Extract normalized bounding box coordinates
    xmin = relative_bb.xmin
    ymin = relative_bb.ymin
    width = relative_bb.width
    height = relative_bb.height

    # Compute the bounding box edges
    xmax = xmin + width
    ymax = ymin + height

    # Define safe margins based on the entire image size
    left_margin = K
    right_margin = 1 - K
    top_margin = K
    bottom_margin = 1 - K

    # Check if the bounding box is too close to any edge
    if xmin < left_margin or xmax > right_margin:
        return False
    if ymin < top_margin or ymax > bottom_margin:
        return False
    
    return True
