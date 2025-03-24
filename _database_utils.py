from typing import Generator, List
import sqlite3
import pickle
import numpy as np



def create_embedding_database(db_path: str) -> bool:
    """
    Creates a sqlite database with the schema:

    | ID | face_path | landmarks | embedding |

    - `face_path` is the path to a processed image of a face.
    - `landmarks` is a BLOB representing a list of all the landmarks (pairs of points) [x, y].
    - `embedding` is the embedding of the face, saved as a BLOB.

    Parameters
    ----------
    db_path : str
        The path to the database file.

    Returns
    -------
    bool
        True if the database was created successfully,
        otherwise False.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_path TEXT NOT NULL,
            landmarks BLOB NOT NULL,
            embedding BLOB NOT NULL
        )
        """)
        
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error creating embedding database: {e}")
        return False


def create_face_mapping_database(db_path : str) -> bool:
    """
    Creates a sqlite database with the following schema:

    | ID  | avg_face_path | collage_filepath | face_list |

    - `face_list` is a list of paths to every face that was used
    to create the average face and is saved as a BLOB.
    - `avg_face_path` is a face created by morphing and averaging
    all the faces in `face_list`.
    - `collage_filepath` is a path to the np.ndarray images
    that compose an animation, saved as a compressed numpy
    archive (.npz).

    Parameters
    ----------
    db_path : str
        The path to the database file.

    Returns
    -------
    bool
        True if the database was created successfully,
        otherwise False.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            avg_face_path TEXT NOT NULL,
            collage_filepath TEXT NOT NULL,
            face_list BLOB NOT NULL
        )
        """)
        
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error creating face mapping database: {e}")
        return False


def insert_embedding(db_path: str,
                     face_path: str,
                     landmarks: list,
                     embedding: np.ndarray) -> bool:
    """
    Inserts a face embedding into the face_embeddings database (`db_path`).

    Parameters
    ----------
    db_path : str
        The path to the database file.
    face_path : str
        The path to the processed image of the face.
    landmarks : list
        A list of landmarks (pairs of points) [x, y].
    embedding : np.ndarray
        The face embedding as a numpy array.

    Returns
    -------
    bool
        True if the insertion was successful, otherwise False.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Serialize landmarks to a BLOB using pickle
        landmarks_blob = pickle.dumps(landmarks)

        # Serialize the embedding to a BLOB
        embedding_blob = pickle.dumps(embedding)

        # Insert data into the database
        cursor.execute("""
        INSERT INTO face_embeddings (face_path, landmarks, embedding)
        VALUES (?, ?, ?)
        """, (face_path, landmarks_blob, embedding_blob))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error inserting embedding: {e}")
        return False


def insert_face_mapping(db_path : str,
                        avg_face_path : str,
                        collage_filepath : str,
                        face_list : list) -> bool:
    """
    Inserts a face mapping into the face_mappings database.

    Parameters
    ----------
    db_path : str
        The path to the database file.
    avg_face_path : str
        Path to the averaged face image.
    animated_face_path : str
        Path to the .npz file containing animation frames.
    face_list : list
        List of paths to the faces used in the averaging process.

    Returns
    -------
    bool
        True if the insertion was successful, otherwise False.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Serialize the face list to a BLOB
        face_list_blob = pickle.dumps(face_list)

        cursor.execute("""
        INSERT INTO face_mappings (avg_face_path, collage_filepath, face_list)
        VALUES (?, ?, ?)
        """, (avg_face_path, collage_filepath, face_list_blob))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error inserting face mapping: {e}")
        return False


def get_recent_embeddings(db_path : str,
                          num_embeddings : int):
    """

    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT embedding
    FROM face_embeddings
    ORDER BY ID DESC
    LIMIT ?
    """
    cursor.execute(query, (num_embeddings,))

    embeddings = cursor.fetchall()

    # Deserialize embeddings and process them
    embeddings = [pickle.loads(embedding[0]) for embedding in embeddings]

    conn.close()

    return embeddings



def read_face_list(db_path: str) -> List[tuple]:
    """
    Reads the database and returns a list of all the faces.

    Parameters
    ----------
    db_path : str
        The path to the database file.

    Returns
    -------
    list
        A list of tuples, where each tuple are the faces used
        to create a composite image.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT face_list FROM face_mappings")
    face_list_rows = cursor.fetchall()
    
    # Deserialize the BLOB data into tuples
    face_lists = [pickle.loads(row[0]) for row in face_list_rows]
    
    conn.close()
    
    return face_lists


def get_most_recent_row(db_path : str) -> tuple:
    """
    Queries the database to get the most recent row and returns
    the paths and deserialized face_list.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    tuple
        A tuple containing:
        - avg_face_path (str): The path to the average face image.
        - animated_face_path (str): The path to the animated face image.
        - face_list (list): The list of paths to faces used in the average.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to get the most recent row based on the highest ID
    cursor.execute("SELECT avg_face_path, animated_face_path, face_list FROM face_mappings ORDER BY ID DESC LIMIT 1")
    row = cursor.fetchone()

    # Close the database connection
    conn.close()

    if row:
        avg_face_path = row[0]
        animated_face_path = row[1]

        # Deserialize the face_list from BLOB (assuming it's serialized as pickle)
        face_list = pickle.loads(row[2])

        return avg_face_path, animated_face_path, face_list
    else:
        # If no data is found, return None
        return None, None, None
    

import sqlite3
import pickle

def query_recent_landmarks(db_path: str,
                           n: int) -> list:
    """
    Queries the most recent `n` entries from the `face_embeddings` database.

    The database schema is:
    | ID | face_path | landmarks | embedding |

    Parameters
    ----------
    db_path : str
        The path to the SQLite database file.
    n : int
        The number of most recent entries to retrieve.

    Returns
    -------
    list of tuples
        Each tuple contains (face_path, landmarks).
        - `face_path` is the path to an image of a saved face.
        - `landmarks` is deserialized from a BLOB into a Python list.

    Raises
    ------
    ValueError
        If `n` is less than or equal to 0.
    """
    if n <= 0:
        raise ValueError("`n` must be greater than 0.")

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query the most recent `n` entries, ordered by ID (descending)
        cursor.execute("""
        SELECT face_path, landmarks
        FROM face_embeddings
        ORDER BY id DESC
        LIMIT ?
        """, (n,))

        # Fetch all results
        rows = cursor.fetchall()

        # Deserialize landmarks and embeddings
        face_paths = []
        face_landmarks = []
        for row in rows:
            face_path, landmarks_blob = row
            landmarks = pickle.loads(landmarks_blob)  # Deserialize landmarks
            face_paths.append(face_path)
            face_landmarks.append(landmarks)

        conn.close()

        return face_paths, face_landmarks

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except pickle.PickleError as e:
        print(f"Error deserializing data: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


if __name__ == "__main__":

    import os
    import glob

    def cleanup_files():
        # Define paths
        averages_dir = "images/averages"
        faces_dir = "images/faces"
        collages_dir = "images/collages"
        db_files = ["face_embeddings.db", "face_mappings.db"]

        # Delete all .jpg files in `images/averages`
        for file_path in glob.glob(os.path.join(averages_dir, "*.jpg")):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Delete all .jpg files in `images/faces`
        for file_path in glob.glob(os.path.join(faces_dir, "*.jpg")):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Delete all .npz files in `images/animations`
        for file_path in glob.glob(os.path.join(collages_dir, "*.npz")):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Delete database files
        for db_file in db_files:
            try:
                if os.path.exists(db_file):
                    os.remove(db_file)
                    print(f"Deleted: {db_file}")
                else:
                    print(f"File not found: {db_file}")
            except Exception as e:
                print(f"Error deleting {db_file}: {e}")




    # Delete everything.
    cleanup_files()

    # Create the embedding database
    create_embedding_database(db_path="face_embeddings.db")

    # Create the face mapping database
    create_face_mapping_database(db_path="face_mappings.db")
