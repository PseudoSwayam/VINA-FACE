import insightface
from insightface.app import FaceAnalysis
import numpy as np
import pickle
import os # For path joining

# Path for the known faces database
DATABASE_DIR = os.path.dirname(os.path.abspath(__file__)) # utils directory
KNOWN_FACES_PKL = os.path.join(os.path.dirname(DATABASE_DIR), "known_faces.pkl") # vina_face/known_faces.pkl


def load_insightface_model():
    """Loads the InsightFace model."""
    try:
        # Try CoreML first for Apple Silicon, then CPU
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        model = FaceAnalysis(name='buffalo_l', 
                             allowed_modules=['detection', 'recognition'], 
                             providers=providers)
    except Exception as e:
        print(f"Failed to load InsightFace with CoreMLExecutionProvider: {e}")
        print("Falling back to CPUExecutionProvider for InsightFace.")
        providers = ['CPUExecutionProvider']
        model = FaceAnalysis(name='buffalo_l', 
                             allowed_modules=['detection', 'recognition'], 
                             providers=providers)
    
    model.prepare(ctx_id=0, det_size=(640, 640)) # det_size can be tuned
    print("InsightFace model loaded successfully.")
    return model

def get_faces_from_frame(frame, model):
    """
    Detects faces and extracts embeddings using the InsightFace model.
    Returns a list of InsightFace Face objects.
    """
    faces = model.get(frame)
    return faces

def get_center_most_face(faces, frame_width, frame_height):
    """
    Selects the face closest to the center of the frame.
    faces: list of InsightFace Face objects.
    """
    if not faces:
        return None

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    closest_face_obj = None
    min_dist_sq = float('inf')

    for face_obj in faces:
        bbox = face_obj.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2
        
        dist_sq = (face_center_x - frame_center_x)**2 + (face_center_y - frame_center_y)**2
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_face_obj = face_obj
            
    return closest_face_obj

def compare_embeddings(emb1, emb2):
    """
    Compares two embeddings using cosine similarity (dot product for normalized vectors).
    """
    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    # Ensure embeddings are L2 normalized if not already (InsightFace provides normed_embedding)
    # emb1 = emb1 / np.linalg.norm(emb1)
    # emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

def load_known_faces(db_path=KNOWN_FACES_PKL):
    """Loads known faces from the pickle database."""
    try:
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
            valid_data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item and 'embedding' in item:
                        # Ensure embedding is numpy array and float32
                        item['embedding'] = np.array(item['embedding'], dtype=np.float32)
                        if isinstance(item['name'], str):
                            valid_data.append(item)
                        else:
                            print(f"Warning: Skipping item with non-string name: {item}")
                    else:
                        print(f"Warning: Skipping malformed item in database: {item}")
                print(f"Loaded {len(valid_data)} known faces.")
                return valid_data
            else:
                print("Warning: Database file is not a list. Initializing new database.")
                return []
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"Could not load known faces from {db_path} (Error: {e}). Starting with an empty database.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading known faces: {e}")
        return []


def save_known_faces(data, db_path=KNOWN_FACES_PKL):
    """Saves known faces to the pickle database."""
    try:
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Known faces database saved to {db_path} with {len(data)} entries.")
    except Exception as e:
        print(f"Error saving known faces database: {e}")