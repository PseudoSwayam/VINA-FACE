import cv2
import argparse
import os
import numpy as np
from utils import face_utils, tts # Assuming tts might be used for confirmation

# Configuration
DB_PATH = face_utils.KNOWN_FACES_PKL

def enroll_from_image(image_path, person_name):
    """
    Enrolls a face from a given image file.
    - Detects faces in the image.
    - If multiple faces are found, attempts to use the largest one.
    - Extracts embedding and saves it with the person's name.
    """
    print(f"Attempting to enroll '{person_name}' from image: {image_path}")

    # 1. Load InsightFace model
    print("Loading InsightFace model for enrollment...")
    model = face_utils.load_insightface_model()
    if model is None:
        print("Failed to load InsightFace model. Exiting enrollment.")
        return False

    # 2. Check if image path exists
    if not os.path.exists(image_path):
        print(f"Error: Image path not found: {image_path}")
        return False

    # 3. Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from path: {image_path}")
        return False
    
    print(f"Image loaded successfully. Shape: {image.shape}")

    # 4. Detect faces in the image
    # The model.get() expects BGR format, which cv2.imread provides.
    detected_faces = face_utils.get_faces_from_frame(image, model)

    target_face_obj = None
    if not detected_faces:
        print("Error: No faces detected in the provided image.")
        return False
    elif len(detected_faces) == 1:
        print("One face detected in the image.")
        target_face_obj = detected_faces[0]
    else:
        print(f"Warning: {len(detected_faces)} faces detected in the image.")
        print("Attempting to use the largest face for enrollment.")
        
        largest_face_area = 0
        for face_obj in detected_faces:
            bbox = face_obj.bbox
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area > largest_face_area:
                largest_face_area = area
                target_face_obj = face_obj
        
        if target_face_obj:
            print("Largest face selected for enrollment.")
        else:
            # This case should ideally not be reached if detected_faces is not empty
            print("Error: Could not determine a target face from multiple detections.")
            return False

    # 5. Extract embedding
    embedding = target_face_obj.normed_embedding
    if embedding is None:
        print("Error: Could not extract embedding for the selected face. The face might be too small, blurry, or obstructed.")
        return False

    # 6. Load known faces database
    known_faces_db = face_utils.load_known_faces(DB_PATH)
    enrolled_names = {face['name'] for face in known_faces_db}

    # 7. Handle name and save
    name_to_enroll = person_name.strip()
    if not name_to_enroll:
        print("Error: Person's name cannot be empty.")
        return False

    if name_to_enroll in enrolled_names:
        overwrite = input(f"Name '{name_to_enroll}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print(f"Enrollment for '{name_to_enroll}' skipped.")
            return False
        else:
            # Remove old entry to overwrite
            known_faces_db = [f for f in known_faces_db if f['name'] != name_to_enroll]
            print(f"Preparing to overwrite existing entry for '{name_to_enroll}'.")
    
    known_faces_db.append({"name": name_to_enroll, "embedding": embedding})
    face_utils.save_known_faces(known_faces_db, DB_PATH)
    
    success_message = f"Successfully enrolled: {name_to_enroll} from image {os.path.basename(image_path)}"
    print(success_message)
    tts.speak(f"{name_to_enroll} has been enrolled from image.") # Optional TTS feedback
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a face from an image file.")
    parser.add_argument("-i", "--image", required=True, help="Path to the image file containing the face.")
    parser.add_argument("-n", "--name", required=True, help="Name of the person in the image.")
    
    args = parser.parse_args()

    enroll_from_image(args.image, args.name)

    # Create a dummy test_face for demo if not exists (similar to face_enroll.py)
    if not os.path.exists(DB_PATH):
        print("\nNote: No 'known_faces.pkl' found. A new one will be created upon successful enrollment.")
        print("If this is the first enrollment, 'known_faces.pkl' will be created.")