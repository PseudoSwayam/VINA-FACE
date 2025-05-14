import cv2
import time
import numpy as np
from utils import camera_utils, face_utils, tts, zoom_utils

# --- Configuration Constants ---
SIMILARITY_THRESHOLD = 0.50
ZOOM_DURATION_SECONDS = 1.7
GREETING_COOLDOWN_SECONDS = 5
DETECTION_INTERVAL_SECONDS = 0.1

# --- Global State ---
# This is defined at the module level, making it global by default.
USE_LOW_LIGHT_ENHANCEMENT = True # Initial state

last_greeted_time = {}
insightface_model = None
known_faces_db = []

def initialize_system():
    """Loads models, database, and camera."""
    global insightface_model, known_faces_db # These are assigned to within this function.
    
    print("Initializing VINA-Face System...")
    insightface_model = face_utils.load_insightface_model()
    if insightface_model is None:
        print("CRITICAL: Failed to load InsightFace model. Exiting.")
        return None, None

    known_faces_db = face_utils.load_known_faces()
    if not known_faces_db:
        print("Warning: No known faces in the database. Please enroll faces using face_enroll.py or enroll_from_image.py.")
    else:
        print(f"Loaded {len(known_faces_db)} known faces: {[f['name'] for f in known_faces_db]}")

    cap = camera_utils.get_camera()
    if cap is None:
        print("CRITICAL: Failed to initialize camera. Exiting.")
        return None, None
    
    print("System Initialized. Starting live feed...")
    return cap, insightface_model


def process_frame(frame_to_process, model, frame_width, frame_height):
    """
    Detects and recognizes faces in a single frame.
    Returns: (recognized_name, face_bbox) or (None, None) or ("Unknown", face_bbox)
    """
    detected_faces = face_utils.get_faces_from_frame(frame_to_process, model)

    if not detected_faces:
        return None, None

    target_face_obj = face_utils.get_center_most_face(detected_faces, frame_width, frame_height)

    if not target_face_obj:
        return None, None

    current_embedding = target_face_obj.normed_embedding
    face_bbox = target_face_obj.bbox.astype(int)

    if current_embedding is None:
        print("Warning: Face detected but embedding could not be extracted.")
        return "Unknown", face_bbox

    if not known_faces_db:
        return "Unknown", face_bbox

    best_match_name = None
    max_similarity = -1.0

    for known_face in known_faces_db:
        similarity = face_utils.compare_embeddings(current_embedding, known_face['embedding'])
        if similarity > SIMILARITY_THRESHOLD and similarity > max_similarity:
            max_similarity = similarity
            best_match_name = known_face['name']
    
    if best_match_name:
        return best_match_name, face_bbox
    else:
        return "Unknown", face_bbox


def main_loop():
    # Declare global variables that will be MODIFIED within this function's scope
    global last_greeted_time
    global USE_LOW_LIGHT_ENHANCEMENT # <-- CORRECTED PLACEMENT

    cap, model = initialize_system()
    if cap is None or model is None:
        return

    cv2.namedWindow("VINA-Face", cv2.WINDOW_AUTOSIZE)

    zoom_active = False
    zoom_return_time = 0
    name_to_greet_after_zoom = None
    last_detection_run_time = 0
    current_processing_bbox_for_zoom = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: No frame from camera. Ending.")
                time.sleep(0.5)
                break
            
            processed_frame = frame.copy()
            # Reading USE_LOW_LIGHT_ENHANCEMENT is now fine because it was declared global above
            if USE_LOW_LIGHT_ENHANCEMENT:
                processed_frame = camera_utils.enhance_contrast(processed_frame)

            display_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            current_time = time.time()

            # --- Zoom Logic ---
            if zoom_active:
                if current_time >= zoom_return_time:
                    zoom_active = False
                    current_processing_bbox_for_zoom = None
                    if name_to_greet_after_zoom and name_to_greet_after_zoom != "Unknown":
                        greeting_text = f"Your friend {name_to_greet_after_zoom} is in front of you."
                        print(f"TTS: {greeting_text}")
                        tts.speak(greeting_text)
                        last_greeted_time[name_to_greet_after_zoom] = current_time # Modifying last_greeted_time
                    name_to_greet_after_zoom = None
                else:
                    if current_processing_bbox_for_zoom is not None:
                        zoomed_display = zoom_utils.get_zoomed_region(frame, current_processing_bbox_for_zoom, 
                                                                      (frame_width, frame_height))
                        if zoomed_display is not None:
                            display_frame = zoomed_display
                        else:
                            zoom_active = False 
                            current_processing_bbox_for_zoom = None
                            name_to_greet_after_zoom = None
                    else:
                        zoom_active = False
                        name_to_greet_after_zoom = None

            # --- Detection and Recognition Logic ---
            if not zoom_active and (current_time - last_detection_run_time > DETECTION_INTERVAL_SECONDS):
                last_detection_run_time = current_time
                recognized_name, face_bbox = process_frame(processed_frame, model, frame_width, frame_height)

                if face_bbox is not None:
                    current_processing_bbox_for_zoom = face_bbox
                    if recognized_name:
                        zoom_active = True
                        zoom_return_time = current_time + ZOOM_DURATION_SECONDS
                        if recognized_name != "Unknown":
                            # Accessing last_greeted_time
                            if recognized_name not in last_greeted_time or \
                               (current_time - last_greeted_time.get(recognized_name, 0) > GREETING_COOLDOWN_SECONDS):
                                name_to_greet_after_zoom = recognized_name
                            else:
                                name_to_greet_after_zoom = None
                        else:
                            name_to_greet_after_zoom = None
                    
                        zoomed_display_on_detect = zoom_utils.get_zoomed_region(frame, current_processing_bbox_for_zoom,
                                                                      (frame_width, frame_height))
                        if zoomed_display_on_detect is not None:
                            display_frame = zoomed_display_on_detect

            cv2.imshow("VINA-Face", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed. Exiting...")
                break
            elif key == ord('l'):
                # Modifying USE_LOW_LIGHT_ENHANCEMENT is now fine
                USE_LOW_LIGHT_ENHANCEMENT = not USE_LOW_LIGHT_ENHANCEMENT
                print(f"Low light enhancement {'ON' if USE_LOW_LIGHT_ENHANCEMENT else 'OFF'}")

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("VINA-Face system shut down.")

if __name__ == "__main__":
    main_loop()