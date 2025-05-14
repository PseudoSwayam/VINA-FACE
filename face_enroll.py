import cv2
import time
import os
import numpy as np
from utils import camera_utils, face_utils, tts

# Configuration
DB_PATH = face_utils.KNOWN_FACES_PKL
CAPTURE_KEY = ord('s')
QUIT_KEY = ord('q')
MIN_FACE_SIZE_ENROLL = (60, 60)
CAPTURE_COOLDOWN = 2.0 # Cooldown between actual capture attempts
PREVIEW_DETECTION_INTERVAL = 0.2 # Seconds: How often to update the preview rectangle (similar to main.py)

def enroll_face():
    print("Loading InsightFace model for enrollment...")
    model = face_utils.load_insightface_model()
    if model is None:
        print("Failed to load InsightFace model. Exiting enrollment.")
        return

    known_faces_db = face_utils.load_known_faces(DB_PATH)
    enrolled_names = {face['name'] for face in known_faces_db}

    print("Initializing camera for enrollment...")
    cap = camera_utils.get_camera()
    if cap is None:
        print("Failed to access camera. Exiting enrollment.")
        return

    cv2.namedWindow("VINA-Face Enrollment", cv2.WINDOW_AUTOSIZE)
    print("\nEnrollment Instructions:")
    print(f"  - Position your face clearly in the frame.")
    print(f"  - Press '{chr(CAPTURE_KEY)}' to capture and enroll the highlighted face.")
    print(f"  - Press '{chr(QUIT_KEY)}' to quit enrollment.")
    print("Ensure good lighting and a clear view of the face.\n")

    last_capture_attempt_time = 0
    last_preview_detection_time = 0 # For throttling preview updates
    
    target_face_obj_for_preview = None # Keep the last known preview target

    while True:
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            print("Error: Failed to capture frame from camera.")
            time.sleep(0.1)
            continue
        
        display_frame = live_frame.copy()
        frame_h, frame_w = live_frame.shape[:2]
        current_time = time.time()

        # --- Throttled Face detection on the LIVE feed for visual feedback ---
        if current_time - last_preview_detection_time > PREVIEW_DETECTION_INTERVAL:
            last_preview_detection_time = current_time
            # print("Running preview detection...") # Debug print
            detected_faces_for_preview = face_utils.get_faces_from_frame(live_frame, model)
            if detected_faces_for_preview:
                target_face_obj_for_preview = face_utils.get_center_most_face(detected_faces_for_preview, frame_w, frame_h)
            else:
                target_face_obj_for_preview = None # Clear if no faces detected in this interval
        
        # --- Draw based on the (potentially stale but recently updated) target_face_obj_for_preview ---
        if target_face_obj_for_preview:
            bbox = target_face_obj_for_preview.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_w, face_h = x2 - x1, y2 - y1

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Target Face", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if face_w < MIN_FACE_SIZE_ENROLL[0] or face_h < MIN_FACE_SIZE_ENROLL[1]:
                cv2.putText(display_frame, "Face too small/far", (frame_w//2 - 100, frame_h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "No target face detected", (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("VINA-Face Enrollment", display_frame)
        key = cv2.waitKey(1) & 0xFF # Crucial: Keep waitKey low for responsive GUI

        if key == CAPTURE_KEY:
            capture_press_time = time.time() # Renamed current_time to avoid conflict
            if capture_press_time - last_capture_attempt_time < CAPTURE_COOLDOWN:
                print("Please wait a moment before capturing again.")
                continue
            last_capture_attempt_time = capture_press_time

            if not target_face_obj_for_preview:
                print("No target face was highlighted for capture. Adjust position and try again.")
                continue

            print("Capture key pressed. Freezing frame and processing selected face for enrollment...")
            frame_to_enroll_from = live_frame.copy() 

            print("Detecting faces on the frozen frame (this is a single, focused detection)...")
            # This call to model.get() is on a single frozen frame, which should be fine,
            # similar to how enroll_from_image.py works.
            detected_faces_on_frozen_frame = face_utils.get_faces_from_frame(frame_to_enroll_from, model)
            
            if not detected_faces_on_frozen_frame:
                print("Error: No faces detected on the frozen frame. Try again.")
                continue

            actual_target_face_obj_for_enrollment = face_utils.get_center_most_face(
                detected_faces_on_frozen_frame, frame_w, frame_h
            )

            if not actual_target_face_obj_for_enrollment:
                print("Error: Could not select a target face from the frozen frame. Try again.")
                continue

            print("Extracting embedding from the selected face in the frozen frame...")
            try:
                embedding = actual_target_face_obj_for_enrollment.normed_embedding
                if embedding is None or not isinstance(embedding, np.ndarray) or embedding.size == 0:
                    print("Error: Failed to extract a valid embedding from the selected face.")
                    problem_frame_filename = f"debug_enroll_fail_embedding_{int(time.time())}.png"
                    cv2.imwrite(problem_frame_filename, frame_to_enroll_from)
                    print(f"Saved problematic frame to: {problem_frame_filename}")
                    continue
                print("Embedding extracted successfully.")
            except Exception as e:
                print(f"An error occurred during embedding extraction: {e}")
                problem_frame_filename = f"debug_enroll_error_embedding_{int(time.time())}.png"
                cv2.imwrite(problem_frame_filename, frame_to_enroll_from)
                print(f"Saved problematic frame to: {problem_frame_filename}")
                continue

            cv2.destroyWindow("VINA-Face Enrollment")
            name = ""
            try:
                name = input("Enter person's name (or type 'cancel'): ").strip()
            except KeyboardInterrupt:
                print("\nEnrollment name input cancelled by user.")
                cv2.namedWindow("VINA-Face Enrollment", cv2.WINDOW_AUTOSIZE)
                continue
            except EOFError:
                print("\nEnrollment name input stream closed. Cancelling.")
                cv2.namedWindow("VINA-Face Enrollment", cv2.WINDOW_AUTOSIZE)
                break
            cv2.namedWindow("VINA-Face Enrollment", cv2.WINDOW_AUTOSIZE)

            if name.lower() == 'cancel' or not name:
                print("Enrollment cancelled for this face.")
                continue
            
            if name in enrolled_names:
                overwrite_input = ""
                try:
                    overwrite_input = input(f"Name '{name}' already exists. Overwrite? (y/n): ").strip().lower()
                except KeyboardInterrupt:
                    print("\nOverwrite confirmation cancelled.")
                    continue
                except EOFError:
                    print("\nOverwrite input stream closed. Cancelling.")
                    break
                if overwrite_input != 'y':
                    print(f"Skipping enrollment for {name}.")
                    continue
                else:
                    known_faces_db = [f for f in known_faces_db if f['name'] != name]
            
            known_faces_db.append({"name": name, "embedding": embedding})
            face_utils.save_known_faces(known_faces_db, DB_PATH)
            enrolled_names.add(name)
            success_message = f"Successfully enrolled: {name}"
            print(success_message)
            tts.speak(f"{name} has been enrolled.")

        elif key == QUIT_KEY:
            print("Exiting enrollment system.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll_face()
    if not os.path.exists(DB_PATH):
        print(f"\nNote: '{DB_PATH}' was not found or created. Please ensure enrollments are successful.")