# utils/camera_utils.py

import cv2
import numpy as np

def list_available_cameras(max_to_test=5): # This function remains for manual debugging if needed
    """Lists available camera indices and their status."""
    available_indices = []
    print("--- Camera Availability Check (for manual debugging) ---")
    for i in range(max_to_test):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            print(f"  Index {i}: Available")
            available_indices.append(i)
            cap_test.release()
        else:
            print(f"  Index {i}: NOT Available or Error")
    print("-------------------------------------------------------")
    return available_indices

def get_camera(width=1280, height=720, desired_fps=30):
    cap = None
    TARGET_CAMERA_INDEX = 0
    current_available = []
    for i in range(3): # Check first few indices
        temp_check = cv2.VideoCapture(i)
        if temp_check.isOpened():
            current_available.append(i)
            temp_check.release()
    print(f"(OpenCV currently sees available indices: {current_available})")


    temp_cap = cv2.VideoCapture(TARGET_CAMERA_INDEX)

    if temp_cap.isOpened():
        print(f"  Successfully opened camera index {TARGET_CAMERA_INDEX}.")
        temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        temp_cap.set(cv2.CAP_PROP_FPS, desired_fps)

        actual_width = temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = temp_cap.get(cv2.CAP_PROP_FPS)

        print(f"  Camera index {TARGET_CAMERA_INDEX}: Requested {width}x{height}@{desired_fps}FPS, Got {actual_width}x{actual_height}@{actual_fps}FPS")

        if actual_width >= 640 and actual_height >= 480: # Basic check for usability
            cap = temp_cap
            print(f"  SELECTED camera index {TARGET_CAMERA_INDEX}.")
        else:
            print(f"  Camera index {TARGET_CAMERA_INDEX} resolution {actual_width}x{actual_height} too low or failed to set. Releasing.")
            temp_cap.release()
            cap = None # Ensure cap is None if conditions not met
    else:
        print(f"  Could not open camera index {TARGET_CAMERA_INDEX}.")
        print(f"  Ensure this camera is active and available (e.g., iPhone Continuity Camera selected in Photo Booth).")
        cap = None # Ensure cap is None

    if cap is None:
        print(f"Error: Failed to open and configure the hardcoded camera index {TARGET_CAMERA_INDEX}.")
        print("Tips: - Double-check TARGET_CAMERA_INDEX in camera_utils.py.")
        print("      - Make sure your iPhone Continuity Camera is active *before* running the script.")
        print("      - Run `python -c \"from utils.camera_utils import list_available_cameras; list_available_cameras()\"` to see what indices OpenCV detects.")
        return None # Return None if the specific camera could not be opened
    
    final_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    final_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    final_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera active (Index {TARGET_CAMERA_INDEX}) with resolution: {final_width}x{final_height} @ {final_fps}FPS")
    if final_height < 720 or final_width < 1280:
         print(f"Warning: Camera resolution may be lower than preferred 1280x720.")

    return cap

def enhance_contrast(frame, alpha=1.2, beta=5):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)