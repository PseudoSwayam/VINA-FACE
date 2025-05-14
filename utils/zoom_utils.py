import cv2
import numpy as np

def get_zoomed_region(frame, bbox, target_display_size, padding_factor=0.3):
    """
    Crops a region around the bounding box, adds padding, and resizes to target_display_size.
    
    Args:
        frame (np.ndarray): The original full frame.
        bbox (tuple/list/np.ndarray): Bounding box (x1, y1, x2, y2).
        target_display_size (tuple): Desired output size (width, height).
        padding_factor (float): Percentage to expand the bounding box by (e.g., 0.2 for 20%).
                                This factor is applied to width and height independently.
                                Total added padding = padding_factor * dimension.

    Returns:
        np.ndarray: The zoomed and resized region, or None if issues occur.
    """
    if frame is None or bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    fh, fw = frame.shape[:2]
    
    bb_w, bb_h = x2 - x1, y2 - y1
    if bb_w <= 0 or bb_h <= 0:
        return None # Invalid bounding box

    # Calculate padding
    pad_w = int(bb_w * padding_factor)
    pad_h = int(bb_h * padding_factor)

    # New coordinates with padding
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(fw, x2 + pad_w)
    crop_y2 = min(fh, y2 + pad_h)

    # Crop the region
    cropped_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if cropped_region.size == 0:
        # Fallback: return a black image of target_display_size
        return np.zeros((target_display_size[1], target_display_size[0], frame.shape[2] if frame.ndim == 3 else 1), dtype=frame.dtype)

    # Resize to the target display size (e.g., full window size)
    try:
        zoomed_image = cv2.resize(cropped_region, target_display_size, interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"OpenCV resize error: {e}. Cropped region shape: {cropped_region.shape}, target: {target_display_size}")
        # Fallback: return a black image or the unresized crop if it fits (less ideal)
        return np.zeros((target_display_size[1], target_display_size[0], frame.shape[2] if frame.ndim == 3 else 1), dtype=frame.dtype)
        
    return zoomed_image