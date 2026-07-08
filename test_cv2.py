import cv2
import numpy as np

def process_image(file_path):
    image_cv = cv2.imread(file_path)
    if image_cv is None:
        return "Failed to load"
        
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast and brightness
    alpha = 1.5 # Contrast control
    beta = 0    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Remove noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return "Success"

print(process_image("dummy.jpg"))
