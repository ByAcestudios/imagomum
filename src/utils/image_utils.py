import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def load_image(image_path):
    """
    Load an image from path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to numpy array
    image_array = np.array(image)
    
    # Resize to standard size
    image_array = cv2.resize(image_array, (224, 224))
    
    # Normalize
    image_array = image_array.astype(np.float32) / 255.0
    
    return image_array

def augment_image(image):
    """
    Apply data augmentation to image.
    
    Args:
        image: PIL Image
        
    Returns:
        PIL.Image: Augmented image
    """
    # Convert to numpy array
    image_array = np.array(image)
    
    # Example augmentation: random brightness
    brightness = np.random.uniform(0.8, 1.2)
    image_array = np.clip(image_array * brightness, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image_array)