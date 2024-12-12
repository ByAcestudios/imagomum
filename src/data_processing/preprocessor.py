import os
import cv2
import numpy as np
from pathlib import Path

from ..utils.image_utils import load_image, preprocess_image, augment_image

class UltrasoundPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
    def process_dataset(self):
        """Process all images in the raw data directory."""
        # Create processed directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for img_path in self.raw_data_path.glob("**/*.jpg"):
            try:
                # Load and preprocess image
                image = load_image(img_path)
                if image is None:
                    continue
                    
                processed_image = preprocess_image(image)
                
                # Save processed image
                output_path = self.processed_data_path / img_path.name
                np.save(str(output_path), processed_image)
                
                print(f"Processed: {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    
    def create_augmented_dataset(self):
        """Create augmented versions of the processed images."""
        augmented_path = self.processed_data_path / 'augmented'
        augmented_path.mkdir(exist_ok=True)
        
        for img_path in self.processed_data_path.glob("*.npy"):
            try:
                # Load processed image
                image = np.load(str(img_path))
                
                # Generate augmentations
                augmentations = augment_image(image)
                
                # Save augmented images
                for i, aug_image in enumerate(augmentations):
                    output_path = augmented_path / f"{img_path.stem}_aug_{i}.npy"
                    np.save(str(output_path), aug_image)
                    
            except Exception as e:
                print(f"Error augmenting {img_path}: {str(e)}") 