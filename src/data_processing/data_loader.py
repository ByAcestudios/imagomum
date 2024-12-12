import os
from pathlib import Path
import numpy as np
from ..utils.image_utils import load_image, preprocess_image

class UltrasoundDataLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_paths = []
        self.labels = {}
        
    def scan_directory(self):
        """Scan directory for images and annotations."""
        for img_path in self.data_dir.glob("**/*.jpg"):
            self.image_paths.append(img_path)
            
        print(f"Found {len(self.image_paths)} images")
    
    def load_batch(self, batch_indices):
        """Load a batch of images."""
        batch_images = []
        for idx in batch_indices:
            image_path = self.image_paths[idx]
            image = load_image(image_path)
            if image is not None:
                processed_image = preprocess_image(image)
                batch_images.append(processed_image)
        
        return np.array(batch_images)

    def get_num_samples(self):
        """Return the total number of samples."""
        return len(self.image_paths)