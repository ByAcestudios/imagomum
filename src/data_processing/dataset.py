import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image

class UltrasoundDataset(Dataset):
    """Dataset class for ultrasound images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the images
            transform: Optional transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = list(self.data_dir.glob('*.jpg'))
        
    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            dict: Dictionary containing image and any associated data
        """
        image_path = self.image_files[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        # For now, return dummy labels (you'll replace these with real data)
        return {
            'image': image,
            'measurements': torch.tensor([0.0, 0.0, 0.0, 0.0]),  # CRL, HC, AC, FL
            'health_score': torch.tensor([1.0, 0.0]),  # [normal, abnormal]
            'gender': torch.tensor([1.0, 0.0]),  # [male, female]
            'filename': image_path.name
        }

    def get_sample_path(self, idx):
        """Get the file path for a sample."""
        return str(self.image_files[idx]) 