import pytest
import numpy as np
from pathlib import Path
import sys
from PIL import Image

# Add the src directory to the Python path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.dataset import UltrasoundDataset
from src.data_processing.preprocessor import UltrasoundPreprocessor
from src.utils.image_utils import load_image, preprocess_image

class TestDataProcessing:
    @pytest.fixture
    def setup_paths(self, tmp_path):
        """Create temporary directories for testing."""
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()
        return raw_dir, processed_dir
    
    def test_image_loading(self):
        """Test image loading functionality."""
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = Path("test_image.jpg")
        
        try:
            # Save test image
            Image.fromarray(img).save(img_path)
            
            # Test loading
            loaded_img = load_image(img_path)
            assert loaded_img is not None, "Failed to load image"
            assert loaded_img.size == (224, 224), "Incorrect image size"
            
        finally:
            # Cleanup
            if img_path.exists():
                img_path.unlink()
    
    def test_preprocessing(self):
        """Test image preprocessing."""
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # Test preprocessing
        processed_img = preprocess_image(img)
        assert processed_img.shape == (224, 224, 3), "Incorrect processed shape"
        assert processed_img.dtype == np.float32, "Incorrect data type"
        assert processed_img.max() <= 1.0, "Values not normalized"
        assert processed_img.min() >= 0.0, "Values not normalized" 