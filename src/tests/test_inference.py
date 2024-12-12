import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import Config
from src.model import UltrasoundModel
from src.inference import UltrasoundPredictor

class TestInference:
    @pytest.fixture
    def predictor(self, tmp_path):
        """Create predictor instance for testing."""
        config = Config()
        model = UltrasoundModel()
        
        # Create dummy model weights with correct architecture
        model_path = tmp_path / "best_model.pth"
        state_dict = model.state_dict()  # Get the correct state dict structure
        torch.save(state_dict, model_path)
        
        return UltrasoundPredictor(model_path=model_path, config=config)
    
    def test_prediction_format(self, predictor):
        """Test if prediction output has correct format."""
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_image)
        
        # Get prediction
        prediction = predictor.predict(image)
        
        # Check prediction format
        assert isinstance(prediction, dict), "Prediction should be a dictionary"
        assert "fetal_measurement" in prediction, "Prediction should contain fetal measurements"
        assert "health_assessment" in prediction, "Prediction should contain health assessment"
        assert "gender" in prediction, "Prediction should contain gender"
        assert "anomaly" in prediction, "Prediction should contain anomaly detection"
        
        # Check prediction shapes
        config = Config()
        assert prediction['fetal_measurement'].shape == (config.NUM_CLASSES['fetal_measurement'],), \
            "Incorrect shape for fetal measurements"
        assert prediction['health_assessment'].shape == (config.NUM_CLASSES['health_assessment'],), \
            "Incorrect shape for health assessment"
        assert prediction['gender'].shape == (config.NUM_CLASSES['gender'],), \
            "Incorrect shape for gender"
        assert prediction['anomaly'].shape == (config.NUM_CLASSES['anomaly'],), \
            "Incorrect shape for anomaly detection"
        
        # Check value ranges
        assert np.all(prediction['health_assessment'] >= 0) and np.all(prediction['health_assessment'] <= 1), \
            "Health assessment values should be between 0 and 1"
        assert np.all(prediction['gender'] >= 0) and np.all(prediction['gender'] <= 1), \
            "Gender values should be between 0 and 1"
        assert np.all(prediction['anomaly'] >= 0) and np.all(prediction['anomaly'] <= 1), \
            "Anomaly values should be between 0 and 1"