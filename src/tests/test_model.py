import pytest
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.model import UltrasoundModel
from config import Config

class TestUltrasoundModel:
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        config = Config()
        return UltrasoundModel(num_classes=config.NUM_CLASSES)
    
    def test_model_structure(self, model):
        """Test model architecture."""
        # Check if model has all required components
        assert hasattr(model, 'base_model'), "Model missing base_model"
        assert hasattr(model, 'fetal_measurement'), "Model missing fetal_measurement head"
        assert hasattr(model, 'health_assessment'), "Model missing health_assessment head"
        assert hasattr(model, 'gender'), "Model missing gender head"
        assert hasattr(model, 'anomaly'), "Model missing anomaly head"
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        batch_size = 4
        channels = 3
        height = 224
        width = 224
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        outputs = model(x)
        
        # Check outputs
        assert isinstance(outputs, dict), "Model output should be a dictionary"
        assert 'fetal_measurement' in outputs, "Missing fetal_measurement in output"
        assert 'health_assessment' in outputs, "Missing health_assessment in output"
        assert 'gender' in outputs, "Missing gender in output"
        assert 'anomaly' in outputs, "Missing anomaly in output"
        
        # Check output shapes
        config = Config()
        assert outputs['fetal_measurement'].shape == (batch_size, config.NUM_CLASSES['fetal_measurement'])
        assert outputs['health_assessment'].shape == (batch_size, config.NUM_CLASSES['health_assessment'])
        assert outputs['gender'].shape == (batch_size, config.NUM_CLASSES['gender'])
        assert outputs['anomaly'].shape == (batch_size, config.NUM_CLASSES['anomaly']) 