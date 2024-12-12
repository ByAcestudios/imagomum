from pathlib import Path
import torch

class Config:
    def __init__(self):
        # Model parameters
        self.NUM_CLASSES = {
            'fetal_measurement': 4,  # CRL, HC, AC, FL
            'health_assessment': 2,  # Normal/Abnormal
            'gender': 2,  # Male/Female
            'anomaly': 10   # Different types of anomalies
        }
        self.INPUT_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        
        # Training parameters
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 100
        
        # Data parameters
        self.TRAIN_SPLIT = 0.8
        self.VAL_SPLIT = 0.1
        self.TEST_SPLIT = 0.1
        
        # Paths
        self.DATA_DIR = Path("data")
        self.MODEL_DIR = Path("models")
        self.LOG_DIR = Path("logs")
        self.TRAINED_MODELS_PATH = self.MODEL_DIR / "trained"
        self.PROCESSED_DATA_PATH = self.DATA_DIR / "processed"
        
        # Create directories if they don't exist
        self.TRAINED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"