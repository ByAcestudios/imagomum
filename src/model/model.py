import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class UltrasoundModel(nn.Module):
    def __init__(self, num_classes=None):
        super(UltrasoundModel, self).__init__()
        
        if num_classes is None:
            from src.config import Config
            config = Config()
            num_classes = config.NUM_CLASSES
        
        # Load pretrained ResNet50 with the latest weights
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        
        # Feature dimension from ResNet50
        feature_dim = 2048
        
        # Task-specific heads
        self.fetal_measurement = nn.Linear(feature_dim, num_classes['fetal_measurement'])
        self.health_assessment = nn.Linear(feature_dim, num_classes['health_assessment'])
        self.gender = nn.Linear(feature_dim, num_classes['gender'])
        self.anomaly = nn.Linear(feature_dim, num_classes['anomaly'])
        
    def forward(self, x):
        # Extract features
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        
        # Get predictions for each task
        return {
            'fetal_measurement': self.fetal_measurement(x),
            'health_assessment': torch.sigmoid(self.health_assessment(x)),
            'gender': torch.sigmoid(self.gender(x)),
            'anomaly': torch.sigmoid(self.anomaly(x))
        } 