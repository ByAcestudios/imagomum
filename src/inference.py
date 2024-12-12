import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from model.model import UltrasoundModel
from utils.visualizer import PredictionVisualizer

class UltrasoundPredictor:
    def __init__(self, model_path, config):
        """
        Initialize the predictor.
        Args:
            model_path: Path to trained model weights
            config: Configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = UltrasoundModel(num_classes=config.NUM_CLASSES)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize visualizer
        self.visualizer = PredictionVisualizer(save_dir=Path(config.TRAINED_MODELS_PATH) / 'predictions')
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Make predictions for a single image.
        Args:
            image: PIL Image object
        Returns:
            Dictionary containing predictions
        """
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Process predictions
        return {
            'fetal_measurement': outputs['fetal_measurement'].cpu().numpy()[0],
            'health_assessment': outputs['health_assessment'].cpu().numpy()[0],
            'gender': outputs['gender'].cpu().numpy()[0],
            'anomaly': outputs['anomaly'].cpu().numpy()[0]
        }
    
    def predict_single_image(self, image_path):
        """
        Make predictions for a single image from file.
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary containing predictions
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        predictions = self.predict(image)
        
        # Visualize predictions
        self.visualizer.visualize_prediction(
            np.array(image),
            predictions,
            Path(image_path).stem
        )
        
        return predictions
    
    def predict_batch(self, image_paths):
        """
        Make predictions for multiple images.
        Args:
            image_paths: List of paths to ultrasound images
        Returns:
            List of prediction dictionaries
        """
        predictions_list = []
        
        for image_path in image_paths:
            try:
                predictions = self.predict_single_image(image_path)
                predictions_list.append(predictions)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        return predictions_list

def main():
    # Initialize config
    config = Config()
    
    # Initialize predictor
    predictor = UltrasoundPredictor(
        model_path=Path(config.TRAINED_MODELS_PATH) / 'best_model.pth',
        config=config
    )
    
    # Example usage
    image_path = Path(config.RAW_DATA_PATH) / 'test_image.jpg'
    if image_path.exists():
        predictions = predictor.predict_single_image(image_path)
        print("\nPredictions:")
        for key, value in predictions.items():
            print(f"\n{key}:")
            print(value)
    else:
        print(f"No test image found at {image_path}")

if __name__ == "__main__":
    main() 