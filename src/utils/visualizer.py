from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class PredictionVisualizer:
    def __init__(self, save_dir):
        """
        Initialize the visualizer.
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_prediction(self, image, predictions, filename):
        """
        Visualize predictions on the image.
        Args:
            image: Input image
            predictions: Model predictions
            filename: Name for the output file
        """
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Add predictions as text
        text = []
        for key, value in predictions.items():
            if isinstance(value, dict):
                text.extend([f"{key}.{k}: {v:.2f}" for k, v in value.items()])
            else:
                text.append(f"{key}: {value}")
        
        plt.text(10, 10, "\n".join(text), color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
        
        # Save the visualization
        plt.savefig(self.save_dir / f"{filename}_prediction.png")
        plt.close() 