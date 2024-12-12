import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import Config
from data_processing.dataset import UltrasoundDataset
from model.model import UltrasoundModel

class ModelEvaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def evaluate(self):
        """Evaluate the model on test data."""
        self.model.eval()
        
        # Initialize dictionaries to store predictions and ground truth
        all_predictions = {
            'fetal_measurement': [],
            'health_assessment': [],
            'gender': [],
            'anomaly': []
        }
        all_targets = {
            'fetal_measurement': [],
            'health_assessment': [],
            'gender': [],
            'anomaly': []
        }
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                # Move data to device
                data = data.to(self.device)
                
                # Get model predictions
                outputs = self.model(data)
                
                # Store predictions and targets
                for task in outputs.keys():
                    preds = outputs[task].cpu().numpy()
                    if task == 'fetal_measurement':
                        # Regression task
                        all_predictions[task].extend(preds)
                    else:
                        # Classification tasks
                        preds = np.argmax(preds, axis=1)
                        all_predictions[task].extend(preds)
                    
                    all_targets[task].extend(targets[task].numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """Calculate evaluation metrics for each task."""
        metrics = {}
        
        # Fetal measurements (regression metrics)
        mse = np.mean((np.array(predictions['fetal_measurement']) - 
                      np.array(targets['fetal_measurement'])) ** 2)
        metrics['fetal_measurement'] = {
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
        
        # Classification metrics for other tasks
        for task in ['health_assessment', 'gender', 'anomaly']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets[task],
                predictions[task],
                average='weighted'
            )
            accuracy = accuracy_score(targets[task], predictions[task])
            
            metrics[task] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return metrics

def main():
    # Initialize config
    config = Config()
    
    # Load test dataset
    test_dataset = UltrasoundDataset(
        data_dir=Path(config.PROCESSED_DATA_PATH) / 'test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model and load trained weights
    model = UltrasoundModel(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(
        Path(config.TRAINED_MODELS_PATH) / 'best_model.pth'
    ))
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, test_loader, config)
    
    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluator.evaluate()
    
    # Print results
    for task, task_metrics in metrics.items():
        print(f"\n{task} metrics:")
        for metric_name, value in task_metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 