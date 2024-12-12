import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir):
        """
        Initialize the logger.
        Args:
            log_dir: Directory to save logs and plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logger()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
    def setup_logger(self):
        """Set up the logging configuration."""
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        """
        Log the results of an epoch.
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of additional metrics
        """
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if metrics:
            for key, value in metrics.items():
                if key not in self.history['metrics']:
                    self.history['metrics'][key] = []
                self.history['metrics'][key].append(value)
        
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"Training Loss: {train_loss:.4f}")
        self.logger.info(f"Validation Loss: {val_loss:.4f}")
        if metrics:
            self.logger.info("Metrics:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}")
    
    def plot_training_history(self):
        """Plot and save training history."""
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.log_dir / 'training_history.png')
        plt.close()
        
        # Plot additional metrics
        if self.history['metrics']:
            plt.figure(figsize=(10, 6))
            for metric_name, metric_values in self.history['metrics'].items():
                plt.plot(metric_values, label=metric_name)
            plt.title('Training Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(self.log_dir / 'metrics_history.png')
            plt.close()
    
    def save_history(self):
        """Save training history to JSON file."""
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def log_model_summary(self, model):
        """Log model architecture summary."""
        self.logger.info("Model Architecture:")
        self.logger.info(str(model)) 