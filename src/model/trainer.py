import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from ..utils.logger import TrainingLogger

class UltrasoundTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize the trainer.
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration object with training parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss functions
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = {
            'fetal_measurement': nn.MSELoss(),
            'health_assessment': nn.CrossEntropyLoss(),
            'gender': nn.CrossEntropyLoss(),
            'anomaly': nn.CrossEntropyLoss()
        }
        
        self.logger = TrainingLogger(Path(config.TRAINED_MODELS_PATH) / 'logs')
        
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader)):
            # Move data to device
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Calculate loss for each task
            loss = sum(self.criterion[task](outputs[task], targets[task]) 
                      for task in outputs.keys())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                # Move data to device
                data = data.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss
                loss = sum(self.criterion[task](outputs[task], targets[task]) 
                          for task in outputs.keys())
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs.
        Args:
            num_epochs (int): Number of epochs to train
        """
        best_loss = float('inf')
        
        # Log model summary
        self.logger.log_model_summary(self.model)
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Log epoch results
            self.logger.log_epoch(
                epoch + 1,
                train_loss,
                val_loss
            )
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model('best_model.pth')
        
        # Save training history
        self.logger.plot_training_history()
        self.logger.save_history()
    
    def save_model(self, filename):
        """Save the model to a file."""
        save_path = Path(self.config.TRAINED_MODELS_PATH) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(save_path)) 