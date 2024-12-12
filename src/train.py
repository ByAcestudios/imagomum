import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config import Config
from data_processing.dataset import UltrasoundDataset
from data_processing.preprocessor import UltrasoundPreprocessor
from model.model import UltrasoundModel
from model.trainer import UltrasoundTrainer

def main():
    # Initialize config
    config = Config()
    
    # Preprocess data if needed
    preprocessor = UltrasoundPreprocessor(
        raw_data_path=config.RAW_DATA_PATH,
        processed_data_path=config.PROCESSED_DATA_PATH
    )
    
    # Process dataset and create augmentations
    print("Processing dataset...")
    preprocessor.process_dataset()
    preprocessor.create_augmented_dataset()
    
    # Create datasets
    train_dataset = UltrasoundDataset(
        data_dir=Path(config.PROCESSED_DATA_PATH) / 'train'
    )
    val_dataset = UltrasoundDataset(
        data_dir=Path(config.PROCESSED_DATA_PATH) / 'val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = UltrasoundModel(num_classes=config.NUM_CLASSES)
    
    # Initialize trainer
    trainer = UltrasoundTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Start training
    print("Starting training...")
    trainer.train(num_epochs=config.NUM_EPOCHS)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 