import click
import sys
from pathlib import Path
from typing import List
import json

from config import Config
from data_processing.preprocessor import UltrasoundPreprocessor
from model.model import UltrasoundModel
from model.trainer import UltrasoundTrainer
from inference import UltrasoundPredictor
from api.app import start_server

@click.group()
def cli():
    """ImagoMum - Ultrasound Image Analysis CLI"""
    pass

@cli.command()
@click.option('--data-dir', type=click.Path(exists=True), help='Directory containing raw data')
@click.option('--output-dir', type=click.Path(), help='Directory for processed data')
def preprocess(data_dir: str, output_dir: str):
    """Preprocess ultrasound images."""
    config = Config()
    preprocessor = UltrasoundPreprocessor(
        raw_data_path=data_dir or config.RAW_DATA_PATH,
        processed_data_path=output_dir or config.PROCESSED_DATA_PATH
    )
    
    click.echo("Starting preprocessing...")
    preprocessor.process_dataset()
    click.echo("Creating augmented dataset...")
    preprocessor.create_augmented_dataset()
    click.echo("Preprocessing completed!")

@cli.command()
@click.option('--epochs', default=100, help='Number of epochs to train')
@click.option('--batch-size', default=32, help='Batch size for training')
@click.option('--learning-rate', default=0.001, help='Learning rate')
def train(epochs: int, batch_size: int, learning_rate: float):
    """Train the model."""
    from train import main as train_main
    
    click.echo("Starting training...")
    config = Config()
    config.NUM_EPOCHS = epochs
    config.BATCH_SIZE = batch_size
    config.LEARNING_RATE = learning_rate
    
    train_main()
    click.echo("Training completed!")

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model-path', type=click.Path(exists=True), help='Path to trained model')
@click.option('--output', type=click.Path(), help='Output file for predictions')
def predict(image_path: str, model_path: str, output: str):
    """Make predictions on an image."""
    config = Config()
    predictor = UltrasoundPredictor(
        model_path=model_path or Path(config.TRAINED_MODELS_PATH) / 'best_model.pth',
        config=config
    )
    
    click.echo("Making prediction...")
    predictions = predictor.predict_single_image(image_path)
    
    if output:
        with open(output, 'w') as f:
            json.dump(predictions, f, indent=2)
        click.echo(f"Predictions saved to {output}")
    else:
        click.echo("\nPredictions:")
        click.echo(json.dumps(predictions, indent=2))

@cli.command()
@click.option('--port', default=8000, help='Port to run the server on')
def serve(port: int):
    """Start the API server."""
    click.echo(f"Starting server on port {port}...")
    start_server()

@cli.command()
def test():
    """Run tests."""
    import pytest
    sys.exit(pytest.main(['src/tests']))

@cli.command()
def check_setup():
    """Check if all dependencies are installed correctly."""
    from utils.setup_checker import check_dependencies
    check_dependencies()

if __name__ == '__main__':
    cli() 