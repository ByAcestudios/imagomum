from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from inference import UltrasoundPredictor
from utils.results_formatter import ResultsFormatter

app = FastAPI(
    title="ImagoMum API",
    description="API for ultrasound image analysis",
    version="1.0.0"
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize predictor
config = Config()
predictor = UltrasoundPredictor(
    model_path=Path(config.TRAINED_MODELS_PATH) / 'best_model.pth',
    config=config
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to ImagoMum API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze ultrasound image and return predictions.
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save temporary file for prediction
        temp_path = Path(config.PROCESSED_DATA_PATH) / "temp" / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        image.save(temp_path)
        
        # Make prediction
        predictions = predictor.predict_single_image(temp_path)
        
        # Convert numpy arrays to lists
        serializable_predictions = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in predictions.items()
        }
        
        # Format predictions
        formatted_results = ResultsFormatter.format_predictions(serializable_predictions)
        
        # Clean up
        temp_path.unlink()
        
        return JSONResponse(content=formatted_results)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def start_server():
    """Start the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server() 