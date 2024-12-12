import requests
import json
from pathlib import Path

def test_api(image_path, api_url="http://localhost:8000"):
    """
    Test the API with an image.
    Args:
        image_path: Path to test image
        api_url: Base URL of the API
    """
    # Test health check
    try:
        response = requests.get(f"{api_url}/health")
        print("Health check:", response.json())
    except requests.exceptions.ConnectionError:
        print("Error: API server is not running")
        return
    
    # Test prediction
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/predict", files=files)
            
        if response.status_code == 200:
            predictions = response.json()
            print("\nPredictions:")
            print(json.dumps(predictions, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    # Test with example image
    test_image = Path("data/raw/test_image.jpg")
    if test_image.exists():
        test_api(test_image)
    else:
        print(f"No test image found at {test_image}")

if __name__ == "__main__":
    main() 