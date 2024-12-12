# ImagoMum by Fetal AI - Ahamisi Godsfavour

An AI-powered ultrasound analysis system for fetal health assessment.

## Features

- Fetal measurements (CRL, HC, AC, FL)
- Gender prediction
- Health assessment
- Anomaly detection
- Real-time analysis via API

## Setup

### Backend Setup


1. Create and activate virtual environment:

```bash
python3 -m venv imagomum-env
source imagomum-env/bin/activate # for mac
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
python3 src/api/app.py

```

### API ENDPOINT

```bash
- `GET /health`: Health check endpoint
- `POST /predict`: Upload and analyze ultrasound image
- `GET /`: Welcome messag

```

JSON Response:

```json
{
  "message": "Welcome to ImagoMum API"
}
```


## Development

Currently in development phase with dummy model. Training with real data coming soon.

### Roadmap

1. Data Collection and Preprocessing
2. Model Training
3. Model Evaluation
4. API Enhancement
5. Frontend Development
6. Deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Authors

- Ahamisi Godsfavour

## Acknowledgments

- Medical professionals who provided guidance
- Open-source community
- Research papers and datasets that made this possible