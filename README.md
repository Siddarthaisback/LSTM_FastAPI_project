# LSTM_FastAPI_project

A machine learning project combining LSTM neural networks with FastAPI for 
real-time predictions.

## Features
- LSTM model implementation for time series prediction
- FastAPI web framework for API endpoints
- RESTful API for model predictions
- Easy deployment and scalability

## Prerequisites
- Python 3.9+
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Siddarthaisback/LSTM_FastAPI_project.git
cd LSTM_FastAPI_project
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation at: `http://localhost:8000/docs`

## API Endpoints
- `GET /` - Health check
- `POST /predict` - Make predictions using LSTM model

## Project Structure
```
├── main.py              # FastAPI application
├── model/               # LSTM model files
├── data/                # Dataset files
├── requirements.txt     # Dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technologies Used
- FastAPI
- TensorFlow/Keras
- NumPy
- Pandas
- Uvicorn

