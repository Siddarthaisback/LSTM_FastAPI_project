from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import logging
import os
import json
from models import WeatherData

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model, scaler, and selected features
model_path = 'models/best_model.joblib'
scaler_path = 'models/scaler.joblib'
features_path = 'models/selected_features.json'

model = None
scaler = None
selected_features = None

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            selected_features = json.load(f)
        logger.info("Model, scaler, and features loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model, scaler, or features: {e}")
else:
    logger.warning("Model, scaler, or features not found. Please run train_models.py first.")

# Load dataset for date-based predictions and feature means
try:
    df = pd.read_csv('data/weatherAUS.csv')
    df = df[df['Location'] == 'Sydney']
    df = df.dropna(subset=['RainTomorrow'])
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    all_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                    'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                    'Temp9am', 'Temp3pm']
    feature_means = df[all_features].mean()
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    df = None
    feature_means = None

def process_input_data(data: WeatherData) -> np.ndarray:
    """Convert WeatherData object to numpy array for prediction using selected features."""
    input_dict = data.dict(exclude_unset=True)  # Only include provided fields
    logger.info(f"Selected features: {selected_features}")
    logger.info(f"Input dict: {input_dict}")
    logger.info(f"Feature means available: {feature_means is not None}")
    
    # Handle case where feature_means might be None
    if feature_means is None:
        # Use default values if feature_means is not available
        default_values = {
            'MinTemp': 12.2, 'MaxTemp': 23.2, 'Rainfall': 2.4, 
            'Evaporation': 5.5, 'Sunshine': 7.6, 'WindGustSpeed': 40.0,
            'WindSpeed9am': 14.0, 'WindSpeed3pm': 18.0, 'Humidity9am': 68.0,
            'Humidity3pm': 51.0, 'Pressure9am': 1017.0, 'Pressure3pm': 1015.0,
            'Cloud9am': 4.4, 'Cloud3pm': 4.5, 'Temp9am': 16.0, 'Temp3pm': 21.0
        }
        input_array = np.array([[input_dict.get(feature, default_values.get(feature, 0)) 
                               for feature in selected_features]])
    else:
        input_array = np.array([[input_dict.get(feature, feature_means.get(feature, 0)) 
                               for feature in selected_features]])
    
    logger.info(f"Processed input array: {input_array}")
    return input_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page with the prediction form and forecast plot."""
    if selected_features is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Model not trained. Please run train_models.py first."
        })
    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": selected_features
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    """Handle prediction requests from the form."""
    if model is None or scaler is None or selected_features is None:
        logger.error("Prediction attempted without trained model or features")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": selected_features,
            "error": "Model not trained. Please run train_models.py first."
        })
    
    try:
        # Get form data
        form_data = await request.form()
        logger.info(f"Received form data: {dict(form_data)}")
        
        # Create WeatherData object from form data
        weather_dict = {}
        for key, value in form_data.items():
            if value and value.strip():  # Only include non-empty values
                try:
                    weather_dict[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {key}: {value}")
                    continue
        
        logger.info(f"Processed weather dict: {weather_dict}")
        
        # Create WeatherData object
        data = WeatherData(**weather_dict)
        
        input_array = process_input_data(data)
        logger.info(f"Input array: {input_array}")
        input_scaled = scaler.transform(input_array)
        logger.info(f"Scaled input: {input_scaled}")
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        rain_tomorrow = "Yes" if prediction == 1 else "No"
        logger.info(f"Prediction made: {rain_tomorrow} with probability {probability:.2f}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": selected_features,
            "rain_tomorrow": rain_tomorrow,
            "probability": f"{probability:.2f}"
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": selected_features,
            "error": f"Prediction error: {str(e)}"
        })

@app.get("/train")
async def train_model_endpoint():
    """Endpoint to trigger model training."""
    try:
        # Fixed import to match the actual filename
        from train_models import train_model
        global model, scaler, selected_features
        model, scaler, selected_features = train_model()
        logger.info("Model trained successfully via endpoint")
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return {"error": str(e)}

@app.get("/predict_date")
async def predict_date(date: str):
    """Predict rain for a specific historical date and compare with actual outcome."""
    if df is None or feature_means is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    if model is None or scaler is None or selected_features is None:
        raise HTTPException(status_code=500, detail="Model not trained.")
    try:
        date_dt = pd.to_datetime(date)
    except:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if date_dt not in df.index:
        raise HTTPException(status_code=404, detail="Date not found in dataset.")
    try:
        row = df.loc[date_dt]
        input_data = row[selected_features].fillna(feature_means[selected_features])
        input_array = input_data.values.reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        actual = row['RainTomorrow']
        return {
            "date": date,
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": probability,
            "actual": "Yes" if actual == 1 else "No"
        }
    except Exception as e:
        logger.error(f"Error predicting for date {date}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)