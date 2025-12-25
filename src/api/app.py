from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.quantile_regressor import QuantileRegressor
from src.models.decision_engine import DecisionEngine
from src.feature_engineering.simple_features import SimpleFeatureCalculator

app = FastAPI(title="Probabilistic Price Forecasting API")

# Initialize components
feature_calc = SimpleFeatureCalculator()
regressor = QuantileRegressor(quantiles=[0.1, 0.5, 0.9])
decision_engine = DecisionEngine()

# Load models safely
MODEL_PATH = "models/quantile_regressor"
try:
    if Path(f"{MODEL_PATH}_models.joblib").exists():
        regressor.load_models(MODEL_PATH)
        print("Probabilistic models loaded successfully.")
    else:
        print(f"Warning: Models not found at {MODEL_PATH}")
except Exception as e:
    print(f"Error loading models: {e}")

class CandleData(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0

class PredictionRequest(BaseModel):
    candles: List[CandleData]
    symbol: str

class PredictionResponse(BaseModel):
    symbol: str
    last_time: str
    q_01: float
    q_05: float
    q_09: float
    confidence_interval_width: float
    signal: int  # 1: Long, -1: Short, 0: Hold
    recommendation: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate probabilistic price forecast and trading signal.
    """
    try:
        if len(request.candles) < 20:
            raise HTTPException(status_code=400, detail="Insufficient data. Need at least 20 candles.")

        # 1. Convert to DataFrame
        df = pd.DataFrame([c.dict() for c in request.candles])
        df['time'] = pd.to_datetime(df['time'])

        # 2. Calculate Features
        df_features = feature_calc.calculate_all_features(df)
        
        # 3. Handle NaNs from rolling calculations
        df_features = df_features.dropna()
        if df_features.empty:
            raise HTTPException(status_code=400, detail="Feature calculation resulted in empty data.")

        # 4. Generate Quantile Forecasts
        # Only predict for the most recent bar
        last_features = df_features.tail(1)
        forecasts = regressor.predict(last_features)
        
        # 5. Generate Trading Signal
        # Note: DecisionEngine expects enough history for rolling confidence filter
        # For a single prediction, we'll append the forecast to a mock history if needed,
        # but the decision engine should ideally have the context.
        # Here we'll pass the tail of the calculated data for signal generation.
        
        # For simplicity in this endpoint, we'll use the latest forecasts
        q_01 = float(forecasts['q_0.1'].iloc[0])
        q_05 = float(forecasts['q_0.5'].iloc[0])
        q_09 = float(forecasts['q_0.9'].iloc[0])
        
        # Get signal from full history to satisfy rolling requirements
        full_forecasts = regressor.predict(df_features)
        signals = decision_engine.generate_signal(full_forecasts)
        last_signal = int(signals.iloc[-1])

        # 6. Formatting Recommendation
        width = q_09 - q_01
        if last_signal == 1:
            rec = "LONG (High Confidence Asymmetric Payoff)"
        elif last_signal == -1:
            rec = "SHORT (High Confidence Asymmetric Payoff)"
        else:
            rec = "HOLD (Insufficient conviction or low confidence)"

        return PredictionResponse(
            symbol=request.symbol,
            last_time=str(df['time'].iloc[-1]),
            q_01=q_01,
            q_05=q_05,
            q_09=q_09,
            confidence_interval_width=width,
            signal=last_signal,
            recommendation=rec
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "system": "Probabilistic Price Forecasting",
        "models_loaded": len(regressor.models) > 0
    }
