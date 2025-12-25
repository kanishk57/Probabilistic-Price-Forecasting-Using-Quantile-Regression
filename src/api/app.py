from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from src.models.lgbm_classifier import FVGFillClassifier
from src.models.trade_outcome_classifier import TradeOutcomeClassifier
from src.feature_engineering.feature_calculator import FeatureCalculator

app = FastAPI(title="ICT ML Trading API")

# Load models at startup
fill_classifier = FVGFillClassifier()
try:
    fill_classifier.load_model('models/fvg_fill_classifier.pkl')
except Exception:
    pass

outcome_classifier = TradeOutcomeClassifier()
try:
    outcome_classifier.load_model('models/trade_outcome_classifier.pkl')
except Exception:
    pass

feature_calc = FeatureCalculator()

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
    fvg_detected: bool
    fvg_type: Optional[str]
    fill_probability: float
    tp_probability: float
    sl_probability: float
    timeout_probability: float
    expected_candles_to_fill: Optional[float]
    confidence_score: float
    recommendation: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict FVG fill probability and trade outcome
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([c.dict() for c in request.candles])
        df['time'] = pd.to_datetime(df['time'])

        # Calculate features
        df = feature_calc.calculate_all_features(df)

        # Check if FVG detected in last candle
        last_row = df.iloc[-1]

        if pd.isna(last_row.get('fvg_type')):
            return PredictionResponse(
                fvg_detected=False,
                fvg_type=None,
                fill_probability=0.0,
                tp_probability=0.0,
                sl_probability=0.0,
                timeout_probability=0.0,
                expected_candles_to_fill=None,
                confidence_score=0.0,
                recommendation="NO_SIGNAL"
            )

        # Get predictions
        fill_prob = fill_classifier.predict(df)[-1] if fill_classifier.model is not None else 0.0
        outcome_probs = outcome_classifier.predict_proba(df)[-1] if outcome_classifier.model is not None else [0.0, 0.0, 0.0]

        tp_prob = outcome_probs[0]
        sl_prob = outcome_probs[1]
        timeout_prob = outcome_probs[2]

        # Confidence score (combination of both models)
        confidence = (fill_prob + tp_prob) / 2

        # Generate recommendation
        if confidence > 0.70 and tp_prob > 0.60:
            recommendation = "STRONG_BUY" if last_row['fvg_type'] == 'bullish' else "STRONG_SELL"
        elif confidence > 0.60 and tp_prob > 0.50:
            recommendation = "MODERATE_BUY" if last_row['fvg_type'] == 'bullish' else "MODERATE_SELL"
        else:
            recommendation = "NO_TRADE"

        return PredictionResponse(
            fvg_detected=True,
            fvg_type=last_row['fvg_type'],
            fill_probability=float(fill_prob),
            tp_probability=float(tp_prob),
            sl_probability=float(sl_prob),
            timeout_probability=float(timeout_prob),
            expected_candles_to_fill=None,  # Add time regressor if needed
            confidence_score=float(confidence),
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": (fill_classifier.model is not None and outcome_classifier.model is not None)}
