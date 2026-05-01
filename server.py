from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    symbol: str
    interval: str  # Added: '15m', '1d', '2d'

@app.get("/")
def home():
    return {"status": "Online", "message": "Stock Brain AI Multi-Timeframe is running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Map UI selection to yfinance periods
        period_map = {"15m": "1d", "1d": "1mo", "2d": "2mo"}
        interval_map = {"15m": "15m", "1d": "1d", "2d": "1d"}
        
        # Fetch data
        data = yf.download(
            tickers=request.symbol, 
            period=period_map.get(request.interval, "1mo"), 
            interval=interval_map.get(request.interval, "1d"),
            progress=False
        )

        if data.empty:
            return {"error": "No data found for this symbol."}

        # Use 'Close' prices for the AI
        df = data[['Close']].copy()
        df['S_1'] = df['Close'].shift(1)
        df = df.dropna()

        # AI Model (Linear Regression)
        X = df[['S_1']].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        
        # Adjust prediction step based on timeframe
        # If '2d', we predict 2 steps ahead relative to 1d data
        prediction_step = 2 if request.interval == "2d" else 1
        predicted_price = current_price
        
        for _ in range(prediction_step):
            predicted_price = float(model.predict([[predicted_price]])[0])

        return {
            "symbol": request.symbol,
            "interval": request.interval,
            "current_price": round(current_price, 2),
            "prediction": round(predicted_price, 2)
        }
    except Exception as e:
        return {"error": str(e)}