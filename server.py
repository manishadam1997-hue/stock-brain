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
    interval: str 

@app.get("/")
def home():
    return {"status": "Online", "message": "Level 2 AI Terminal Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Determine data period based on interval
        fetch_period = "2d" if request.interval == "15m" else "3mo"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "Symbol not found or market closed."}

        # Clean the data (handling Multi-Index if needed)
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].iloc[:, 0].to_frame()
        else:
            df = data['Close'].to_frame()
        
        df.columns = ['Close']

        # FEATURE ENGINEERING: Give the AI 5 "Lags" and a Moving Average
        for i in range(1, 6):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df = df.dropna()

        # Define Features (X) and Target (y)
        features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'MA_5']
        X = df[features].values
        y = df['Close'].values

        # Train the model
        model = LinearRegression().fit(X, y)

        # Get current state for the next prediction
        current_close = float(df['Close'].iloc[-1])
        last_features = df[features].iloc[-1].values.reshape(1, -1)
        
        # Predict 1 step ahead (or 2 for '2d')
        steps = 2 if request.interval == "2d" else 1
        prediction = current_close
        
        for _ in range(steps):
            prediction = float(model.predict(last_features)[0])
            # Update 'last_features' for next step (simplified)
            last_features = np.roll(last_features, 1)
            last_features[0, 0] = prediction 

        return {
            "current_price": round(current_close, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}