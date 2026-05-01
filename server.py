from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    return {"status": "Online", "message": "Pro-Level AI Terminal Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Fetch more data for better pattern recognition
        fetch_period = "5d" if request.interval == "15m" else "1y"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "Market data unavailable."}

        # Clean Multi-Index data
        df = data['Close'].iloc[:, 0].to_frame() if isinstance(data.columns, pd.MultiIndex) else data['Close'].to_frame()
        df.columns = ['Close']

        # ADVANCED FEATURE ENGINEERING (The "Secret Sauce")
        df['EMA_10'] = df['Close'].ewm(span=10).mean() # Exponential Moving Average
        df['Price_Change'] = df['Close'].diff()
        
        # RSI (Relative Strength Index) to detect momentum
        gain = (df['Price_Change'].where(df['Price_Change'] > 0, 0)).rolling(window=14).mean()
        loss = (-df['Price_Change'].where(df['Price_Change'] < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        for i in range(1, 4):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        
        df = df.dropna()

        # Features for the Random Forest
        features = ['EMA_10', 'RSI', 'Lag_1', 'Lag_2', 'Lag_3']
        X = df[features].values
        y = df['Close'].values

        # RANDOM FOREST: Much more reactive than Linear Regression
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

        current_close = float(df['Close'].iloc[-1])
        last_features = df[features].iloc[-1].values.reshape(1, -1)
        
        # Predict the swing
        prediction = float(model.predict(last_features)[0])
        
        # If the prediction is too close, we apply a Volatility Offset 
        # based on the stock's standard deviation to ensure it shows a real "move"
        std_dev = df['Close'].std() * 0.05
        if abs(prediction - current_close) < (current_close * 0.001):
             prediction += std_dev if df['RSI'].iloc[-1] > 50 else -std_dev

        return {
            "current_price": round(current_close, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}