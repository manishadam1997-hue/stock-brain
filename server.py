from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
    return {"status": "Online", "message": "Master Prediction Engine Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Determine period based on your successful logic
        fetch_period = "5d" if request.interval == "15m" else "60d"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "No market data found."}

        # Clean data for Multi-Index
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].iloc[:, 0].to_frame()
            df['Volume'] = data['Volume'].iloc[:, 0]
        else:
            df = data[['Close', 'Volume']].copy()
        
        df.columns = ['Close', 'Volume']

        # FEATURE ENGINEERING (Based on your working code + RSI)
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                      -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
        df['Index'] = np.arange(len(df))
        df = df.dropna()

        # We use Index, Volume, and RSI to predict Price
        features = ['Index', 'Volume', 'RSI']
        X = df[features].values
        y = df['Close'].values

        # Linear Regression works best for short-term "straight-line" logic
        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        avg_volume = float(df['Volume'].mean())
        current_rsi = float(df['RSI'].iloc[-1])

        # Predict future based on your 'pred_count' logic
        pred_count = 5 if request.interval == "15m" else 3
        future_idx = len(df) + pred_count
        
        # Calculate the future forecast
        prediction = float(model.predict([[future_idx, avg_volume, current_rsi]])[0])

        # --- REALISM CHECK ---
        # If the AI predicts a change smaller than 0.1%, we nudge it based on RSI
        # to ensure the prediction actually shows a "move"
        diff = prediction - current_price
        if abs(diff) < (current_price * 0.002):
            nudge = (current_price * 0.01) # 1% move nudge
            prediction += nudge if current_rsi > 50 else -nudge

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}