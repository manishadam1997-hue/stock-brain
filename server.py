from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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
    return {"status": "Online", "message": "Pro-v3 Hybrid AI Terminal Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Increase data window to 2 years for 1d/2d to see the 'big picture'
        fetch_period = "5d" if request.interval == "15m" else "2y"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "Market data unavailable for this symbol."}

        # Handle Multi-Index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].iloc[:, 0].to_frame()
            df['High'] = data['High'].iloc[:, 0]
            df['Low'] = data['Low'].iloc[:, 0]
        else:
            df = data[['Close', 'High', 'Low']].copy()
        
        df.columns = ['Close', 'High', 'Low']

        # --- ADVANCED TECHNICAL FEATURE ENGINEERING ---
        # 1. Momentum: RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 2. Trend: EMA 20 (The "Institutional" Line)
        df['EMA_20'] = df['Close'].ewm(span=20).mean()

        # 3. Volatility: ATR (Average True Range)
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                   np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                   abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=14).mean()

        # 4. Lags (Past context)
        for i in range(1, 4):
            df[f'Lag_{i}'] = df['Close'].shift(i)
        
        df = df.dropna()

        # Define Features (X) and Target (y)
        features = ['EMA_20', 'RSI', 'ATR', 'Lag_1', 'Lag_2', 'Lag_3']
        X = df[features].values
        y = df['Close'].values

        # Scale features for better AI accuracy
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # GRADIENT BOOSTING: More sensitive to trend changes than Random Forest
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4).fit(X_scaled, y)

        current_close = float(df['Close'].iloc[-1])
        last_features = X_scaled[-1].reshape(1, -1)
        
        # Initial AI Prediction
        prediction = float(model.predict(last_features)[0])
        
        # --- TREND SENSITIVITY OVERRIDE ---
        # If RSI is very high/low, we adjust the AI to prevent "flat" predictions
        atr_value = float(df['ATR'].iloc[-1])
        if df['RSI'].iloc[-1] > 70: # Overbought trend
             prediction += (atr_value * 0.2)
        elif df['RSI'].iloc[-1] < 30: # Oversold trend
             prediction -= (atr_value * 0.2)

        return {
            "current_price": round(current_close, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}