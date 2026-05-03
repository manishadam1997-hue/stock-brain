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
    return {"status": "Online", "message": "AI Analyst Pro - Advanced Features Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # 1. Ticker Search
        search_results = yf.Search(request.symbol, max_results=1).quotes
        if not search_results:
            return {"error": f"No results for '{request.symbol}'"}
        
        ticker_symbol = search_results[0]['symbol']
        stock = yf.Ticker(ticker_symbol)

        # 2. Optimized Fetching
        fetch_period = "5d" if request.interval == "15m" else "60d"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = stock.history(period=fetch_period, interval=fetch_interval)

        if data.empty:
            return {"error": "Market data unavailable."}

        # 3. Feature Engineering
        df = data[['Close', 'Volume']].copy()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['Index'] = np.arange(len(df))
        df = df.dropna()

        # 4. AI Training
        features = ['Index', 'Volume', 'RSI']
        X = df[features].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        avg_volume = float(df['Volume'].mean())
        current_rsi = float(df['RSI'].iloc[-1])

        # 5. Forecast
        pred_count = 5 if request.interval == "15m" else 3
        future_idx = len(df) + pred_count
        prediction = float(model.predict([[future_idx, avg_volume, current_rsi]])[0])

        # Volatility nudge
        diff = prediction - current_price
        if abs(diff) < (current_price * 0.002):
            nudge = (current_price * 0.012) 
            prediction += nudge if current_rsi > 50 else -nudge

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": ticker_symbol.upper(),
            "rsi": round(current_rsi, 2),
            "volume": int(avg_volume)
        }
    except Exception as e:
        return {"error": str(e)}