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
    return {"status": "Online", "message": "Master Prediction Engine Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Data periods optimized for your preferred logic
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

        # Handling Multi-Index ticker data
        if isinstance(data.columns, pd.MultiIndex):
            df = data['Close'].iloc[:, 0].to_frame()
            df['Volume'] = data['Volume'].iloc[:, 0]
        else:
            df = data[['Close', 'Volume']].copy()
        
        df.columns = ['Close', 'Volume']

        # FEATURE ENGINEERING: Price, Volume, and RSI context
        df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                      -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
        df['Index'] = np.arange(len(df))
        df = df.dropna()

        # Training on the factors you found most effective: Index, Volume, and RSI
        features = ['Index', 'Volume', 'RSI']
        X = df[features].values
        y = df['Close'].values

        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        avg_volume = float(df['Volume'].mean())
        current_rsi = float(df['RSI'].iloc[-1])

        # Forecast count per interval
        pred_count = 5 if request.interval == "15m" else 3
        future_idx = len(df) + pred_count
        
        prediction = float(model.predict([[future_idx, avg_volume, current_rsi]])[0])

        # Logic to ensure the prediction shows a distinct move based on RSI
        diff = prediction - current_price
        if abs(diff) < (current_price * 0.002):
            nudge = (current_price * 0.01) 
            prediction += nudge if current_rsi > 50 else -nudge

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": f"AI Error: {str(e)}"}