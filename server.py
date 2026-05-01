from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for the Flutter application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    symbol: str
    interval: str  # Options: '15m', '1d', '2d'

@app.get("/")
def home():
    return {"status": "Online", "message": "Multi-Interval AI Terminal Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Determine data windows based on selection
        # 15m needs a narrow 1-day window; 1d/2d needs a wider 1-month window
        fetch_period = "1d" if request.interval == "15m" else "1mo"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        # Download market data
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "Symbol not found or market closed for this interval."}

        # AI Prediction Logic
        df = data[['Close']].copy()
        df['S_1'] = df['Close'].shift(1)
        df = df.dropna()

        X = df[['S_1']].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        
        # Steps determine how far out the prediction goes (2 steps for '2d')
        steps = 2 if request.interval == "2d" else 1
        prediction = current_price
        for _ in range(steps):
            prediction = float(model.predict([[prediction]])[0])

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}