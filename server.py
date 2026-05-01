from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your Flutter app can talk to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    symbol: str
    interval: str  # This tells the AI which timeframe to use

@app.get("/")
def home():
    return {"status": "Online", "message": "Multi-Timeframe AI is Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Determine data period based on interval
        # 15m needs a short period (1 day), 1d/2d needs a longer period (1 month)
        fetch_period = "1d" if request.interval == "15m" else "1mo"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        # Download data
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "No market data found for this symbol."}

        # AI Logic: Predict the next closing price
        df = data[['Close']].copy()
        df['S_1'] = df['Close'].shift(1)
        df = df.dropna()

        X = df[['S_1']].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        current_price = float(df['Close'].iloc[-1])
        
        # If interval is '2d', we predict 2 steps ahead
        steps = 2 if request.interval == "2d" else 1
        prediction = current_price
        for _ in range(steps):
            prediction = float(model.predict([[prediction]])[0])

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval
        }
    except Exception as e:
        return {"error": str(e)}