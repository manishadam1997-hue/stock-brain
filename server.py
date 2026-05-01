from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
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
    return {"status": "Online", "message": "Multi-Interval AI Terminal Active"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        fetch_period = "1d" if request.interval == "15m" else "1mo"
        fetch_interval = "15m" if request.interval == "15m" else "1d"
        
        data = yf.download(
            tickers=request.symbol, 
            period=fetch_period, 
            interval=fetch_interval, 
            progress=False
        )

        if data.empty:
            return {"error": "Symbol not found or market closed."}

        # --- FIX FOR THE 'SERIES' ERROR ---
        # If columns are layered (MultiIndex), we grab just the 'Close' column for our ticker
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close'].iloc[:, 0]
        else:
            close_data = data['Close']
        
        # Convert to a clean DataFrame for the AI
        df = pd.DataFrame(close_data)
        df.columns = ['Close']
        df['S_1'] = df['Close'].shift(1)
        df = df.dropna()

        X = df[['S_1']].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        # Get the very last price as a single number
        current_price = float(df['Close'].iloc[-1])
        
        steps = 2 if request.interval == "2d" else 1
        prediction = current_price
        for _ in range(steps):
            # Ensure prediction stays a single float
            pred_array = model.predict([[prediction]])
            prediction = float(pred_array[0])

        return {
            "current_price": round(current_price, 2),
            "prediction": round(prediction, 2),
            "interval": request.interval,
            "symbol": request.symbol.upper()
        }
    except Exception as e:
        return {"error": str(e)}