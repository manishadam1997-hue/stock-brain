from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/predict/{user_input}/{mode}")
def predict_stock(user_input: str, mode: str):
    try:
        search_results = yf.Search(user_input, max_results=1).quotes
        if not search_results:
            return {"error": f"No results for '{user_input}'"}
        
        ticker = search_results[0]['symbol']
        stock = yf.Ticker(ticker)
        
        if mode == "15min":
            hist = stock.history(period="5d", interval="1m")
            pred_count = 15
            time_format = '%H:%M'
        else:
            hist = stock.history(period="60d")
            pred_count = 3
            time_format = '%b %d'

        hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

        if len(hist) < 15:
            return {"error": f"Not enough clean data available for {ticker}."}

        past_ohlc = []
        for index, row in hist.tail(150).iterrows(): 
            past_ohlc.append({
                "timestamp": int(index.timestamp() * 1000), 
                "time": index.strftime(time_format),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
            })

        close_series = hist['Close']
        prices = [float(p) for p in close_series.values]
        volumes = [float(v) for v in hist['Volume'].values]
        
        sma_short = close_series.rolling(window=5).mean()
        sma_long = close_series.rolling(window=15).mean()
        
        last_sma_short = float(sma_short.values[-1]) if not pd.isna(sma_short.values[-1]) else 0.0
        last_sma_long = float(sma_long.values[-1]) if not pd.isna(sma_long.values[-1]) else 0.0
        
        momentum_is_up = bool(last_sma_short > last_sma_long)
        
        X = np.array([[i, volumes[i]] for i in range(len(prices))])
        y = np.array(prices)
        model = LinearRegression().fit(X, y)

        avg_volume = float(np.mean(volumes[-10:]))
        future_X = np.array([[len(prices) + i, avg_volume] for i in range(pred_count)])
        predictions = model.predict(future_X)
        
        last_price = float(prices[-1])
        final_prediction = float(predictions[-1])
        percent_change = float(((final_prediction - last_price) / last_price) * 100)
        
        threshold = 0.3 if mode == "15min" else 1.5
        
        if percent_change > threshold and momentum_is_up: signal = "STRONG BUY"
        elif percent_change > threshold and not momentum_is_up: signal = "BUY (Risky)"
        elif percent_change < -threshold and not momentum_is_up: signal = "STRONG SELL"
        elif percent_change < -threshold and momentum_is_up: signal = "SELL (Risky)"
        else: signal = "HOLD"

        last_time = hist.index[-1].to_pydatetime()
        predicted_timestamps = []
        for i in range(1, pred_count + 1):
            if mode == "15min":
                next_time = last_time + datetime.timedelta(minutes=i)
            else:
                next_time = last_time + datetime.timedelta(days=i)
            predicted_timestamps.append(int(next_time.timestamp() * 1000))

        return {
            "ticker": ticker.upper(),
            "signal": signal,
            "change": round(percent_change, 2),
            "past_ohlc": past_ohlc, 
            "predicted_data": [round(float(p), 2) for p in predictions],
            "predicted_timestamps": predicted_timestamps
        }
    except Exception as e:
        return {"error": f"AI Error: {str(e)}"}