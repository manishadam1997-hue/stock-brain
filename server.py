from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# This allows your Flutter app to talk to the server safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    symbol: str

# FIXES THE 404 ERROR: This is the "Home" page for Render's health check
@app.get("/")
def home():
    return {"status": "Online", "message": "Stock Brain AI is running!"}

# This is the "Door" your Flutter app will knock on
@app.post("/predict")
def predict_stock(request: StockRequest):
    try:
        stock = yf.Ticker(request.symbol)
        df = stock.history(period="5d")

        if df.empty:
            return {"error": "Invalid symbol", "status": "failed"}

        current_price = df['Close'].iloc[-1]
        # AI Logic: Predicting a slight 2% movement
        prediction = current_price * 1.02 

        return {
            "symbol": request.symbol.upper(),
            "current_price": round(float(current_price), 2),
            "prediction": round(float(prediction), 2),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}