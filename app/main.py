import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import DatabaseManager
from app.data_fetcher import GoldDataFetcher
from app.predictor import GoldPricePredictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

# FastAPI app config
app = FastAPI(
    title="Gold Price Predictor",
    description="API for predicting short-term gold price movements.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Services
db_manager = DatabaseManager()
data_fetcher = GoldDataFetcher()
predictor = GoldPricePredictor(db_manager)

# ========================
#          ROUTES
# ========================

@app.get("/")
def root():
    return {"message": "Gold Price Predictor API"}


@app.get("/current")
def get_current_price():
    try:
        latest = db_manager.get_latest_price()
        if latest and datetime.now() - datetime.fromisoformat(latest['created_at']) < timedelta(hours=1):
            return {"success": True, "data": latest, "source": "database"}
        data = data_fetcher.fetch_and_save_current_price()
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"])
        db_manager.add_price_data(data)
        return {"success": True, "data": data, "source": "api"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history(
    limit: int = Query(100),
    offset: int = Query(0),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    try:
        records = db_manager.get_price_data(limit, offset, start_date, end_date)
        return {"success": True, "count": len(records), "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict")
def predict_prices(
    days: int = Query(7),
    model: str = Query("ensemble")
):
    try:
        if not predictor.models:
            predictor.load_models()
        if not predictor.models:
            df = db_manager.get_data_for_training()
            if len(df) < 30:
                raise HTTPException(status_code=400, detail="Not enough historical data for prediction. Need at least 30 data points.")
            predictor.train_all_models()
            predictor.save_models()

        if model == "ensemble":
            result = predictor.get_ensemble_prediction(days)
        elif model in predictor.models:
            predict_func = getattr(predictor, f"predict_with_{model}", None)
            result = predict_func(days) if predict_func else []
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model}' not available.")
        return {"success": True, "model": model, "days": days, "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def train_models():
    try:
        df = db_manager.get_data_for_training()
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Not enough historical data for training. Need at least 30 data points.")
        models = predictor.train_all_models()
        predictor.save_models()
        return {"success": True, "message": "Models trained", "models": list(models.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
def get_dashboard():
    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, "..", "frontend", "index.html")
    return FileResponse(index_path)

# ========================
#       STARTUP HOOK
# ========================

@app.on_event("startup")
async def startup_event():
    base_dir = os.path.dirname(__file__)
    frontend_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend"))
    os.makedirs(frontend_dir, exist_ok=True)

    index_path = os.path.join(frontend_dir, "index.html")
    css_path = os.path.join(frontend_dir, "styles.css")
    js_path = os.path.join(frontend_dir, "app.js")

    # Create index.html
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("<html><head><title>Gold Dashboard</title></head><body><h1>Gold Price Dashboard</h1></body></html>")

    # Create dummy CSS
    if not os.path.exists(css_path):
        with open(css_path, "w") as f:
            f.write("body { font-family: Arial; background: #f4f4f4; }")

    # Create dummy JS
    if not os.path.exists(js_path):
        with open(js_path, "w") as f:
            f.write("console.log('Dashboard loaded');")

    # Auto-fetch data on cold start if empty
    latest = db_manager.get_latest_price()
    if not latest:
        logger.info("[Startup] No price data found. Fetching and importing historical data...")
        data_fetcher.backfill_historical_data(days=30)
        db_manager.import_from_csv()
        logger.info("[Startup] Historical data saved to DB.")

# Serve static assets
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="frontend")
