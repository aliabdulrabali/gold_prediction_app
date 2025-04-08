import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI`nfrom dotenv import load_dotenv`nimport os`nload_dotenv()`nGOLD_API_KEY = os.getenv("GOLD_API_KEY"), HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import logging
import pandas as pd
import json

from data_fetcher import GoldDataFetcher
from database import DatabaseManager
from predictor import GoldPricePredictor
from config import PREDICTION_DAYS, DATA_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gold Price Prediction API",
    description="API for fetching gold prices and making price predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
data_fetcher = GoldDataFetcher()
predictor = GoldPricePredictor(db_manager)

# Load prediction models if they exist
predictor.load_models()

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Gold Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/current": "Get current gold price",
            "/history": "Get historical gold prices",
            "/predict": "Get gold price predictions",
            "/models": "Get information about trained models",
            "/train": "Train prediction models",
            "/dashboard": "Web dashboard for visualizing gold prices and predictions"
        }
    }

@app.get("/current")
async def get_current_price():
    """Get the current gold price"""
    try:
        # First check if we have a recent price in the database
        latest_price = db_manager.get_latest_price()
        
        # If we have a price from the last hour, use it
        if latest_price and datetime.now() - datetime.fromisoformat(latest_price['created_at']) < timedelta(hours=1):
            return {
                "success": True,
                "data": latest_price,
                "source": "database"
            }
        
        # Otherwise fetch a new price
        current_data = data_fetcher.fetch_and_save_current_price()
        
        if "error" in current_data:
            raise HTTPException(status_code=500, detail=f"Error fetching current price: {current_data['error']}")
        
        # Add to database
        db_manager.add_price_data({
            'timestamp': current_data.get('timestamp', int(datetime.now().timestamp())),
            'date': datetime.fromtimestamp(current_data.get('timestamp', int(datetime.now().timestamp()))).strftime('%Y-%m-%d'),
            'time': datetime.fromtimestamp(current_data.get('timestamp', int(datetime.now().timestamp()))).strftime('%H:%M:%S'),
            'metal': current_data.get('metal', 'XAU'),
            'currency': current_data.get('currency', 'USD'),
            'price': current_data.get('price'),
            'bid': current_data.get('bid'),
            'ask': current_data.get('ask'),
            'high': current_data.get('high_price'),
            'low': current_data.get('low_price'),
            'open': current_data.get('open_price'),
            'close': current_data.get('prev_close_price'),
            'ch': current_data.get('ch'),
            'chp': current_data.get('chp')
        })
        
        return {
            "success": True,
            "data": current_data,
            "source": "api"
        }
    except Exception as e:
        logger.error(f"Error in get_current_price: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    start_date: Optional[str] = Query(None, description="Start date in format YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date in format YYYY-MM-DD")
):
    """Get historical gold prices"""
    try:
        # Get data from database
        history = db_manager.get_price_data(limit, offset, start_date, end_date)
        
        return {
            "success": True,
            "count": len(history),
            "data": history
        }
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict_prices(
    days: int = Query(7, description="Number of days to predict ahead"),
    model: str = Query("ensemble", description="Model to use for prediction (lstm, linear_regression, random_forest, prophet, ensemble)")
):
    """Get gold price predictions"""
    try:
        # Check if we have trained models
        if not predictor.models:
            # Try to load models
            predictor.load_models()
            
            # If still no models, train them
            if not predictor.models:
                # Check if we have enough data
                df = db_manager.get_data_for_training()
                if len(df) < 30:
                    raise HTTPException(
                        status_code=400, 
                        detail="Not enough historical data for prediction. Need at least 30 data points."
                    )
                
                # Train models
                predictor.train_all_models()
                predictor.save_models()
        
        # Make predictions
        if model == "ensemble":
            predictions = predictor.get_ensemble_prediction(days)
        elif model == "lstm" and "lstm" in predictor.models:
            predictions = predictor.predict_with_lstm(days)
        elif model == "linear_regression" and "linear_regression" in predictor.models:
            predictions = predictor.predict_with_linear_regression(days)
        elif model == "random_forest" and "random_forest" in predictor.models:
            predictions = predictor.predict_with_random_forest(days)
        elif model == "prophet" and "prophet" in predictor.models:
            predictions = predictor.predict_with_prophet(days)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model}' not available. Available models: {list(predictor.models.keys())}"
            )
        
        return {
            "success": True,
            "model": model,
            "days": days,
            "predictions": predictions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_prices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get information about trained models"""
    try:
        # Load models if not already loaded
        if not predictor.models:
            predictor.load_models()
        
        # Get model information
        models_info = {}
        for model_name, model_info in predictor.models.items():
            # Extract metrics and other relevant info
            models_info[model_name] = {
                "metrics": model_info.get("metrics", {}),
                "features": model_info.get("features", []),
                "target_col": model_info.get("target_col", "price")
            }
        
        return {
            "success": True,
            "models": models_info
        }
    except Exception as e:
        logger.error(f"Error in get_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models():
    """Train prediction models"""
    try:
        # Check if we have enough data
        df = db_manager.get_data_for_training()
        if len(df) < 30:
            raise HTTPException(
                status_code=400, 
                detail="Not enough historical data for training. Need at least 30 data points."
            )
        
        # Train models
        models = predictor.train_all_models()
        
        # Save models
        predictor.save_models()
        
        # Return model metrics
        models_info = {}
        for model_name, model_info in models.items():
            models_info[model_name] = {
                "metrics": model_info.get("metrics", {})
            }
        
        return {
            "success": True,
            "message": f"Successfully trained {len(models)} models",
            "models": models_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch")
async def fetch_data(days: int = Query(7, description="Number of days of historical data to fetch")):
    """Fetch historical gold price data"""
    try:
        # Backfill historical data
        success = data_fetcher.backfill_historical_data(days=days)
        
        # Import data to database
        db_success = db_manager.import_from_csv()
        
        return {
            "success": success and db_success,
            "message": f"Fetched and imported {days} days of historical data"
        }
    except Exception as e:
        logger.error(f"Error in fetch_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_dashboard():
    """Serve the dashboard HTML page"""
    return FileResponse("../frontend/index.html")

# Mount static files for frontend
@app.on_event("startup")
async def startup_event():
    # Create frontend directory if it doesn't exist
    os.makedirs("../frontend", exist_ok=True)
    
    # Create a simple index.html if it doesn't exist
    index_path = "../frontend/index.html"
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Gold Price Prediction Dashboard</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/luxon@2.0.2"></script>
                <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.0.0"></script>
                <link rel="stylesheet" href="styles.css">
            </head>
            <body>
                <div class="container">
                    <h1>Gold Price Prediction Dashboard</h1>
                    <div class="current-price">
                        <h2>Current Gold Price</h2>
                        <div id="price-display">Loading...</div>
                    </div>
                    <div class="chart-container">
                        <h2>Historical Prices</h2>
                        <canvas id="historyChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>Price Predictions</h2>
                        <div class="model-selector">
                            <label for="model-select">Prediction Model:</label>
                            <select id="model-select">
                                <option value="ensemble">Ensemble (All Models)</option>
                                <option value="lstm">LSTM</option>
                                <option value="linear_regression">Linear Regression</option>
                                <option value="random_forest">Random Forest</option>
                                <option value="prophet">Prophet</option>
                            </select>
                            <label for="days-select">Days to Predict:</label>
                            <select id="days-select">
                                <option value="1">1 Day</option>
                                <option value="3">3 Days</option>
                                <option value="7" selected>7 Days</option>
                                <option value="14">14 Days</option>
                                <option value="30">30 Days</option>
                            </select>
                        </div>
                        <canvas id="predictionChart"></canvas>
                    </div>
                    <div class="actions">
                        <button id="refresh-btn">Refresh Data</button>
                        <button id="train-btn">Train Models</button>
                    </div>
                </div>
                <script src="app.js"></script>
            </body>
            </html>
            """)
    
    # Create a simple CSS file if it doesn't exist
    css_path = "../frontend/styles.css"
    if not os.path.exists(css_path):
        with open(css_path, "w") as f:
            f.write("""
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }
            
            body {
                background-color: #f5f5f5;
                color: #333;
                line-height: 1.6;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            h1 {
                text-align: center;
                margin-bottom: 30px;
                color: #1a237e;
            }
            
            h2 {
                margin-bottom: 15px;
                color: #303f9f;
            }
            
            .current-price {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            #price-display {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1a237e;
            }
            
            .chart-container {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .model-selector {
                display: flex;
                justify-content: center;
                margin-bottom: 15px;
                gap: 15px;
                flex-wrap: wrap;
            }
            
            select {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            
            .actions {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 20px;
            }
            
            button {
                padding: 10px 20px;
                background-color: #3f51b5;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 1rem;
                transition: background-color 0.3s;
            }
            
            button:hover {
                background-color: #303f9f;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                #price-display {
                    font-size: 2rem;
                }
                
                .model-selector {
                    flex-direction: column;
                    align-items: center;
                }
            }
            """)
    
    # Create a simple JavaScript file if it doesn't exist
    js_path = "../frontend/app.js"
    if not os.path.exists(js_path):
        with open(js_path, "w") as f:
            f.write("""
            // DOM Elements
            const priceDisplay = document.getElementById('price-display');
            const historyChart = document.getElementById('historyChart');
            const predictionChart = document.getElementById('predictionChart');
            const modelSelect = document.getElementById('model-select');
            const daysSelect = document.getElementById('days-select');
            const refreshBtn = document.getElementById('refresh-btn');
            const trainBtn = document.getElementById('train-btn');
            
            // Chart instances
            let historyChartInstance = null;
            let predictionChartInstance = null;
            
            // API endpoints
            const API_BASE_URL = '';  // Empty for same origin
            
            // Initialize the dashboard
            async function initDashboard() {
                await fetchCurrentPrice();
                await fetchHistoricalData();
                await fetchPredictions();
                
                // Add event listeners
                modelSelect.addEventListener('change', fetchPredictions);
                daysSelect.addEventListener('change', fetchPredictions);
                refreshBtn.addEventListener('click', refreshData);
                trainBtn.addEventListener('click', trainModels);
            }
            
            // Fetch current gold price
            async function fetchCurrentPrice() {
                try {
                    const response = await fetch(`${API_BASE_URL}/current`);
                    const data = await response.json();
                    
                    if (data.success) {
                        const price = data.data.price;
                        const currency = data.data.currency || 'USD';
                        const timestamp = new Date(data.data.timestamp * 1000).toLocaleString();
                        
                        priceDisplay.innerHTML = `
                            <div class="price-value">$${price.toFixed(2)} ${currency}</div>
                            <div class="price-timestamp">Last updated: ${timestamp}</div>
                        `;
                    } else {
                        priceDisplay.textContent = 'Error fetching current price';
                    }
                } catch (error) {
                    console.error('Error fetching current price:', error);
                    priceDisplay.textContent = 'Error fetching current price';
                }
            }
            
            // Fetch historical gold price data
            async function fetchHistoricalData() {
                try {
                    const response = await fetch(`${API_BASE_URL}/history?limit=90`);
                    const data = await response.json();
                    
                    if (data.success) {
                        renderHistoryChart(data.data);
                    } else {
                        console.error('Error fetching historical data:', data);
                    }
                } catch (error) {
                    console.error('Error fetching historical data:', error);
                }
            }
            
            // Fetch price predictions
            async function fetchPredictions() {
                try {
                    const model = modelSelect.value;
                    const days = daysSelect.value;
                    
                    const response = await fetch(`${API_BASE_URL}/predict?model=${model}&days=${days}`);
                    const data = await response.json();
                    
                    if (data.success) {
                        renderPredictionChart(data.predictions);
                    } else {
                        console.error('Error fetching predictions:', data);
                    }
                } catch (error) {
                    console.error('Error fetching predictions:', error);
                }
            }
            
            // Render historical price chart
            function renderHistoryChart(historyData) {
                // Sort data by date (oldest to newest)
                const sortedData = [...historyData].sort((a, b) => new Date(a.date) - new Date(b.date));
                
                const labels = sortedData.map(item => item.date);
                const prices = sortedData.map(item => item.price);
                
                // Destroy existing chart if it exists
                if (historyChartInstance) {
                    historyChartInstance.destroy();
                }
                
                // Create new chart
                historyChartInstance = new Chart(historyChart, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Gold Price (USD)',
                            data: prices,
                            borderColor: '#ffc107',
                            backgroundColor: 'rgba(255, 193, 7, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Historical Gold Prices (Last 90 Days)'
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'day',
                                    tooltipFormat: 'MMM d, yyyy',
                                    displayFormats: {
                                        day: 'MMM d'
                                    }
                                },
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Price (USD)'
                                }
                            }
                        }
                    }
                });
            }
            
            // Render prediction chart
            function renderPredictionChart(predictions) {
                // Sort predictions by date
                const sortedPredictions = [...predictions].sort((a, b) => new Date(a.date) - new Date(b.date));
                
                const labels = sortedPredictions.map(item => item.date);
                const prices = sortedPredictions.map(item => item.price);
                
                // Check if we have min/max values (ensemble model)
                const hasRange = sortedPredictions[0].hasOwnProperty('min_price') && 
                                sortedPredictions[0].hasOwnProperty('max_price');
                
                const datasets = [{
                    label: 'Predicted Price (USD)',
                    data: prices,
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    fill: !hasRange,
                    tension: 0.1
                }];
                
                // Add range dataset if available
                if (hasRange) {
                    datasets.push({
                        label: 'Prediction Range',
                        data: sortedPredictions.map(item => ({
                            x: item.date,
                            y: item.min_price,
                            y1: item.max_price
                        })),
                        backgroundColor: 'rgba(76, 175, 80, 0.2)',
                        borderWidth: 0,
                        fill: true,
                        tension: 0.1
                    });
                }
                
                // Destroy existing chart if it exists
                if (predictionChartInstance) {
                    predictionChartInstance.destroy();
                }
                
                // Create new chart
                predictionChartInstance = new Chart(predictionChart, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: `Gold Price Predictions (Next ${daysSelect.value} Days)`
                            },
                            tooltip: {
                                mode: 'index',
                                intersect: false
                            }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'day',
                                    tooltipFormat: 'MMM d, yyyy',
                                    displayFormats: {
                                        day: 'MMM d'
                                    }
                                },
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Price (USD)'
                                }
                            }
                        }
                    }
                });
            }
            
            // Refresh all data
            async function refreshData() {
                refreshBtn.disabled = true;
                refreshBtn.textContent = 'Refreshing...';
                
                try {
                    await fetchCurrentPrice();
                    await fetchHistoricalData();
                    await fetchPredictions();
                    
                    refreshBtn.textContent = 'Data Refreshed!';
                    setTimeout(() => {
                        refreshBtn.textContent = 'Refresh Data';
                        refreshBtn.disabled = false;
                    }, 2000);
                } catch (error) {
                    console.error('Error refreshing data:', error);
                    refreshBtn.textContent = 'Error Refreshing';
                    setTimeout(() => {
                        refreshBtn.textContent = 'Refresh Data';
                        refreshBtn.disabled = false;
                    }, 2000);
                }
            }
            
            // Train models
            async function trainModels() {
                trainBtn.disabled = true;
                trainBtn.textContent = 'Training...';
                
                try {
                    const response = await fetch(`${API_BASE_URL}/train`, {
                        method: 'POST'
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        trainBtn.textContent = 'Training Complete!';
                        // Refresh predictions with newly trained models
                        await fetchPredictions();
                    } else {
                        console.error('Error training models:', data);
                        trainBtn.textContent = 'Training Failed';
                    }
                    
                    setTimeout(() => {
                        trainBtn.textContent = 'Train Models';
                        trainBtn.disabled = false;
                    }, 2000);
                } catch (error) {
                    console.error('Error training models:', error);
                    trainBtn.textContent = 'Training Failed';
                    setTimeout(() => {
                        trainBtn.textContent = 'Train Models';
                        trainBtn.disabled = false;
                    }, 2000);
                }
            }
            
            // Initialize the dashboard when the page loads
            document.addEventListener('DOMContentLoaded', initDashboard);
            """)
    
    # Mount the static files
    app.mount("/", StaticFiles(directory="../frontend"), name="frontend")

# Run the application
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    
    # Ensure we have some initial data
    try:
        # Check if we have any data
        if not db_manager.get_latest_price():
            # Fetch current price
            current_data = data_fetcher.fetch_and_save_current_price()
            
            # Import to database
            db_manager.import_from_csv()
            
            # Backfill some historical data
            data_fetcher.backfill_historical_data(days=30)
            db_manager.import_from_csv()
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
    
    # Run the server
    uvicorn.run(app, host=API_HOST, port=API_PORT)
