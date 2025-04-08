import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GOLD_API_KEY = os.getenv("GOLD_API_KEY", "")
GOLD_API_BASE_URL = "https://www.goldapi.io/api"

# Default currency and metal for gold price
DEFAULT_CURRENCY = "USD"
DEFAULT_METAL = "XAU"  # Gold

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./gold_prices.db")

# Model Configuration
MODEL_SAVE_PATH = "../models/"
PREDICTION_DAYS = [1, 3, 7, 14, 30]  # Prediction horizons in days

# Data Configuration
DATA_PATH = "../data/"
HISTORICAL_DATA_FILE = os.path.join(DATA_PATH, "historical_gold_prices.csv")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
