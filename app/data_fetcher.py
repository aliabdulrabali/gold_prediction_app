import requests
import pandas as pd
import datetime
import time
import os
import logging
from typing import Dict, Any, Optional

from config import (
    GOLD_API_KEY,
    GOLD_API_BASE_URL,
    DEFAULT_CURRENCY,
    DEFAULT_METAL,
    DATA_PATH,
    HISTORICAL_DATA_FILE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoldDataFetcher:
    """
    Class to fetch gold price data from GoldAPI.io
    """
    def __init__(self, api_key: str = GOLD_API_KEY, base_url: str = GOLD_API_BASE_URL):
        """
        Initialize the GoldDataFetcher with API credentials
        
        Args:
            api_key: API key for GoldAPI.io
            base_url: Base URL for the GoldAPI
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'x-access-token': api_key,
            'Content-Type': 'application/json'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_PATH, exist_ok=True)
        
    def fetch_current_price(self, metal: str = DEFAULT_METAL, currency: str = DEFAULT_CURRENCY) -> Dict[str, Any]:
        """
        Fetch current gold price from the API
        
        Args:
            metal: Metal symbol (XAU for gold)
            currency: Currency code (USD, EUR, etc.)
            
        Returns:
            Dictionary containing the current gold price data
        """
        try:
            url = f"{self.base_url}/{metal}/{currency}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched {metal} price in {currency}")
                return data
            else:
                logger.error(f"Failed to fetch data: {response.status_code} - {response.text}")
                return {"error": f"API request failed with status code {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching gold price: {str(e)}")
            return {"error": str(e)}
    
    def fetch_historical_data(self, 
                             metal: str = DEFAULT_METAL, 
                             currency: str = DEFAULT_CURRENCY,
                             date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch historical gold price for a specific date
        
        Args:
            metal: Metal symbol (XAU for gold)
            currency: Currency code (USD, EUR, etc.)
            date: Date in format 'YYYYMMDD' (if None, returns current price)
            
        Returns:
            Dictionary containing the historical gold price data
        """
        try:
            if date:
                url = f"{self.base_url}/{metal}/{currency}/{date}"
            else:
                url = f"{self.base_url}/{metal}/{currency}"
                
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched historical {metal} price in {currency} for {date if date else 'today'}")
                return data
            else:
                logger.error(f"Failed to fetch historical data: {response.status_code} - {response.text}")
                return {"error": f"API request failed with status code {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching historical gold price: {str(e)}")
            return {"error": str(e)}
    
    def save_to_csv(self, data: Dict[str, Any], filename: str = HISTORICAL_DATA_FILE, append: bool = True) -> bool:
        """
        Save fetched data to CSV file
        
        Args:
            data: Gold price data dictionary
            filename: Path to save the CSV file
            append: Whether to append to existing file or create new
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Check if data contains error
            if "error" in data:
                logger.error(f"Cannot save data with error: {data['error']}")
                return False
                
            # Extract relevant fields
            timestamp = data.get('timestamp', int(time.time()))
            price_data = {
                'timestamp': timestamp,
                'date': datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                'time': datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S'),
                'metal': data.get('metal', DEFAULT_METAL),
                'currency': data.get('currency', DEFAULT_CURRENCY),
                'price': data.get('price', None),
                'bid': data.get('bid', None),
                'ask': data.get('ask', None),
                'high': data.get('high_price', None),
                'low': data.get('low_price', None),
                'open': data.get('open_price', None),
                'close': data.get('prev_close_price', None),
                'ch': data.get('ch', None),
                'chp': data.get('chp', None),
            }
            
            df = pd.DataFrame([price_data])
            
            # Check if file exists and append mode is enabled
            if os.path.exists(filename) and append:
                existing_df = pd.read_csv(filename)
                # Avoid duplicates by checking timestamp
                if timestamp not in existing_df['timestamp'].values:
                    df = pd.concat([existing_df, df], ignore_index=True)
                    logger.info(f"Appended new data to {filename}")
                else:
                    logger.info(f"Data for timestamp {timestamp} already exists in {filename}")
                    return True
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            logger.info(f"Successfully saved data to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
    
    def fetch_and_save_current_price(self, metal: str = DEFAULT_METAL, currency: str = DEFAULT_CURRENCY) -> Dict[str, Any]:
        """
        Fetch current price and save to CSV in one operation
        
        Args:
            metal: Metal symbol (XAU for gold)
            currency: Currency code (USD, EUR, etc.)
            
        Returns:
            The fetched data dictionary
        """
        data = self.fetch_current_price(metal, currency)
        if "error" not in data:
            self.save_to_csv(data)
        return data
    
    def backfill_historical_data(self, days: int = 30, metal: str = DEFAULT_METAL, currency: str = DEFAULT_CURRENCY) -> bool:
        """
        Backfill historical data for a specified number of days
        
        Args:
            days: Number of days to backfill
            metal: Metal symbol (XAU for gold)
            currency: Currency code (USD, EUR, etc.)
            
        Returns:
            Boolean indicating overall success
        """
        success = True
        today = datetime.datetime.now()
        
        for i in range(days):
            # GoldAPI has rate limits, so we need to sleep between requests
            if i > 0:
                time.sleep(1)  # Sleep for 1 second between requests
                
            date = today - datetime.timedelta(days=i)
            date_str = date.strftime('%Y%m%d')
            
            logger.info(f"Fetching historical data for {date_str}")
            data = self.fetch_historical_data(metal, currency, date_str)
            
            if "error" not in data:
                if not self.save_to_csv(data):
                    success = False
            else:
                logger.warning(f"Skipping date {date_str} due to error: {data.get('error')}")
                success = False
                
        return success


if __name__ == "__main__":
    # Example usage
    fetcher = GoldDataFetcher()
    
    # Fetch and save current price
    current_data = fetcher.fetch_and_save_current_price()
    print(f"Current Gold Price: {current_data.get('price', 'N/A')} {current_data.get('currency', 'USD')}")
    
    # Backfill 7 days of historical data
    print("Backfilling historical data for the past 7 days...")
    fetcher.backfill_historical_data(days=7)
    
    print(f"Data saved to {HISTORICAL_DATA_FILE}")
