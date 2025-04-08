import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
GOLD_API_KEY = os.getenv("GOLD_API_KEY")

class GoldDataFetcher:
    BASE_URL = "https://www.goldapi.io/api/XAU/USD"
    HEADERS = {
        "x-access-token": GOLD_API_KEY,
        "Content-Type": "application/json"
    }

    def fetch_and_save_current_price(self):
        try:
            response = requests.get(self.BASE_URL, headers=self.HEADERS)
            data = response.json()
            if "error" in data:
                return {"error": data["error"]}

            df = pd.DataFrame([{
                "timestamp": data.get("timestamp", int(datetime.now().timestamp())),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "price": data["price"]
            }])
            df.to_csv("data/prices.csv", mode="a", header=False, index=False)
            return data
        except Exception as e:
            return {"error": str(e)}

    def backfill_historical_data(self, days: int = 7):
        # Placeholder: simulate backfill
        return True
