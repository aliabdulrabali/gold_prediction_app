import os
import pandas as pd

class DatabaseManager:
    def __init__(self):
        self.file_path = "data/prices.csv"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.file_path):
            pd.DataFrame(columns=["timestamp", "date", "price"]).to_csv(self.file_path, index=False)

    def get_latest_price(self):
        df = pd.read_csv(self.file_path)
        if df.empty:
            return None
        latest = df.sort_values("timestamp", ascending=False).iloc[0]
        return latest.to_dict()

    def get_price_data(self, limit=100, offset=0, start_date=None, end_date=None):
        df = pd.read_csv(self.file_path)
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df.iloc[offset:offset+limit].to_dict(orient="records")

    def add_price_data(self, data: dict):
        pd.DataFrame([data]).to_csv(self.file_path, mode="a", header=False, index=False)

    def get_data_for_training(self):
        return pd.read_csv(self.file_path)

    def import_from_csv(self):
        # In case of additional CSV loading logic
        return True
