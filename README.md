@'
# 🟡 Gold Price Prediction App

This app fetches real-time gold prices using [GoldAPI](https://www.goldapi.io/) and predicts short-term price trends using machine learning. It includes a FastAPI backend and a lightweight dashboard to visualize historical and predicted prices.

## 🚀 Features
- Real-time gold price fetching (XAU/USD)
- CSV storage of price history
- Dashboard with Chart.js
- Prediction endpoint (future-ready)
- Clean FastAPI backend

## 📦 Stack
- FastAPI + Python
- Chart.js + HTML (served from FastAPI)
- pandas, requests, uvicorn

## 📈 Endpoints
- `/predict` – Fetches current gold price and saves to CSV
- `/dashboard` – Interactive price trend chart
- `/` – Welcome message

## ⚙️ Setup
```bash
git clone https://github.com/aliabdulrabali/gold_prediction_app.git
cd gold_prediction_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
