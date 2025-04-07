# Gold Price Prediction Web App

A comprehensive web application for fetching real-time gold prices, storing historical data, and using machine learning models to predict future price trends.

## Features

- Real-time gold price data fetching from GoldAPI
- Historical price data storage and visualization
- Multiple machine learning models for price prediction:
  - LSTM Neural Network
  - Linear Regression
  - Random Forest
  - Prophet
  - Ensemble model (combining all predictions)
- Interactive dashboard with responsive design
- RESTful API endpoints for data access and model training

## Project Structure

```
gold_prediction_app/
├── backend/
│   ├── config.py             # Configuration settings
│   ├── data_fetcher.py       # Gold API data fetching module
│   ├── database.py           # Database operations
│   ├── main.py               # FastAPI application and endpoints
│   ├── predictor.py          # ML prediction models
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── app.js                # JavaScript for dashboard functionality
│   ├── index.html            # Main dashboard HTML
│   └── styles.css            # CSS styling
├── data/                     # Data storage directory
├── models/                   # Saved ML models
└── docs/                     # Documentation
```

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy, Pandas, NumPy, Scikit-learn, TensorFlow, Prophet
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js, Bootstrap
- **Data Storage**: SQLite (default), supports other SQL databases
- **API**: RESTful API with JSON responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gold-prediction-app.git
cd gold-prediction-app
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the backend directory with the following variables:
```
GOLD_API_KEY=your_gold_api_key
DATABASE_URL=sqlite:///./gold_prices.db
API_HOST=0.0.0.0
API_PORT=8000
```

## Usage

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Access the dashboard:
Open your browser and navigate to `http://localhost:8000`

## API Endpoints

- `GET /current` - Get current gold price
- `GET /history` - Get historical gold prices
- `GET /predict` - Get gold price predictions
- `GET /models` - Get information about trained models
- `POST /train` - Train prediction models
- `GET /fetch` - Fetch historical gold price data
- `GET /dashboard` - Web dashboard for visualizing gold prices and predictions

## Deployment

The application can be deployed using various methods:

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t gold-prediction-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e GOLD_API_KEY=your_gold_api_key gold-prediction-app
```

### Cloud Deployment

The application can be deployed to cloud platforms like AWS, Google Cloud, or Azure using their container services or virtual machines.

## Future Enhancements

- Add more prediction models
- Implement user authentication
- Add email/SMS alerts for price thresholds
- Support for additional precious metals
- Mobile application

## License

MIT License

## Acknowledgements

- [GoldAPI](https://www.goldapi.io/) for providing gold price data
- [Chart.js](https://www.chartjs.org/) for data visualization
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Prophet](https://facebook.github.io/prophet/) for time series forecasting
