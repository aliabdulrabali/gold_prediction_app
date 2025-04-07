import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import prophet

from config import MODEL_SAVE_PATH, PREDICTION_DAYS
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class GoldPricePredictor:
    """Class for training and using gold price prediction models"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize the predictor with a database manager
        
        Args:
            db_manager: DatabaseManager instance for data access
        """
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'price', 
                    sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for LSTM model training
        
        Args:
            df: DataFrame containing price data
            target_col: Column to predict
            sequence_length: Number of time steps to use for sequence prediction
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        # Make a copy of the dataframe to avoid modifying the original
        data = df.copy()
        
        # Extract the target column
        dataset = data[target_col].values
        dataset = dataset.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape X to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_lstm_model(self, df: pd.DataFrame, target_col: str = 'price', 
                        sequence_length: int = 10, epochs: int = 50, 
                        batch_size: int = 32) -> Dict[str, Any]:
        """
        Train an LSTM model for gold price prediction
        
        Args:
            df: DataFrame containing price data
            target_col: Column to predict
            sequence_length: Number of time steps to use for sequence prediction
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing model, scaler, and metrics
        """
        try:
            logger.info("Preparing data for LSTM model training...")
            X_train, X_test, y_train, y_test, scaler = self.prepare_data(
                df, target_col, sequence_length
            )
            
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train the model
            logger.info(f"Training LSTM model with {len(X_train)} samples...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Inverse transform to get actual prices
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_actual = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, y_pred_actual)
            
            logger.info(f"LSTM Model Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Save model and scaler
            model_info = {
                'model': model,
                'scaler': scaler,
                'sequence_length': sequence_length,
                'target_col': target_col,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'history': history.history
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return None
    
    def train_linear_regression(self, df: pd.DataFrame, target_col: str = 'price',
                               features: List[str] = None) -> Dict[str, Any]:
        """
        Train a linear regression model for gold price prediction
        
        Args:
            df: DataFrame containing price data
            target_col: Column to predict
            features: List of feature columns to use (if None, uses default features)
            
        Returns:
            Dictionary containing model, scaler, and metrics
        """
        try:
            # Make a copy of the dataframe to avoid modifying the original
            data = df.copy()
            
            # Default features if none provided
            if features is None:
                features = ['open', 'high', 'low', 'close']
                
            # Filter out features not in the dataframe
            features = [f for f in features if f in data.columns and f != target_col]
            
            if not features:
                logger.error("No valid features for linear regression")
                return None
                
            logger.info(f"Training linear regression with features: {features}")
            
            # Prepare features and target
            X = data[features].values
            y = data[target_col].values
            
            # Scale the data
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            
            X_scaled = X_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Split into train and test sets (80% train, 20% test)
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Inverse transform to get actual prices
            y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_actual = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, y_pred_actual)
            
            logger.info(f"Linear Regression Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Save model and scaler
            model_info = {
                'model': model,
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'features': features,
                'target_col': target_col,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training linear regression model: {str(e)}")
            return None
    
    def train_random_forest(self, df: pd.DataFrame, target_col: str = 'price',
                           features: List[str] = None, n_estimators: int = 100) -> Dict[str, Any]:
        """
        Train a random forest model for gold price prediction
        
        Args:
            df: DataFrame containing price data
            target_col: Column to predict
            features: List of feature columns to use (if None, uses default features)
            n_estimators: Number of trees in the forest
            
        Returns:
            Dictionary containing model, scaler, and metrics
        """
        try:
            # Make a copy of the dataframe to avoid modifying the original
            data = df.copy()
            
            # Default features if none provided
            if features is None:
                features = ['open', 'high', 'low', 'close']
                
            # Filter out features not in the dataframe
            features = [f for f in features if f in data.columns and f != target_col]
            
            if not features:
                logger.error("No valid features for random forest")
                return None
                
            logger.info(f"Training random forest with features: {features}")
            
            # Prepare features and target
            X = data[features].values
            y = data[target_col].values
            
            # Scale the data
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            
            X_scaled = X_scaler.fit_transform(X)
            y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Split into train and test sets (80% train, 20% test)
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
            
            # Train the model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            
            # Inverse transform to get actual prices
            y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_actual = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_actual, y_pred_actual)
            
            logger.info(f"Random Forest Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Save model and scaler
            model_info = {
                'model': model,
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'features': features,
                'target_col': target_col,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training random forest model: {str(e)}")
            return None
    
    def train_prophet_model(self, df: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
        """
        Train a Prophet model for gold price prediction
        
        Args:
            df: DataFrame containing price data
            target_col: Column to predict
            
        Returns:
            Dictionary containing model and metrics
        """
        try:
            # Make a copy of the dataframe to avoid modifying the original
            data = df.copy()
            
            # Prophet requires columns named 'ds' and 'y'
            prophet_df = pd.DataFrame()
            prophet_df['ds'] = data['date']
            prophet_df['y'] = data[target_col]
            
            # Split into train and test sets (80% train, 20% test)
            train_size = int(len(prophet_df) * 0.8)
            train_df = prophet_df[:train_size]
            test_df = prophet_df[train_size:]
            
            logger.info(f"Training Prophet model with {len(train_df)} samples...")
            
            # Train the model
            model = prophet.Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(train_df)
            
            # Make predictions on test set
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            
            # Extract predictions for test period
            y_pred = forecast.iloc[-len(test_df):]['yhat'].values
            y_test = test_df['y'].values
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Prophet Model Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            # Save model info
            model_info = {
                'model': model,
                'target_col': target_col,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return None
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all prediction models
        
        Returns:
            Dictionary containing all trained models and their info
        """
        # Get data for training
        df = self.db_manager.get_data_for_training()
        
        if df.empty:
            logger.error("No data available for training models")
            return {}
            
        logger.info(f"Training models with {len(df)} data points")
        
        # Train models
        models = {}
        
        # Linear Regression
        lr_model = self.train_linear_regression(df)
        if lr_model:
            models['linear_regression'] = lr_model
            
        # Random Forest
        rf_model = self.train_random_forest(df)
        if rf_model:
            models['random_forest'] = rf_model
            
        # LSTM
        lstm_model = self.train_lstm_model(df)
        if lstm_model:
            models['lstm'] = lstm_model
            
        # Prophet
        prophet_model = self.train_prophet_model(df)
        if prophet_model:
            models['prophet'] = prophet_model
            
        self.models = models
        return models
    
    def save_models(self, models_dir: str = MODEL_SAVE_PATH) -> bool:
        """
        Save trained models to disk
        
        Args:
            models_dir: Directory to save models
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name, model_info in self.models.items():
                if model_name == 'lstm':
                    # Save Keras model
                    model_path = os.path.join(models_dir, f"{model_name}_model")
                    model_info['model'].save(model_path)
                    
                    # Save other info
                    info_path = os.path.join(models_dir, f"{model_name}_info.joblib")
                    info_to_save = {k: v for k, v in model_info.items() if k != 'model'}
                    joblib.dump(info_to_save, info_path)
                    
                elif model_name == 'prophet':
                    # Prophet models need special handling
                    model_path = os.path.join(models_dir, f"{model_name}_model.json")
                    with open(model_path, 'w') as f:
                        f.write(model_info['model'].to_json())
                        
                    # Save other info
                    info_path = os.path.join(models_dir, f"{model_name}_info.joblib")
                    info_to_save = {k: v for k, v in model_info.items() if k != 'model'}
                    joblib.dump(info_to_save, info_path)
                    
                else:
                    # Save scikit-learn models
                    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
                    joblib.dump(model_info, model_path)
                    
            logger.info(f"Saved {len(self.models)} models to {models_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, models_dir: str = MODEL_SAVE_PATH) -> Dict[str, Dict[str, Any]]:
        """
        Load trained models from disk
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            Dictionary containing loaded models
        """
        try:
            models = {}
            
            # Check for linear regression model
            lr_path = os.path.join(models_dir, "linear_regression_model.joblib")
            if os.path.exists(lr_path):
                models['linear_regression'] = joblib.load(lr_path)
                logger.info("Loaded linear regression model")
                
            # Check for random forest model
            rf_path = os.path.join(models_dir, "random_forest_model.joblib")
            if os.path.exists(rf_path):
                models['random_forest'] = joblib.load(rf_path)
                logger.info("Loaded random forest model")
                
            # Check for LSTM model
            lstm_model_path = os.path.join(models_dir, "lstm_model")
            lstm_info_path = os.path.join(models_dir, "lstm_info.joblib")
            if os.path.exists(lstm_model_path) and os.path.exists(lstm_info_path):
                # Load Keras model
                lstm_model = tf.keras.models.load_model(lstm_model_path)
                
                # Load other info
                lstm_info = joblib.load(lstm_info_path)
                
                # Combine model and info
                models['lstm'] = {**lstm_info, 'model': lstm_model}
                logger.info("Loaded LSTM model")
                
            # Check for Prophet model
            prophet_model_path = os.path.join(models_dir, "prophet_model.json")
            prophet_info_path = os.path.join(models_dir, "prophet_info.joblib")
            if os.path.exists(prophet_model_path) and os.path.exists(prophet_info_path):
                # Load Prophet model
                with open(prophet_model_path, 'r') as f:
                    prophet_model = prophet.Prophet.from_json(f.read())
                    
                # Load other info
                prophet_info = joblib.load(prophet_info_path)
                
                # Combine model and info
                models['prophet'] = {**prophet_info, 'model': prophet_model}
                logger.info("Loaded Prophet model")
                
            self.models = models
            return models
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return {}
    
    def predict_with_lstm(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Make predictions using the LSTM model
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of dictionaries containing predictions
        """
        try:
            if 'lstm' not in self.models:
                logger.error("LSTM model not found")
                return []
                
            model_info = self.models['lstm']
            model = model_info['model']
            scaler = model_info['scaler']
            sequence_length = model_info['sequence_length']
            target_col = model_info['target_col']
            
            # Get recent data for prediction
            df = self.db_manager.get_data_for_training(days=sequence_length + days_ahead)
            
            if df.empty or len(df) < sequence_length:
                logger.error(f"Not enough data for LSTM prediction (need at least {sequence_length} points)")
                return []
                
            # Extract the target column
            data = df[target_col].values[-sequence_length:].reshape(-1, 1)
            
            # Scale the data
            scaled_data = scaler.transform(data)
            
            # Create input sequence
            X = np.array([scaled_data[:, 0]])
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Make predictions
            predictions = []
            last_date = df['date'].iloc[-1]
            
            for i in range(days_ahead):
                # Predict next value
                pred_scaled = model.predict(X)
                
                # Inverse transform to get actual price
                pred_value = scaler.inverse_transform(pred_scaled)[0][0]
                
                # Calculate prediction date
                pred_date = last_date + timedelta(days=i+1)
                
                # Add to predictions
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': float(pred_value),
                    'model': 'lstm'
                })
                
                # Update input sequence for next prediction
                X = np.append(X[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {str(e)}")
            return []
    
    def predict_with_linear_regression(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Make predictions using the linear regression model
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of dictionaries containing predictions
        """
        try:
            if 'linear_regression' not in self.models:
                logger.error("Linear regression model not found")
                return []
                
            model_info = self.models['linear_regression']
            model = model_info['model']
            X_scaler = model_info['X_scaler']
            y_scaler = model_info['y_scaler']
            features = model_info['features']
            target_col = model_info['target_col']
            
            # Get recent data for prediction
            df = self.db_manager.get_data_for_training(days=30)  # Get enough data for feature calculation
            
            if df.empty:
                logger.error("Not enough data for linear regression prediction")
                return []
                
            # Extract the latest features
            latest_features = df[features].iloc[-1].values.reshape(1, -1)
            
            # Scale the features
            latest_features_scaled = X_scaler.transform(latest_features)
            
            # Make predictions
            predictions = []
            last_date = df['date'].iloc[-1]
            
            for i in range(days_ahead):
                # Predict next value
                pred_scaled = model.predict(latest_features_scaled)
                
                # Inverse transform to get actual price
                pred_value = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                # Calculate prediction date
                pred_date = last_date + timedelta(days=i+1)
                
                # Add to predictions
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': float(pred_value),
                    'model': 'linear_regression'
                })
                
                # For simplicity, we use the same features for all future predictions
                # In a more sophisticated model, we would update the features based on previous predictions
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making linear regression predictions: {str(e)}")
            return []
    
    def predict_with_random_forest(self, days_ahead: int = 1) -> List[Dict[str, Any]]:
        """
        Make predictions using the random forest model
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of dictionaries containing predictions
        """
        try:
            if 'random_forest' not in self.models:
                logger.error("Random forest model not found")
                return []
                
            model_info = self.models['random_forest']
            model = model_info['model']
            X_scaler = model_info['X_scaler']
            y_scaler = model_info['y_scaler']
            features = model_info['features']
            target_col = model_info['target_col']
            
            # Get recent data for prediction
            df = self.db_manager.get_data_for_training(days=30)  # Get enough data for feature calculation
            
            if df.empty:
                logger.error("Not enough data for random forest prediction")
                return []
                
            # Extract the latest features
            latest_features = df[features].iloc[-1].values.reshape(1, -1)
            
            # Scale the features
            latest_features_scaled = X_scaler.transform(latest_features)
            
            # Make predictions
            predictions = []
            last_date = df['date'].iloc[-1]
            
            for i in range(days_ahead):
                # Predict next value
                pred_scaled = model.predict(latest_features_scaled)
                
                # Inverse transform to get actual price
                pred_value = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                # Calculate prediction date
                pred_date = last_date + timedelta(days=i+1)
                
                # Add to predictions
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': float(pred_value),
                    'model': 'random_forest'
                })
                
                # For simplicity, we use the same features for all future predictions
                # In a more sophisticated model, we would update the features based on previous predictions
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making random forest predictions: {str(e)}")
            return []
    
    def predict_with_prophet(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Make predictions using the Prophet model
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of dictionaries containing predictions
        """
        try:
            if 'prophet' not in self.models:
                logger.error("Prophet model not found")
                return []
                
            model_info = self.models['prophet']
            model = model_info['model']
            target_col = model_info['target_col']
            
            # Make future dataframe for prediction
            future = model.make_future_dataframe(periods=days_ahead)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract predictions for future days
            predictions = []
            for i in range(days_ahead):
                idx = len(forecast) - days_ahead + i
                pred_date = forecast['ds'].iloc[idx]
                pred_value = forecast['yhat'].iloc[idx]
                
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': float(pred_value),
                    'model': 'prophet',
                    'lower_bound': float(forecast['yhat_lower'].iloc[idx]),
                    'upper_bound': float(forecast['yhat_upper'].iloc[idx])
                })
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {str(e)}")
            return []
    
    def predict_all_models(self, days_ahead: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Make predictions using all available models
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary containing predictions from all models
        """
        predictions = {}
        
        # Load models if not already loaded
        if not self.models:
            self.load_models()
            
        # Make predictions with each model
        if 'lstm' in self.models:
            lstm_preds = self.predict_with_lstm(days_ahead)
            if lstm_preds:
                predictions['lstm'] = lstm_preds
                
        if 'linear_regression' in self.models:
            lr_preds = self.predict_with_linear_regression(days_ahead)
            if lr_preds:
                predictions['linear_regression'] = lr_preds
                
        if 'random_forest' in self.models:
            rf_preds = self.predict_with_random_forest(days_ahead)
            if rf_preds:
                predictions['random_forest'] = rf_preds
                
        if 'prophet' in self.models:
            prophet_preds = self.predict_with_prophet(days_ahead)
            if prophet_preds:
                predictions['prophet'] = prophet_preds
                
        return predictions
    
    def get_ensemble_prediction(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get ensemble prediction by averaging predictions from all models
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            List of dictionaries containing ensemble predictions
        """
        # Get predictions from all models
        all_predictions = self.predict_all_models(days_ahead)
        
        if not all_predictions:
            logger.error("No predictions available for ensemble")
            return []
            
        # Organize predictions by date
        predictions_by_date = {}
        
        for model_name, preds in all_predictions.items():
            for pred in preds:
                date = pred['date']
                if date not in predictions_by_date:
                    predictions_by_date[date] = []
                    
                predictions_by_date[date].append({
                    'model': model_name,
                    'price': pred['price']
                })
                
        # Calculate ensemble predictions
        ensemble_predictions = []
        
        for date, preds in sorted(predictions_by_date.items()):
            if len(preds) > 0:
                # Calculate average price
                avg_price = sum(p['price'] for p in preds) / len(preds)
                
                # Calculate min and max prices
                min_price = min(p['price'] for p in preds)
                max_price = max(p['price'] for p in preds)
                
                # Add to ensemble predictions
                ensemble_predictions.append({
                    'date': date,
                    'price': float(avg_price),
                    'min_price': float(min_price),
                    'max_price': float(max_price),
                    'model': 'ensemble',
                    'models_used': [p['model'] for p in preds]
                })
                
        return ensemble_predictions


if __name__ == "__main__":
    # Example usage
    from database import DatabaseManager
    
    db_manager = DatabaseManager()
    predictor = GoldPricePredictor(db_manager)
    
    # Check if we have enough data
    df = db_manager.get_data_for_training()
    
    if len(df) > 30:
        print(f"Training models with {len(df)} data points...")
        
        # Train models
        models = predictor.train_all_models()
        
        # Save models
        predictor.save_models()
        
        # Make predictions
        for days in PREDICTION_DAYS:
            print(f"\nPredictions for next {days} days:")
            
            ensemble_preds = predictor.get_ensemble_prediction(days)
            
            for pred in ensemble_preds:
                print(f"Date: {pred['date']}, Price: ${pred['price']:.2f} (Range: ${pred['min_price']:.2f} - ${pred['max_price']:.2f})")
    else:
        print(f"Not enough data for training (have {len(df)} points, need at least 30)")
