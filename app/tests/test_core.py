import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from data_fetcher import GoldDataFetcher
from database import DatabaseManager, GoldPrice
from predictor import GoldPricePredictor

class TestGoldDataFetcher(unittest.TestCase):
    """Test the GoldDataFetcher class"""
    
    @patch('data_fetcher.requests.get')
    def test_fetch_current_price(self, mock_get):
        """Test fetching current price"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'timestamp': 1617753600,
            'metal': 'XAU',
            'currency': 'USD',
            'price': 1750.25,
            'bid': 1749.50,
            'ask': 1751.00,
            'high_price': 1755.30,
            'low_price': 1745.20,
            'open_price': 1748.10,
            'prev_close_price': 1747.80,
            'ch': 2.45,
            'chp': 0.14
        }
        mock_get.return_value = mock_response
        
        # Create fetcher and call method
        fetcher = GoldDataFetcher(api_key='test_key')
        result = fetcher.fetch_current_price()
        
        # Assertions
        self.assertEqual(result['price'], 1750.25)
        self.assertEqual(result['currency'], 'USD')
        self.assertEqual(result['metal'], 'XAU')
        
    @patch('data_fetcher.requests.get')
    def test_fetch_error_handling(self, mock_get):
        """Test error handling in fetch methods"""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = 'Unauthorized'
        mock_get.return_value = mock_response
        
        # Create fetcher and call method
        fetcher = GoldDataFetcher(api_key='invalid_key')
        result = fetcher.fetch_current_price()
        
        # Assertions
        self.assertIn('error', result)
        self.assertIn('401', result['error'])

class TestDatabaseManager(unittest.TestCase):
    """Test the DatabaseManager class"""
    
    def setUp(self):
        """Set up test database"""
        self.db_manager = DatabaseManager(db_url='sqlite:///:memory:')
        
    def test_add_price_data(self):
        """Test adding price data to database"""
        # Create test data
        test_data = {
            'timestamp': int(datetime.now().timestamp()),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'metal': 'XAU',
            'currency': 'USD',
            'price': 1750.25,
            'bid': 1749.50,
            'ask': 1751.00,
            'high': 1755.30,
            'low': 1745.20,
            'open': 1748.10,
            'close': 1747.80,
            'ch': 2.45,
            'chp': 0.14
        }
        
        # Add data
        result = self.db_manager.add_price_data(test_data)
        
        # Assertions
        self.assertTrue(result)
        
        # Verify data was added
        session = self.db_manager.SessionLocal()
        record = session.query(GoldPrice).first()
        session.close()
        
        self.assertIsNotNone(record)
        self.assertEqual(record.price, 1750.25)
        self.assertEqual(record.metal, 'XAU')
        
    def test_get_latest_price(self):
        """Test getting latest price from database"""
        # Add test data
        test_data = {
            'timestamp': int(datetime.now().timestamp()),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'metal': 'XAU',
            'currency': 'USD',
            'price': 1750.25,
            'bid': 1749.50,
            'ask': 1751.00,
            'high': 1755.30,
            'low': 1745.20,
            'open': 1748.10,
            'close': 1747.80,
            'ch': 2.45,
            'chp': 0.14
        }
        
        self.db_manager.add_price_data(test_data)
        
        # Get latest price
        latest = self.db_manager.get_latest_price()
        
        # Assertions
        self.assertIsNotNone(latest)
        self.assertEqual(latest['price'], 1750.25)
        self.assertEqual(latest['metal'], 'XAU')

class TestGoldPricePredictor(unittest.TestCase):
    """Test the GoldPricePredictor class"""
    
    def setUp(self):
        """Set up test predictor"""
        # Mock database manager
        self.mock_db_manager = MagicMock()
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100)
        prices = np.linspace(1700, 1800, 100) + np.random.normal(0, 5, 100)
        
        self.test_df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'high': prices + 10,
            'low': prices - 10,
            'open': prices - 5,
            'close': prices + 5
        })
        
        # Mock get_data_for_training to return test data
        self.mock_db_manager.get_data_for_training.return_value = self.test_df
        
        # Create predictor with mock db_manager
        self.predictor = GoldPricePredictor(db_manager=self.mock_db_manager)
    
    def test_prepare_data(self):
        """Test data preparation for LSTM model"""
        X_train, X_test, y_train, y_test, scaler = self.predictor.prepare_data(
            self.test_df, target_col='price', sequence_length=5
        )
        
        # Assertions
        self.assertEqual(X_train.shape[2], 1)  # Features dimension
        self.assertEqual(X_train.shape[1], 5)  # Sequence length
        self.assertTrue(len(X_train) + len(X_test) == len(self.test_df) - 5)
    
    def test_linear_regression_training(self):
        """Test linear regression model training"""
        model_info = self.predictor.train_linear_regression(self.test_df)
        
        # Assertions
        self.assertIsNotNone(model_info)
        self.assertIn('model', model_info)
        self.assertIn('metrics', model_info)
        self.assertIn('mse', model_info['metrics'])
        self.assertIn('r2', model_info['metrics'])

if __name__ == '__main__':
    unittest.main()
