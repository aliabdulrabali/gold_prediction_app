import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL, HISTORICAL_DATA_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create SQLAlchemy Base
Base = declarative_base()

class GoldPrice(Base):
    """SQLAlchemy model for gold price data"""
    __tablename__ = 'gold_prices'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(Integer, unique=True, index=True)
    date = Column(String)
    time = Column(String)
    metal = Column(String)
    currency = Column(String)
    price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    close = Column(Float)
    ch = Column(Float)
    chp = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'date': self.date,
            'time': self.time,
            'metal': self.metal,
            'currency': self.currency,
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'high': self.high,
            'low': self.low,
            'open': self.open,
            'close': self.close,
            'ch': self.ch,
            'chp': self.chp,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DatabaseManager:
    """Manager for database operations"""
    def __init__(self, db_url: str = DATABASE_URL):
        """
        Initialize the database manager
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized at {db_url}")
    
    def get_session(self):
        """Get a database session"""
        session = self.SessionLocal()
        try:
            return session
        finally:
            session.close()
    
    def add_price_data(self, data: Dict[str, Any]) -> bool:
        """
        Add a single price data point to the database
        
        Args:
            data: Dictionary containing price data
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            session = self.SessionLocal()
            
            # Check if record with this timestamp already exists
            existing = session.query(GoldPrice).filter_by(timestamp=data.get('timestamp')).first()
            if existing:
                logger.info(f"Record with timestamp {data.get('timestamp')} already exists")
                session.close()
                return True
            
            # Create new record
            gold_price = GoldPrice(
                timestamp=data.get('timestamp'),
                date=data.get('date'),
                time=data.get('time'),
                metal=data.get('metal'),
                currency=data.get('currency'),
                price=data.get('price'),
                bid=data.get('bid'),
                ask=data.get('ask'),
                high=data.get('high'),
                low=data.get('low'),
                open=data.get('open'),
                close=data.get('close'),
                ch=data.get('ch'),
                chp=data.get('chp')
            )
            
            session.add(gold_price)
            session.commit()
            session.close()
            
            logger.info(f"Added price data for timestamp {data.get('timestamp')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding price data: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def add_price_data_batch(self, data_list: List[Dict[str, Any]]) -> bool:
        """
        Add multiple price data points to the database
        
        Args:
            data_list: List of dictionaries containing price data
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            session = self.SessionLocal()
            
            for data in data_list:
                # Check if record with this timestamp already exists
                existing = session.query(GoldPrice).filter_by(timestamp=data.get('timestamp')).first()
                if existing:
                    logger.info(f"Record with timestamp {data.get('timestamp')} already exists, skipping")
                    continue
                
                # Create new record
                gold_price = GoldPrice(
                    timestamp=data.get('timestamp'),
                    date=data.get('date'),
                    time=data.get('time'),
                    metal=data.get('metal'),
                    currency=data.get('currency'),
                    price=data.get('price'),
                    bid=data.get('bid'),
                    ask=data.get('ask'),
                    high=data.get('high'),
                    low=data.get('low'),
                    open=data.get('open'),
                    close=data.get('close'),
                    ch=data.get('ch'),
                    chp=data.get('chp')
                )
                
                session.add(gold_price)
            
            session.commit()
            session.close()
            
            logger.info(f"Added {len(data_list)} price data points in batch")
            return True
            
        except Exception as e:
            logger.error(f"Error adding price data batch: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_price_data(self, 
                      limit: int = 100, 
                      offset: int = 0, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get price data from the database
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            
        Returns:
            List of dictionaries containing price data
        """
        try:
            session = self.SessionLocal()
            query = session.query(GoldPrice)
            
            if start_date:
                query = query.filter(GoldPrice.date >= start_date)
            
            if end_date:
                query = query.filter(GoldPrice.date <= end_date)
            
            # Order by timestamp descending (newest first)
            query = query.order_by(GoldPrice.timestamp.desc())
            
            # Apply limit and offset
            query = query.limit(limit).offset(offset)
            
            # Convert to list of dictionaries
            result = [record.to_dict() for record in query.all()]
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting price data: {str(e)}")
            if session:
                session.close()
            return []
    
    def get_latest_price(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest price data from the database
        
        Returns:
            Dictionary containing the latest price data or None if no data
        """
        try:
            session = self.SessionLocal()
            latest = session.query(GoldPrice).order_by(GoldPrice.timestamp.desc()).first()
            
            if latest:
                result = latest.to_dict()
                session.close()
                return result
            else:
                session.close()
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest price: {str(e)}")
            if session:
                session.close()
            return None
    
    def import_from_csv(self, csv_file: str = HISTORICAL_DATA_FILE) -> bool:
        """
        Import data from CSV file into the database
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            if not os.path.exists(csv_file):
                logger.error(f"CSV file {csv_file} does not exist")
                return False
                
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Convert DataFrame to list of dictionaries
            data_list = df.to_dict(orient='records')
            
            # Add to database
            return self.add_price_data_batch(data_list)
            
        except Exception as e:
            logger.error(f"Error importing from CSV: {str(e)}")
            return False
    
    def export_to_csv(self, csv_file: str, limit: int = None) -> bool:
        """
        Export data from database to CSV file
        
        Args:
            csv_file: Path to save CSV file
            limit: Maximum number of records to export (None for all)
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            session = self.SessionLocal()
            query = session.query(GoldPrice).order_by(GoldPrice.timestamp.desc())
            
            if limit:
                query = query.limit(limit)
                
            # Convert to DataFrame
            records = [record.to_dict() for record in query.all()]
            df = pd.DataFrame(records)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            
            session.close()
            logger.info(f"Exported {len(records)} records to {csv_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            if session:
                session.close()
            return False
    
    def get_data_for_training(self, days: int = 365) -> pd.DataFrame:
        """
        Get data for model training
        
        Args:
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame containing price data suitable for model training
        """
        try:
            session = self.SessionLocal()
            
            # Get records ordered by timestamp
            query = session.query(GoldPrice).order_by(GoldPrice.timestamp.asc())
            
            # Convert to DataFrame
            records = [record.to_dict() for record in query.all()]
            df = pd.DataFrame(records)
            
            session.close()
            
            if df.empty:
                logger.warning("No data available for training")
                return pd.DataFrame()
                
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Select only necessary columns for training
            training_df = df[['date', 'price', 'high', 'low', 'open', 'close']].copy()
            
            # Fill missing values
            training_df = training_df.fillna(method='ffill')
            
            return training_df
            
        except Exception as e:
            logger.error(f"Error getting data for training: {str(e)}")
            if session:
                session.close()
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    
    # Import data from CSV if it exists
    if os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Importing data from {HISTORICAL_DATA_FILE}...")
        db_manager.import_from_csv()
    
    # Get latest price
    latest = db_manager.get_latest_price()
    if latest:
        print(f"Latest Gold Price: {latest.get('price')} {latest.get('currency')} on {latest.get('date')}")
    else:
        print("No price data available")
    
    # Get data for training
    training_data = db_manager.get_data_for_training()
    print(f"Retrieved {len(training_data)} records for model training")
