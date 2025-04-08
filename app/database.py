import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Read the connection string from the environment variable.
# When deployed on Render, DATABASE_URL will be set to your internal URL.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Example model for storing price data.
class PriceData(Base):
    __tablename__ = "price_data"
    id = Column(Integer, primary_key=True, index=True)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Function to initialize (create) the tables.
def init_db():
    Base.metadata.create_all(bind=engine)

class DatabaseManager:
    def __init__(self):
        init_db()

    def get_latest_price(self):
        session = SessionLocal()
        try:
            # Get the most recent PriceData, ordered by creation date.
            result = session.query(PriceData).order_by(PriceData.created_at.desc()).first()
            if result:
                return {"id": result.id, "value": result.value, "created_at": result.created_at.isoformat()}
            return None
        finally:
            session.close()

    def add_price_data(self, data):
        session = SessionLocal()
        try:
            # Assuming the data contains "value" and an optional "created_at"
            created_at = data.get("created_at", datetime.utcnow().isoformat())
            record = PriceData(value=data["value"], created_at=datetime.fromisoformat(created_at))
            session.add(record)
            session.commit()
        finally:
            session.close()

    def get_price_data(self, limit=100, offset=0, start_date=None, end_date=None):
        session = SessionLocal()
        try:
            query = session.query(PriceData).order_by(PriceData.created_at.desc())
            if start_date:
                query = query.filter(PriceData.created_at >= datetime.fromisoformat(start_date))
            if end_date:
                query = query.filter(PriceData.created_at <= datetime.fromisoformat(end_date))
            records = query.offset(offset).limit(limit).all()
            return [
                {"id": rec.id, "value": rec.value, "created_at": rec.created_at.isoformat()}
                for rec in records
            ]
        finally:
            session.close()

    def get_data_for_training(self):
        # Returns all records as a list of dictionaries
        session = SessionLocal()
        try:
            records = session.query(PriceData).order_by(PriceData.created_at.asc()).all()
            return [
                {"id": rec.id, "value": rec.value, "created_at": rec.created_at.isoformat()}
                for rec in records
            ]
        finally:
            session.close()

    def import_from_csv(self):
        # If you have a CSV seed file, implement CSV import logic here
        pass
