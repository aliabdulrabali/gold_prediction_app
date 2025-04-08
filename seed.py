from datetime import datetime
from app.database import SessionLocal, PriceData, init_db

def seed_data():
    # Create tables if they don't exist
    init_db()
    session = SessionLocal()
    try:
        # Create a sample record; adjust the "value" as needed
        sample_record = PriceData(
            value="1800.00",
            created_at=datetime.utcnow()
        )
        session.add(sample_record)
        session.commit()
        print("Database seeded successfully!")
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        session.close()

if __name__ == '__main__':
    seed_data()
