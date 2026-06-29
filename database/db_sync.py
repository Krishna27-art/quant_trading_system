import os

from dotenv import load_dotenv

load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.logger import get_logger

logger = get_logger("db_sync")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://quant_user:quant_pass@localhost:5432/quant_db"
)

try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=10,
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    logger.info("Synchronous database engine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize synchronous database engine: {e}")
    engine = None
    SessionLocal = None
