import os

from dotenv import load_dotenv

load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.logger import get_logger

logger = get_logger("db_sync")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///quant.db"
)

try:
    is_sqlite = DATABASE_URL.startswith("sqlite")

    if is_sqlite:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
        )
    else:
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
