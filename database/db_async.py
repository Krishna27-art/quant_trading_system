import os

from dotenv import load_dotenv

load_dotenv()
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from utils.logger import get_logger

logger = get_logger("db_async")

DATABASE_URL = os.getenv("DATABASE_URL_ASYNC")
if not DATABASE_URL:
    sync_url = os.getenv("DATABASE_URL", "postgresql://postgres:localdev@localhost/postgres")
    if sync_url.startswith("postgresql://"):
        DATABASE_URL = sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        DATABASE_URL = sync_url

try:
    engine = create_async_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=20,
        pool_pre_ping=True,
    )

    SessionLocal = async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
except Exception as e:
    logger.error(f"Failed to initialize async engine: {e}")
    engine = None
    SessionLocal = None


async def get_db():
    if not SessionLocal:
        raise RuntimeError("Database not initialized")
    async with SessionLocal() as session:
        yield session
