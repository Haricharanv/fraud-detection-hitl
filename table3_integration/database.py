"""
Database setup for the Fraud Detection HITL Integration Layer.
Supports PostgreSQL (default via Docker/local) and SQLite backends.
"""

import os
import logging
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import (Column, Integer, String, Float, DateTime,
                        ForeignKey, JSON, Boolean)
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import DeclarativeBase, relationship

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_DB = "postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_detection"
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_DB)

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_async_engine(DATABASE_URL, echo=False, future=True,
                             connect_args=_connect_args)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String, primary_key=True, index=True)
    dataset_name = Column(String, nullable=True, default="ieee_cis", index=True)
    features = Column(JSON, nullable=True)
    is_fraud = Column(Boolean, nullable=True)
    lightgbm_predictions = relationship("LightGBMPrediction", back_populates="transaction",
                                        cascade="all, delete-orphan")
    rl_actions = relationship("RLAction", back_populates="transaction",
                              cascade="all, delete-orphan")
    operator_feedbacks = relationship("OperatorFeedback", back_populates="transaction",
                                     cascade="all, delete-orphan")


class LightGBMPrediction(Base):
    __tablename__ = "lightgbm_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"),
                            nullable=False, index=True)
    dataset_name = Column(String, nullable=True, default="ieee_cis", index=True)
    prediction_probability = Column(Float, nullable=False)
    top3_shap_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    transaction = relationship("Transaction", back_populates="lightgbm_predictions")


class RLAction(Base):
    __tablename__ = "rl_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"),
                            nullable=False, index=True)
    dataset_name = Column(String, nullable=True, default="ieee_cis", index=True)
    chosen_action = Column(String, nullable=False)
    q_values = Column(JSON, nullable=True)
    top3_explanations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    transaction = relationship("Transaction", back_populates="rl_actions")


class OperatorFeedback(Base):
    __tablename__ = "operator_feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"),
                            nullable=False, index=True)
    dataset_name = Column(String, nullable=True, default="ieee_cis", index=True)
    operator_decision = Column(String, nullable=False)
    feedback_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_source = Column(String, nullable=False)
    transaction = relationship("Transaction", back_populates="operator_feedbacks")


async def init_db():
    """Create all tables if they do not already exist."""
    logger.info("Initialising database (%s) -- creating tables if needed ...",
                DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialisation complete.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
