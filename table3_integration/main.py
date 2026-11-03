"""
FastAPI backend for the Fraud Detection HITL Integration Layer.
Supports PostgreSQL (default) and multi-dataset integration (IEEE-CIS, FiFAR, Sichuan).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import (LightGBMPrediction, OperatorFeedback, RLAction,
                      Transaction, get_db, init_db)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# --- Pydantic schemas ---

class LightGBMPredictionIn(BaseModel):
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    prediction_probability: float = Field(..., ge=0.0, le=1.0)
    top3_shap_features: Optional[Dict[str, float]] = None

class RLActionIn(BaseModel):
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    chosen_action: str = Field(..., pattern="^(Approve|Review|Block)$")
    q_values: Optional[Dict[str, float]] = None
    top3_explanations: Optional[Dict[str, float]] = None

class FeedbackIn(BaseModel):
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    operator_decision: str = Field(..., pattern="^(confirm_fraud|mark_legitimate)$")
    model_source: str = Field(..., pattern="^(lightgbm|rl)$")

class LightGBMPredictionOut(BaseModel):
    id: int
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    prediction_probability: float
    top3_shap_features: Optional[Dict[str, Any]] = None
    created_at: datetime
    class Config:
        from_attributes = True

class RLActionOut(BaseModel):
    id: int
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    chosen_action: str
    q_values: Optional[Dict[str, Any]] = None
    top3_explanations: Optional[Dict[str, Any]] = None
    created_at: datetime
    class Config:
        from_attributes = True

class FeedbackOut(BaseModel):
    id: int
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    operator_decision: str
    feedback_timestamp: datetime
    model_source: str
    class Config:
        from_attributes = True

class TransactionDetailOut(BaseModel):
    transaction_id: str
    dataset_name: Optional[str] = "ieee_cis"
    features: Optional[Dict[str, Any]] = None
    is_fraud: Optional[bool] = None
    lightgbm_predictions: List[LightGBMPredictionOut] = []
    rl_actions: List[RLActionOut] = []
    operator_feedbacks: List[FeedbackOut] = []

class IngestResponse(BaseModel):
    status: str = "ok"
    records_ingested: int


# --- App ---

app = FastAPI(
    title="Fraud Detection HITL Integration API",
    description="Backend API for the Samsung Capstone Fraud Detection project (PostgreSQL + Multi-dataset).",
    version="1.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def on_startup():
    logger.info("Starting up -- initialising database ...")
    await init_db()
    logger.info("Server ready.")


# --- Ingest ---

@app.post("/api/ingest/lightgbm", response_model=IngestResponse, tags=["Ingest"])
async def ingest_lightgbm(records: List[LightGBMPredictionIn],
                           db: AsyncSession = Depends(get_db)):
    """Ingest LightGBM predictions."""
    logger.info("Ingesting %d LightGBM prediction(s) ...", len(records))
    for rec in records:
        existing = await db.get(Transaction, rec.transaction_id)
        if existing is None:
            db.add(Transaction(
                transaction_id=rec.transaction_id,
                dataset_name=rec.dataset_name or "ieee_cis"
            ))
        db.add(LightGBMPrediction(
            transaction_id=rec.transaction_id,
            dataset_name=rec.dataset_name or "ieee_cis",
            prediction_probability=rec.prediction_probability,
            top3_shap_features=rec.top3_shap_features,
        ))
    await db.commit()
    logger.info("LightGBM ingest complete (%d records).", len(records))
    return IngestResponse(records_ingested=len(records))


@app.post("/api/ingest/rl", response_model=IngestResponse, tags=["Ingest"])
async def ingest_rl(records: List[RLActionIn],
                     db: AsyncSession = Depends(get_db)):
    """Ingest RL actions."""
    logger.info("Ingesting %d RL action(s) ...", len(records))
    for rec in records:
        existing = await db.get(Transaction, rec.transaction_id)
        if existing is None:
            db.add(Transaction(
                transaction_id=rec.transaction_id,
                dataset_name=rec.dataset_name or "ieee_cis"
            ))
        db.add(RLAction(
            transaction_id=rec.transaction_id,
            dataset_name=rec.dataset_name or "ieee_cis",
            chosen_action=rec.chosen_action,
            q_values=rec.q_values,
            top3_explanations=rec.top3_explanations,
        ))
    await db.commit()
    logger.info("RL ingest complete (%d records).", len(records))
    return IngestResponse(records_ingested=len(records))


# --- Query ---

@app.get("/api/predictions/lightgbm", response_model=List[LightGBMPredictionOut],
         tags=["Query"])
async def get_lightgbm_predictions(
        top_k: int = Query(default=10, ge=1, le=500),
        dataset: Optional[str] = Query(default=None),
        db: AsyncSession = Depends(get_db)):
    """Top-K highest-risk LightGBM predictions, optionally filtered by dataset."""
    stmt = select(LightGBMPrediction)
    if dataset:
        stmt = stmt.where(LightGBMPrediction.dataset_name == dataset)
    stmt = stmt.order_by(LightGBMPrediction.prediction_probability.desc()).limit(top_k)
    result = await db.execute(stmt)
    return [LightGBMPredictionOut.model_validate(r) for r in result.scalars().all()]


@app.get("/api/predictions/rl", response_model=List[RLActionOut], tags=["Query"])
async def get_rl_actions(
        top_k: int = Query(default=10, ge=1, le=500),
        dataset: Optional[str] = Query(default=None),
        db: AsyncSession = Depends(get_db)):
    """Latest RL actions, optionally filtered by dataset."""
    stmt = select(RLAction)
    if dataset:
        stmt = stmt.where(RLAction.dataset_name == dataset)
    stmt = stmt.order_by(RLAction.created_at.desc()).limit(top_k)
    result = await db.execute(stmt)
    return [RLActionOut.model_validate(r) for r in result.scalars().all()]


@app.get("/api/transaction/{transaction_id}", response_model=TransactionDetailOut,
         tags=["Query"])
async def get_transaction_detail(transaction_id: str,
                                  db: AsyncSession = Depends(get_db)):
    """Full transaction detail."""
    txn = await db.get(Transaction, transaction_id)
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    lgbm_result = await db.execute(
        select(LightGBMPrediction).where(
            LightGBMPrediction.transaction_id == transaction_id))
    rl_result = await db.execute(
        select(RLAction).where(RLAction.transaction_id == transaction_id))
    fb_result = await db.execute(
        select(OperatorFeedback).where(
            OperatorFeedback.transaction_id == transaction_id))

    return TransactionDetailOut(
        transaction_id=txn.transaction_id,
        dataset_name=txn.dataset_name,
        features=txn.features,
        is_fraud=txn.is_fraud,
        lightgbm_predictions=[LightGBMPredictionOut.model_validate(r)
                              for r in lgbm_result.scalars().all()],
        rl_actions=[RLActionOut.model_validate(r)
                    for r in rl_result.scalars().all()],
        operator_feedbacks=[FeedbackOut.model_validate(r)
                            for r in fb_result.scalars().all()],
    )


# --- Feedback ---

@app.post("/api/feedback", response_model=FeedbackOut, tags=["Feedback"])
async def submit_feedback(payload: FeedbackIn,
                           db: AsyncSession = Depends(get_db)):
    """Record operator feedback."""
    txn = await db.get(Transaction, payload.transaction_id)
    if txn is None:
        raise HTTPException(status_code=404,
                            detail=f"Transaction '{payload.transaction_id}' not found")
    feedback = OperatorFeedback(
        transaction_id=payload.transaction_id,
        dataset_name=payload.dataset_name or txn.dataset_name or "ieee_cis",
        operator_decision=payload.operator_decision,
        feedback_timestamp=datetime.utcnow(),
        model_source=payload.model_source,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    logger.info("Feedback recorded: txn=%s, dataset=%s, decision=%s, source=%s",
                payload.transaction_id, feedback.dataset_name, payload.operator_decision, payload.model_source)
    return FeedbackOut.model_validate(feedback)


@app.get("/api/feedback/export", response_model=List[FeedbackOut], tags=["Retrain"])
async def export_feedback(dataset: Optional[str] = Query(default=None),
                          db: AsyncSession = Depends(get_db)):
    """Export all operator feedback for retraining."""
    stmt = select(OperatorFeedback)
    if dataset:
        stmt = stmt.where(OperatorFeedback.dataset_name == dataset)
    stmt = stmt.order_by(OperatorFeedback.feedback_timestamp.desc())
    result = await db.execute(stmt)
    return [FeedbackOut.model_validate(r) for r in result.scalars().all()]


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "database": "PostgreSQL"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
