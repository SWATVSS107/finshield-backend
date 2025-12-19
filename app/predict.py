from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import uuid
import requests

from sqlalchemy.orm import Session
from app.database import get_db

from app.rule_engine import apply_rules
from app.store import store_prediction

router = APIRouter()


# ---- Request Schema ----
class PredictRequest(BaseModel):
    dataset: str | None = None
    payload: Dict[str, Any]


# ---- ML Service URL ----
ML_URL = "http://127.0.0.1:8001/predict"   # unified_api.py (ML engine)


# ---- Predict Endpoint ----
@router.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    # 1. Call ML engine via HTTP
    try:
        response = requests.post(
            ML_URL,
            json={
                "dataset": req.dataset,
                "payload": req.payload
            },
            timeout=5
        )
        result = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML service error: {e}")

    if "error" in result:
        raise HTTPException(status_code=500, detail=result)

    # 2. Apply fraud typology rules
    fraud_type, explanation = apply_rules(db, req.payload, result)
    result["fraud_type"] = fraud_type
    result["fraud_explanation"] = explanation

    # 3. Generate transaction ID
    transaction_id = str(uuid.uuid4())

    # 4. Store in database
    store_prediction(
    transaction_id=transaction_id,
    dataset=result["dataset"],
    score=result["score"],
    risk_level=result["risk"],
    label=result["label"],
    fraud_type=fraud_type,
    explanation=explanation,
    amount=req.payload.get("amount") or req.payload.get("TransactionAmt"),
    entity_id=req.payload.get("entity_id"),
    payload=req.payload
)


    # 5. Return final response
    return {
        "transaction_id": transaction_id,
        **result
    }

