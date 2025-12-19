from sqlalchemy import text
from app.database import SessionLocal
import json

def store_prediction(
    transaction_id,
    dataset,
    score,
    risk_level,
    label,
    fraud_type,
    explanation,
    amount,
    entity_id,
    payload
):
    db = SessionLocal()
    try:
        query = text("""
            INSERT INTO fraud_predictions
            (transaction_id, dataset, score, risk, label, fraud_type, explanation, amount, entity_id, payload)
            VALUES
            (:transaction_id, :dataset, :score, :risk, :label, :fraud_type, :explanation, :amount, :entity_id, :payload)
        """)

        db.execute(query, {
            "transaction_id": transaction_id,
            "dataset": dataset,
            "score": score,
            "risk": risk_level,
            "label": label,
            "fraud_type": fraud_type,
            "explanation": explanation,
            "amount": amount,
            "entity_id": entity_id,
            "payload": json.dumps(payload)
        })

        db.commit()
    finally:
        db.close()
