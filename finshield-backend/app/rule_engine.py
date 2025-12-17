from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta


def apply_rules(db: Session, payload: dict, ml_output: dict):
    amount = payload.get("amount", payload.get("TransactionAmt", 0))
    entity = payload.get("entity_id")
    dataset = ml_output["dataset"]
    label = ml_output["label"]

    # 1) HIGH_AMOUNT
    if amount is not None and amount >= 10000:
        return "HIGH_AMOUNT", f"High amount transaction detected: ₹{amount} exceeds threshold ₹10,000."

    # 2) VELOCITY_ATTACK
    if entity:
        recent_tx = db.execute(
            text("""
                SELECT COUNT(*)
                FROM fraud_predictions
                WHERE entity_id = :eid
                AND created_at >= :t
            """),
            {
                "eid": entity,
                "t": datetime.utcnow() - timedelta(minutes=10)
            }
        ).scalar()

        if recent_tx >= 5:
            return "VELOCITY_ATTACK", f"Velocity attack detected: {recent_tx} transactions in last 10 minutes."

    # 3) STRUCTURING
    if entity:
        small_tx = db.execute(
            text("""
                SELECT COUNT(*)
                FROM fraud_predictions
                WHERE entity_id = :eid
                AND amount < 1000
                AND created_at >= :t
            """),
            {
                "eid": entity,
                "t": datetime.utcnow() - timedelta(hours=1)
            }
        ).scalar()

        if small_tx >= 8:
            return "STRUCTURING", f"Structuring activity detected: {small_tx} small transactions (< ₹1,000) in last hour."

    # 4) NETWORK_RISK
    if dataset == "elliptic" and label == 1:
        return "NETWORK_RISK", "Elliptic graph anomaly detected by crypto network model."

    # 5) MODEL_ONLY
    if label == 1:
        score = ml_output.get("score")
        return "MODEL_ONLY", f"Model-only fraud: ML score {score} crossed threshold."

    return None, "No fraud detected by rules."
