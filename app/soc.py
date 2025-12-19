from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
from pydantic import BaseModel

from app.database import get_db
from app.explain import generate_soc_explanation
from app.llm_service import generate_investigation

router = APIRouter(prefix="/soc", tags=["SOC"])


# =========================
# 1️⃣ LIST ALERTS
# =========================
@router.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):

    rows = db.execute(
        text("""
            SELECT 
                id, transaction_id, dataset, score, risk, label,
                fraud_type, explanation, amount, entity_id, created_at
            FROM fraud_predictions
            WHERE risk IN ('manual_review', 'auto_block')
            ORDER BY created_at DESC
            LIMIT 50;
        """)
    ).fetchall()

    return [
        {
            "id": r.id,
            "transaction_id": r.transaction_id,
            "dataset": r.dataset,
            "score": r.score,
            "risk": r.risk,
            "label": r.label,
            "fraud_type": r.fraud_type,
            "fraud_explanation": r.explanation,
            "amount": r.amount,
            "entity_id": r.entity_id,
            "created_at": r.created_at
        }
        for r in rows
    ]


# =========================
# 2️⃣ ALERT DETAIL
# =========================
@router.get("/alert/{alert_id}")
def get_alert_detail(alert_id: int, db: Session = Depends(get_db)):

    row = db.execute(
        text("SELECT * FROM fraud_predictions WHERE id = :id"),
        {"id": alert_id}
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    # Entity history
    history_rows = []
    if row.entity_id:
        history_rows = db.execute(
            text("""
                SELECT id, amount, risk, fraud_type, created_at
                FROM fraud_predictions
                WHERE entity_id = :eid AND id != :id
                ORDER BY created_at DESC
                LIMIT 20
            """),
            {"eid": row.entity_id, "id": alert_id}
        ).fetchall()

    history = [
        {
            "id": h.id,
            "amount": h.amount,
            "risk": h.risk,
            "fraud_type": h.fraud_type,
            "created_at": h.created_at
        }
        for h in history_rows
    ]

    # SOC explanation (always generated)
    soc_explanation = generate_soc_explanation({
        "fraud_type": row.fraud_type,
        "risk": row.risk,
        "score": row.score,
        "fraud_explanation": row.explanation
    })

    return {
        "id": row.id,
        "transaction_id": row.transaction_id,
        "dataset": row.dataset,
        "score": row.score,
        "risk": row.risk,
        "label": row.label,
        "fraud_type": row.fraud_type,
        "fraud_explanation": row.explanation,
        "amount": row.amount,
        "entity_id": row.entity_id,
        "payload": row.payload,
        "created_at": row.created_at,
        "soc_explanation": soc_explanation,
        "history": history
    }


# =========================
# 3️⃣ INVESTIGATION (LLM / MOCK)
# =========================
@router.post("/alert/{alert_id}/investigate")
def investigate_alert(alert_id: int, db: Session = Depends(get_db)):

    row = db.execute(
        text("SELECT * FROM fraud_predictions WHERE id = :id"),
        {"id": alert_id}
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Alert not found")

    history_rows = []
    if row.entity_id:
        history_rows = db.execute(
            text("""
                SELECT amount, risk, fraud_type, created_at
                FROM fraud_predictions
                WHERE entity_id = :eid AND id != :id
                ORDER BY created_at DESC
                LIMIT 10
            """),
            {"eid": row.entity_id, "id": alert_id}
        ).fetchall()

    history = [
        {
            "amount": h.amount,
            "risk": h.risk,
            "fraud_type": h.fraud_type,
            "created_at": h.created_at
        }
        for h in history_rows
    ]

    investigation = generate_investigation({
        "fraud_type": row.fraud_type,
        "fraud_explanation": row.explanation,
        "risk": row.risk,
        "amount": row.amount,
        "dataset": row.dataset,
        "history": history
    })

    return {
        "alert_id": alert_id,
        "investigation": investigation
    }


# =========================
# 4️⃣ SOC ACTION
# =========================
class SOCAction(BaseModel):
    action: str           # approve | block | escalate
    comment: str | None = None


@router.post("/alert/{alert_id}/action")
def take_action(
    alert_id: int,
    body: SOCAction,
    db: Session = Depends(get_db)
):
    action = body.action.lower()

    if action not in ("approve", "block", "escalate"):
        raise HTTPException(status_code=400, detail="Invalid action")

    status_map = {
        "approve": "APPROVED",
        "block": "BLOCKED",
        "escalate": "ESCALATED"
    }

    result = db.execute(
        text("""
            UPDATE fraud_predictions
            SET
                status = :status,
                analyst_comment = :comment,
                action_taken_at = :ts
            WHERE id = :id
            RETURNING id, status;
        """),
        {
            "id": alert_id,
            "status": status_map[action],
            "comment": body.comment,
            "ts": datetime.utcnow()
        }
    ).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")

    db.commit()

    return {
        "alert_id": result.id,
        "new_status": result.status,
        "message": f"Alert {result.status.lower()} successfully"
    }
