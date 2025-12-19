from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db

router = APIRouter(prefix="/risk", tags=["Risk Intelligence"])

@router.get("/summary")
def risk_summary(db: Session = Depends(get_db)):

    total_predictions = db.execute(
        text("SELECT COUNT(*) FROM fraud_predictions;")
    ).scalar()

    fraud_count = db.execute(
        text("SELECT COUNT(*) FROM fraud_predictions WHERE label = 1;")
    ).scalar()

    risk_rows = db.execute(
        text("SELECT risk, COUNT(*) FROM fraud_predictions GROUP BY risk;")
    ).fetchall()
    by_risk = {row[0]: row[1] for row in risk_rows}

    ds_rows = db.execute(
        text("SELECT dataset, COUNT(*) FROM fraud_predictions GROUP BY dataset;")
    ).fetchall()
    by_dataset = {row[0]: row[1] for row in ds_rows}

    fraud_rate = fraud_count / total_predictions if total_predictions else 0

    return {
        "total_predictions": total_predictions,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "by_risk": by_risk,
        "by_dataset": by_dataset
    }
@router.get("/by-type")
def risk_by_type(db: Session = Depends(get_db)):

    rows = db.execute(
        text("""
            SELECT fraud_type, COUNT(*)
            FROM fraud_predictions
            WHERE fraud_type IS NOT NULL
            GROUP BY fraud_type;
        """)
    ).fetchall()

    by_type = {row[0]: row[1] for row in rows}
    return by_type

@router.get("/by-amount")
def risk_by_amount(db: Session = Depends(get_db)):

    buckets = {
        "0_100": db.execute(
            text("SELECT COUNT(*) FROM fraud_predictions WHERE amount < 100;")
        ).scalar(),

        "100_1000": db.execute(
            text("SELECT COUNT(*) FROM fraud_predictions WHERE amount >= 100 AND amount < 1000;")
        ).scalar(),

        "1000_10000": db.execute(
            text("SELECT COUNT(*) FROM fraud_predictions WHERE amount >= 1000 AND amount < 10000;")
        ).scalar(),

        "10000_plus": db.execute(
            text("SELECT COUNT(*) FROM fraud_predictions WHERE amount >= 10000;")
        ).scalar(),
    }

    return buckets

@router.get("/trends")
def risk_trends(db: Session = Depends(get_db)):

    rows = db.execute(
        text("""
            SELECT 
                DATE(created_at) AS day,
                COUNT(*) AS total,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS fraud
            FROM fraud_predictions
            GROUP BY day
            ORDER BY day DESC;
        """)
    ).fetchall()

    return [
        {
            "date": str(row.day),
            "total": row.total,
            "fraud": row.fraud or 0
        }
        for row in rows
    ]