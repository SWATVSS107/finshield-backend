def generate_investigation(alert: dict) -> dict:
    """
    Mock LLM investigation response.
    This will later be replaced with real GenAI.
    """

    fraud_type = alert.get("fraud_type")
    amount = alert.get("amount")
    risk = alert.get("risk")
    history = alert.get("history", [])

    signals = []

    if fraud_type:
        signals.append(f"Rule triggered: {fraud_type}")

    if amount and amount >= 10000:
        signals.append(f"High transaction amount: ₹{amount}")

    if len(history) >= 5:
        signals.append(f"Multiple past transactions detected: {len(history)}")

    recommended_action = "Review"
    if risk == "auto_block":
        recommended_action = "Block"
    elif risk == "manual_review":
        recommended_action = "Escalate"

    return {
        "summary": f"Transaction flagged due to {fraud_type or 'model prediction'}.",
        "signals": signals,
        "risk_assessment": risk,
        "recommended_action": recommended_action
    }
