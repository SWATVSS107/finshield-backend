def generate_soc_explanation(alert: dict) -> str:
    lines = []

    lines.append(f"Alert Type: {alert.get('fraud_type', 'UNKNOWN')}")
    lines.append(f"Risk Level: {alert.get('risk')}")
    lines.append(f"Model Score: {round(alert.get('score', 0), 4)}")

    reason = alert.get("fraud_explanation")
    if reason:
        lines.append(f"Rule Triggered: {reason}")

    if alert.get("risk") == "auto_block":
        lines.append("Recommended Action: Immediate block")
    elif alert.get("risk") == "manual_review":
        lines.append("Recommended Action: Analyst review required")
    else:
        lines.append("Recommended Action: Allow transaction")

    return " | ".join(lines)
