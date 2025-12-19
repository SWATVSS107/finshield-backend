import requests
import random
import time
from datetime import datetime, timezone

API_URL = "http://127.0.0.1:8000/predict"

DATASETS = ["paysim", "ieee", "elliptic"]
ENTITIES = [f"user_{i}" for i in range(1, 101)]


def generate_payload(dataset):
    amount = random.choice([
        random.uniform(50, 500),
        random.uniform(500, 2000),
        random.uniform(10000, 50000)
    ])

    payload = {
        "amount": round(amount, 2),
        "entity_id": random.choice(ENTITIES),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if dataset == "ieee":
        payload["TransactionAmt"] = payload.pop("amount")
        payload["TransactionDT"] = random.randint(1_000_000, 10_000_000)
        payload["card1"] = random.randint(1000, 5000)

    if dataset == "elliptic":
        payload["amount"] = round(amount * 0.001, 6)
        payload["pagerank"] = random.uniform(0, 1)
        payload["txid"] = f"tx_{random.randint(1, 100000)}"

    return payload


def seed(n=200):
    for i in range(n):
        dataset = random.choice(DATASETS)
        payload = generate_payload(dataset)

        try:
            res = requests.post(
                API_URL,
                json={"dataset": dataset, "payload": payload},
                timeout=15
            )
            print(f"[{i+1}] {res.status_code} | {res.json().get('risk')}")
        except Exception as e:
            print("Error:", e)

        time.sleep(0.2)


if __name__ == "__main__":
    seed(200)
