# FinShield Backend  
AI-Powered Fraud Detection & SOC Platform (FastAPI + ML)

## Overview
FinShield is a fintech-grade fraud detection backend that ingests transactions, applies machine learning risk scoring, and generates SOC alerts for investigation.

This backend powers:
- Real-time fraud prediction
- Risk intelligence aggregation
- SOC alerting & investigation workflows

## Tech Stack
- **FastAPI** – REST APIs
- **PostgreSQL** – Transaction & alert storage
- **LightGBM / ML models** – Fraud detection
- **Rule Engine** – Fraud typology detection
- **SOC Module** – Analyst workflows
- **Uvicorn** – ASGI server

## Core Features
- ML-based fraud scoring (PaySim, IEEE-CIS, Elliptic)
- Risk levels: `allow`, `manual_review`, `auto_block`
- SOC Alerts with investigation & analyst actions
- Entity-level transaction history
- Seed script for continuous transaction simulation

## API Modules
| Module | Description |
|------|------------|
| `/predict` | Fraud prediction |
| `/risk/*` | Risk intelligence |
| `/soc/*` | SOC alerts & actions |

## Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
