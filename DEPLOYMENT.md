# FinShield Backend Deployment Guide

## Overview
This backend consists of two services:
1. **ML Service** (`ML/unified_api.py`) - Handles ML model predictions
2. **Main API Service** (`app/main.py`) - Main API that calls ML service and handles business logic

## Prerequisites
- GitHub repository with backend code
- Render account (free tier works)

## Deployment Steps

### Step 1: Push to GitHub
```bash
cd C:\Users\vedan\OneDrive\Documents\Fintech\finshield-backend
git init
git add .
git commit -m "Initial commit - ready for Render deployment"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

#### Option A: Using render.yaml (Recommended)
1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and create all services
5. **Important**: After ML service deploys, copy its URL and update the `ML_SERVICE_URL` in the API service environment variables

#### Option B: Manual Setup (If Blueprint doesn't work)

**Deploy ML Service First:**
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `finshield-ml`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd ML && python -m uvicorn unified_api:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Click **"Create Web Service"**
5. Wait for deployment and **copy the service URL** (e.g., `https://finshield-ml.onrender.com`)

**Deploy Main API Service:**
1. Click **"New +"** → **"Web Service"**
2. Connect the same GitHub repository
3. Configure:
   - **Name**: `finshield-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Go to **"Environment"** tab and add:
   - **Key**: `ML_SERVICE_URL`
   - **Value**: `https://finshield-ml.onrender.com` (use the actual URL from step above)
5. Click **"Create Web Service"**

**Create Database:**
1. Click **"New +"** → **"PostgreSQL"**
2. Configure:
   - **Name**: `finshield-db`
   - **Database**: `finshield`
   - **User**: `finshield_user`
   - **Plan**: Free
3. After creation, go to **"finshield-api"** service → **"Environment"** tab
4. Add environment variable:
   - **Key**: `DATABASE_URL`
   - **Value**: Copy from database dashboard (Render provides this automatically if using render.yaml)

### Step 3: Update Frontend
After backend is deployed, update your frontend's environment variable in Vercel:
- **Key**: `NEXT_PUBLIC_API_BASE_URL`
- **Value**: `https://finshield-api.onrender.com` (your actual API service URL)

### Step 4: Verify Deployment
1. Check ML service: `https://finshield-ml.onrender.com` (should show FastAPI docs)
2. Check API service: `https://finshield-api.onrender.com` (should show `{"status": "Backend running"}`)
3. Test prediction endpoint from frontend

## Troubleshooting

### ML Service not accessible
- Check that ML service URL in API service environment variables is correct
- Ensure ML service is deployed and running
- Check Render logs for errors

### Database connection issues
- Verify `DATABASE_URL` is set correctly in API service
- Check database is running on Render
- Ensure database migrations are run (if needed)

### Build failures
- Check `requirements.txt` has all dependencies
- Verify Python version compatibility
- Check Render build logs for specific errors

## Local Development
To run locally:
```bash
# Terminal 1 - ML Service
cd ML
uvicorn unified_api:app --host 127.0.0.1 --port 8001

# Terminal 2 - Main API
cd ..
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Make sure to set environment variables:
- `ML_SERVICE_URL=http://127.0.0.1:8001`
- `DATABASE_URL=postgresql://postgres:VSS107@localhost:5432/finshield`

