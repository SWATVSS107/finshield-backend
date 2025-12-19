from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.predict import router as predict_router
from app.risk import router as risk_router
from app.soc import router as soc_router


# 1️⃣ Create FastAPI app FIRST
app = FastAPI(title="FinShield Backend")


# 2️⃣ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3️⃣ Include routers
app.include_router(predict_router)
app.include_router(risk_router)
app.include_router(soc_router)


# 4️⃣ Health check
@app.get("/")
def root():
    return {"status": "Backend running"}
