# unified_api.py (robust loader + alignment + debug endpoints)
import os
import joblib
import json
import traceback
from math import isfinite

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
app = FastAPI(title="Unified Fraud Engine (robust + aligned)")

app = FastAPI(title="Unified Fraud Engine (robust + aligned)")

# Load thresholds (fallback defaults)
thr_path = os.path.join(MODEL_DIR, "thresholds.json")
if os.path.exists(thr_path):
    try:
        THRESHOLDS = json.load(open(thr_path, "r"))
    except Exception:
        THRESHOLDS = {"paysim": {"BLOCK": 0.80, "REVIEW": 0.20}, "ieee": {"BLOCK": 0.85, "REVIEW": 0.20}, "elliptic": {"BLOCK": 0.60, "REVIEW": 0.25}}
else:
    THRESHOLDS = {"paysim": {"BLOCK": 0.80, "REVIEW": 0.20}, "ieee": {"BLOCK": 0.85, "REVIEW": 0.20}, "elliptic": {"BLOCK": 0.60, "REVIEW": 0.25}}

def decide_risk(dataset: str, score: float):
    block = THRESHOLDS.get(dataset, {}).get("BLOCK", 0.7)
    review = THRESHOLDS.get(dataset, {}).get("REVIEW", 0.2)
    if not isfinite(score):
        return "error", "invalid_score"
    if score >= block:
        return "auto_block", f"score >= {block}"
    if score >= review:
        return "manual_review", f"{review} <= score < {block}"
    return "allow", f"score < {review}"

class Payload(BaseModel):
    dataset: Optional[str] = None
    payload: Dict[str, Any]

# --- Model wrapper to unify different formats ---
class ModelWrapper:
    def __init__(self, name, obj=None, path=None):
        self.name = name
        self.path = path
        self.raw = obj
        self.kind = None
        if obj is not None:
            self._init_from_obj(obj)

    def _init_from_obj(self, obj):
        if isinstance(obj, lgb.Booster):
            self.kind = "lgb_booster"
            self.raw = obj
            return
        if hasattr(obj, "predict_proba"):
            self.kind = "sklearn_proba"
            self.raw = obj
            return
        if hasattr(obj, "predict"):
            self.kind = "sklearn_predict"
            self.raw = obj
            return
        self.kind = "unknown"
        self.raw = obj

    def predict_proba(self, df: pd.DataFrame):
        X = df.copy()
        # lgb Booster: returns 1d array for binary objective (probability-like)
        if self.kind == "lgb_booster":
            try:
                # Try to use best_iteration if present
                niter = getattr(self.raw, "best_iteration", None)
                preds = self.raw.predict(X, num_iteration=niter)
            except Exception:
                preds = self.raw.predict(X)
            return np.asarray(preds).reshape(-1,)
        if self.kind == "sklearn_proba":
            proba = self.raw.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.reshape(-1,)
        if self.kind == "sklearn_predict":
            preds = self.raw.predict(X)
            return np.asarray(preds).astype(float).reshape(-1,)
        raise RuntimeError(f"Unknown model kind for {self.name}")

# --- robust loader that tries booster then joblib ---
def load_models(prefix):
    wrappers = []
    if not os.path.exists(MODEL_DIR):
        print("Warning: MODELS dir not found:", MODEL_DIR)
        return wrappers
    files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith(prefix)])
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        # try LightGBM booster (txt / binary)
        try:
            booster = lgb.Booster(model_file=path)
            w = ModelWrapper(name=f, obj=booster, path=path)
            wrappers.append(w)
            print(f"[loader] Loaded LGB Booster from {f}")
            continue
        except Exception as e_boost:
            # fallback: joblib (sklearn wrappers, calibrated models)
            try:
                obj = joblib.load(path)
                w = ModelWrapper(name=f, obj=obj, path=path)
                # if the joblib contains nested booster_
                if w.kind == "unknown":
                    if hasattr(obj, "booster_"):
                        try:
                            b = obj.booster_
                            w = ModelWrapper(name=f + "::booster_", obj=b, path=path)
                            wrappers.append(w)
                            print(f"[loader] Extracted booster_ from joblib wrapper {f}")
                            continue
                        except Exception:
                            pass
                wrappers.append(w)
                print(f"[loader] Loaded joblib object from {f}, detected kind={w.kind}")
                continue
            except Exception as e_job:
                print(f"[loader] Failed to load model file {f} as booster ({e_boost}) and joblib ({e_job})")
                continue
    return wrappers

# Load models for each dataset
ieee_models = load_models("ieee_lgb_fold")
paysim_models = load_models("paysim_lgb_fold")
elliptic_models = load_models("elliptic_graph_lgb_fold")

# Load aux artifacts safely
try:
    ieee_meta = joblib.load(os.path.join(MODEL_DIR, "ieee_encoders.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "ieee_encoders.pkl")) else {}
except Exception as e:
    print("[loader] Failed to load ieee_encoders.pkl:", e)
    ieee_meta = {}

try:
    paysim_scaler = joblib.load(os.path.join(MODEL_DIR, "paysim_scaler.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "paysim_scaler.pkl")) else None
except Exception as e:
    print("[loader] Failed to load paysim_scaler.pkl:", e)
    paysim_scaler = None

try:
    elliptic_meta = joblib.load(os.path.join(MODEL_DIR, "elliptic_graph_meta.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "elliptic_graph_meta.pkl")) else {}
except Exception as e:
    print("[loader] Failed to load elliptic_graph_meta.pkl:", e)
    elliptic_meta = {}

# --- Feature helpers: determine expected features + align dataframes ---
def get_expected_features_from_wrappers(wrappers):
    """Return first useful feature list found among wrappers, or None."""
    for w in wrappers:
        try:
            if w.kind == "lgb_booster":
                fn = w.raw.feature_name()
                if fn:
                    return list(fn)
            # sklearn-like feature names
            for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
                if hasattr(w.raw, attr):
                    fn = getattr(w.raw, attr)
                    if fn is not None:
                        return list(fn)
            # nested booster inside wrappers
            if hasattr(w.raw, "booster_") and isinstance(w.raw.booster_, lgb.Booster):
                return list(w.raw.booster_.feature_name())
        except Exception:
            continue
    return None

def align_df_to_expected(df: pd.DataFrame, expected_features):
    """Return df reindexed to expected_features (same order). Fill missing cols with -999.
       Convert non-numeric columns to numeric when possible; drop obvious id strings."""
    if expected_features is None:
        return df.fillna(-999)

    df_copy = df.copy()

    # Drop obvious text-ID columns that models don't want as numeric (PaySim-specific)
    for drop_col in ("nameOrig", "nameDest"):
        if drop_col in df_copy.columns:
            df_copy = df_copy.drop(columns=[drop_col])

    # Convert object columns to numeric where possible, otherwise fill -999
    for col in list(df_copy.columns):
        if df_copy[col].dtype == object:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            except Exception:
                df_copy[col] = np.nan

    # Add missing columns with -999 sentinel
    for c in expected_features:
        if c not in df_copy.columns:
            df_copy[c] = -999

    # Reindex in expected order; missing => -999
    df_aligned = df_copy.reindex(columns=expected_features, fill_value=-999)

    # Force numeric dtype and fill NaN
    for c in df_aligned.columns:
        df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce").fillna(-999)

    return df_aligned

# --- ensemble helper ---
def ensemble_score(wrappers, df: pd.DataFrame):
    scores = []
    for w in wrappers:
        try:
            sc = w.predict_proba(df)
            scores.append(np.asarray(sc).reshape(-1,))
        except Exception as e:
            # print for debugging and continue
            print(f"[predict] model {w.name} predict failed:", e)
    if len(scores) == 0:
        raise RuntimeError("No valid models available for ensemble")
    arr = np.vstack(scores)  # shape (n_models, n_rows)
    mean_scores = np.mean(arr, axis=0)
    return mean_scores

# --- Predict functions for each dataset ---
def predict_paysim(payload: dict):
    df = pd.DataFrame([payload])

    # basic feature engineering
    if "amount" in df.columns:
        try:
            df["amount_log"] = np.log1p(df["amount"].astype(float))
        except Exception:
            df["amount_log"] = 0.0
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="t")

    # numeric columns
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()

    # safe scaling: only use scaler columns present in df
    if paysim_scaler is not None and len(numcols) > 0:
        try:
            scaler_feats = getattr(paysim_scaler, "feature_names_in_", None)
            if scaler_feats is not None:
                use_cols = [c for c in scaler_feats if c in df.columns]
                if use_cols:
                    df[use_cols] = paysim_scaler.transform(df[use_cols])
            else:
                df[numcols] = paysim_scaler.transform(df[numcols])
        except Exception as e:
            print("[predict_paysim] paysim_scaler.transform failed:", e)

    df = df.fillna(-999)

    # Align df to model expected features
    expected = get_expected_features_from_wrappers(paysim_models)
    # If expected wants specific dummy cols like 't_CASH_OUT', ensure they exist
    if expected is not None:
        for col in expected:
            if col.startswith("t_") and col not in df.columns:
                df[col] = 0
    df = align_df_to_expected(df, expected)

    if len(paysim_models) == 0:
        raise RuntimeError("No PaySim models loaded")
    scs = ensemble_score(paysim_models, df)
    return float(scs[0])

def predict_ieee(payload: dict):
    df = pd.DataFrame([payload])

    if "TransactionAmt" in df.columns:
        try:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype(float))
        except Exception:
            df["TransactionAmt_log"] = 0.0
    if "TransactionDT" in df.columns:
        try:
            df["DT_day"] = (df["TransactionDT"] // (3600 * 24)).astype(int)
            df["DT_hour"] = (df["TransactionDT"] // 3600 % 24).astype(int)
        except Exception:
            df["DT_day"] = df["DT_hour"] = 0

    # label encoders map (safe usage)
    le_map = ieee_meta.get("le_map", {})
    for col, le in le_map.items():
        if col in df.columns:
            val = str(df.at[0, col])
            try:
                df[col] = le.transform([val])[0] if val in getattr(le, "classes_", []) else -1
            except Exception:
                df[col] = -1

    # target encoder for high-card cols
    te = ieee_meta.get("te", None)
    high_cols = ieee_meta.get("high_card", ieee_meta.get("high_card_cols", []))
    for c in high_cols:
        if c not in df.columns:
            df[c] = "missing"
    if te is not None and len(high_cols) > 0:
        try:
            df[high_cols] = te.transform(df[high_cols])
        except Exception as e:
            for c in high_cols:
                if c in df.columns:
                    df[c] = -1

    df = df.fillna(-999)

    # Align to expected features
    expected = get_expected_features_from_wrappers(ieee_models)
    df = align_df_to_expected(df, expected)

    if len(ieee_models) == 0:
        raise RuntimeError("No IEEE models loaded")
    scs = ensemble_score(ieee_models, df)
    return float(scs[0])

def predict_elliptic(payload: dict):
    # prefer numeric features; if txid only provided return 0.0 as fallback
    df = pd.DataFrame([payload]).select_dtypes(include=[float, int]).fillna(-999)
    if df.shape[1] == 0:
        # if you have precomputed features lookup by txid, implement here
        return 0.0

    # Align to expected features (if known)
    expected = get_expected_features_from_wrappers(elliptic_models)
    df = align_df_to_expected(df, expected)

    if len(elliptic_models) == 0:
        raise RuntimeError("No Elliptic models loaded")
    scs = ensemble_score(elliptic_models, df)
    return float(scs[0])

def auto_detect(payload):
    keys = set(payload.keys())
    if "TransactionAmt" in keys or "card1" in keys:
        return "ieee"
    if "amount" in keys and "isFraud" not in keys:
        return "paysim"
    if "txid" in keys or "pagerank" in keys:
        return "elliptic"
    return "paysim"

# --- Debug endpoints to inspect loaded models and processed vectors ---
@app.get("/debug/model_features")
def debug_model_features():
    def inspect_wrappers(wrappers):
        out = []
        for w in wrappers:
            info = {"name": w.name, "kind": w.kind, "path": w.path}
            try:
                if w.kind == "lgb_booster":
                    info["feature_count"] = len(w.raw.feature_name())
                    info["features_sample"] = w.raw.feature_name()[:30]
                else:
                    # sklearn-like
                    for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
                        if hasattr(w.raw, attr):
                            fn = getattr(w.raw, attr)
                            info["feature_count"] = len(fn) if fn is not None else None
                            if fn is not None:
                                info["features_sample"] = list(fn)[:30]
                            break
            except Exception as e:
                info["inspect_error"] = str(e)
            out.append(info)
        return out

    return {
        "ieee": inspect_wrappers(ieee_models),
        "paysim": inspect_wrappers(paysim_models),
        "elliptic": inspect_wrappers(elliptic_models),
    }

@app.post("/debug/predict_vector")
def debug_predict_vector(body: Payload):
    ds = body.dataset or auto_detect(body.payload)
    try:
        if ds == "paysim":
            df = pd.DataFrame([body.payload])
            # apply same transforms as predict_paysim but stop before ensemble_score
            if "amount" in df.columns:
                try:
                    df["amount_log"] = np.log1p(df["amount"].astype(float))
                except Exception:
                    df["amount_log"] = 0.0
            if "type" in df.columns:
                df = pd.get_dummies(df, columns=["type"], prefix="t")
            # handle scaler columns display
            scaler_feats = getattr(paysim_scaler, "feature_names_in_", None) if paysim_scaler is not None else None
            expected = get_expected_features_from_wrappers(paysim_models)
            if expected is not None:
                for col in expected:
                    if col.startswith("t_") and col not in df.columns:
                        df[col] = 0
            aligned = align_df_to_expected(df, expected)
            return {
                "dataset": "paysim",
                "scaler_feature_names": list(scaler_feats) if scaler_feats is not None else None,
                "expected_features_sample": expected[:40] if expected is not None else None,
                "raw_payload": body.payload,
                "processed_vector": aligned.iloc[0].to_dict()
            }
        elif ds == "ieee":
            df = pd.DataFrame([body.payload])
            if "TransactionAmt" in df.columns:
                try:
                    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype(float))
                except Exception:
                    df["TransactionAmt_log"] = 0.0
            if "TransactionDT" in df.columns:
                try:
                    df["DT_day"] = (df["TransactionDT"] // (3600 * 24)).astype(int)
                    df["DT_hour"] = (df["TransactionDT"] // 3600 % 24).astype(int)
                except Exception:
                    df["DT_day"] = df["DT_hour"] = 0
            expected = get_expected_features_from_wrappers(ieee_models)
            aligned = align_df_to_expected(df, expected)
            return {
                "dataset": "ieee",
                "expected_features_sample": expected[:40] if expected is not None else None,
                "raw_payload": body.payload,
                "processed_vector": aligned.iloc[0].to_dict()
            }
        elif ds == "elliptic":
            df = pd.DataFrame([body.payload]).select_dtypes(include=[float, int]).fillna(-999)
            expected = get_expected_features_from_wrappers(elliptic_models)
            aligned = align_df_to_expected(df, expected)
            return {
                "dataset": "elliptic",
                "expected_features_sample": expected[:40] if expected is not None else None,
                "raw_payload": body.payload,
                "processed_vector": aligned.iloc[0].to_dict() if aligned.shape[1] > 0 else {}
            }
        else:
            return {"error": "unknown_dataset"}
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": "debug_failed", "detail": str(e), "trace": tb}

# --- main predict endpoint ---
@app.post("/predict")
def predict(body: Payload):
    ds = body.dataset or auto_detect(body.payload)
    try:
        if ds == "paysim":
            score = predict_paysim(body.payload)
        elif ds == "ieee":
            score = predict_ieee(body.payload)
        elif ds == "elliptic":
            score = predict_elliptic(body.payload)
        else:
            return {"error": "unknown_dataset"}
    except Exception as e:
        tb = traceback.format_exc()
        print("[predict] inference error:", e, tb)
        return {"error": "inference_failed", "detail": str(e)}

    risk, reason = decide_risk(ds, float(score))
    return {
        "dataset": ds,
        "score": float(score),
        "label": 1 if risk != "allow" and risk != "error" else 0,
        "risk": risk,
        "reason": reason
    }

# Run with:
# uvicorn unified_api:app --reload --port 8000
