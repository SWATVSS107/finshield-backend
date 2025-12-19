# train_tabular.py
# Trains LightGBM tabular experts for CreditCard and PaySim
import os, json, joblib, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

ROOT = os.path.dirname(__file__) or "."
DATA_DIR = os.path.join(ROOT, "Dataset")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def precision_at_k(y_true, y_scores, k):
    if k <= 0: return 0.0
    idx = np.argsort(y_scores)[::-1][:k]
    return float(np.mean(np.array(y_true)[idx]))

def choose_threshold_by_precision_at_k(y_true, y_scores, target_k):
    if len(y_scores)==0:
        return 0.5, 0.0
    kth_score = sorted(y_scores, reverse=True)[min(len(y_scores)-1, target_k-1)]
    prec = precision_at_k(y_true, y_scores, target_k)
    return float(kth_score), float(prec)

def save_json(path, obj):
    with open(path,"w") as f:
        json.dump(obj, f, indent=2)

# ---------- CREDITCARD TRAIN ----------
def train_creditcard():
    print("\n=== TRAIN: CreditCard ===")
    fn = os.path.join(DATA_DIR, "creditcard.csv")
    df = pd.read_csv(fn)

    features = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    df = df[features + ["Class"]].dropna()

    X = df[features].copy()
    y = df["Class"].astype(int).values

    scaler = StandardScaler()
    X[["Amount","Time"]] = scaler.fit_transform(X[["Amount","Time"]])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold,(tr,te) in enumerate(skf.split(X,y)):
        print(" credit fold", fold)

        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42
        )

        clf.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[te], y[te])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(period=0)
            ]
        )

        calib = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calib.fit(X.iloc[te], y[te])

        p = calib.predict_proba(X.iloc[te])[:,1]
        oof[te] = p

        joblib.dump(calib, os.path.join(MODEL_DIR, f"creditcard_lgb_fold{fold}.pkl"))

    roc = roc_auc_score(y, oof)
    pr = average_precision_score(y, oof)
    print(" CreditCard ROC:", roc, "PR:", pr)

    block_thr, prec50 = choose_threshold_by_precision_at_k(y, oof, 50)
    review_thr = float(np.quantile(oof, 0.01))

    print(" Recommend BLOCK =", block_thr, "REVIEW =", review_thr)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "creditcard_scaler.pkl"))
    save_json(os.path.join(MODEL_DIR, "creditcard_feature_cols.json"), features)

    return {"roc":roc,"pr":pr,"block":block_thr,"review":review_thr}

# ---------- PAYSIM TRAIN ----------
def train_paysim():
    print("\n=== TRAIN: PaySim ===")
    fn = os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(fn)

    df["amount_log"] = np.log1p(df["amount"].astype(float))
    df["same_account"] = (df["nameOrig"] == df["nameDest"]).astype(int)

    types = ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]
    for t in types:
        df[f"t_{t}"] = (df["type"] == t).astype(int)

    features = ["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
                "isFlaggedFraud","amount_log","same_account"] + [f"t_{t}" for t in types]

    df = df[features + ["isFraud"]].fillna(0)

    X = df[features].copy()
    y = df["isFraud"].astype(int).values

    scaler = StandardScaler()
    numeric_cols = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","amount_log"]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold,(tr,te) in enumerate(skf.split(X,y)):
        print(" paysim fold", fold)

        clf = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42
        )

        clf.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[te], y[te])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(40),
                lgb.log_evaluation(period=0)
            ]
        )

        calib = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
        calib.fit(X.iloc[te], y[te])

        p = calib.predict_proba(X.iloc[te])[:,1]
        oof[te] = p

        joblib.dump(calib, os.path.join(MODEL_DIR, f"paysim_lgb_fold{fold}.pkl"))

    roc = roc_auc_score(y, oof)
    pr = average_precision_score(y, oof)
    print(" PaySim ROC:", roc, "PR:", pr)

    block_thr, prec50 = choose_threshold_by_precision_at_k(y, oof, 50)
    review_thr = float(np.quantile(oof, 0.02))

    print(" Recommend BLOCK =", block_thr, "REVIEW =", review_thr)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "paysim_scaler.pkl"))
    save_json(os.path.join(MODEL_DIR, "paysim_feature_cols.json"), features)

    return {"roc":roc,"pr":pr,"block":block_thr,"review":review_thr}

# ---------- MAIN ----------
if __name__ == "__main__":
    t0 = time.time()

    credit = train_creditcard()
    paysim = train_paysim()

    thresholds = {
        "creditcard": {"BLOCK": credit["block"], "REVIEW": credit["review"]},
        "paysim": {"BLOCK": paysim["block"], "REVIEW": paysim["review"]}
    }

    save_json(os.path.join(MODEL_DIR,"thresholds.json"), thresholds)

    print("\n=== FINISHED TRAINING ===")
    print(json.dumps(thresholds, indent=2))
    print("Time:", int(time.time()-t0), "seconds")
