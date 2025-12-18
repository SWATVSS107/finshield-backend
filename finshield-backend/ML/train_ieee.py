# train_ieee_gpu.py
import os, gc, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from category_encoders import TargetEncoder

RND = 42
N_SPLITS = 5
TX = "Dataset/ieee_cis/train_transaction.csv"
ID = "Dataset/ieee_cis/train_identity.csv"
OUTDIR = "models"

def load_merge():
    tx = pd.read_csv(TX)
    idf = pd.read_csv(ID)
    df = tx.merge(idf, how="left", on="TransactionID")
    return df

def features(df):
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].fillna(0).astype(float))
    if "TransactionDT" in df.columns:
        df["DT_day"] = (df["TransactionDT"] // (3600*24)).astype(int)
        df["DT_hour"] = (df["TransactionDT"] // 3600 % 24).astype(int)
    for c in ["card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain","R_emaildomain"]:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).to_dict()
            df[f"{c}_freq"] = df[c].map(vc).fillna(0).astype(int)
    num = df.select_dtypes(include=["int64","float64"]).columns
    obj = df.select_dtypes(include=["object"]).columns
    df[num] = df[num].fillna(-999)
    df[obj] = df[obj].fillna("missing")
    return df

def encode(df, target):
    # label encode low-cardinality object cols and target-encode high-cardinality cols
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    low_card = [c for c in cat_cols if df[c].nunique() < 50]
    high_card = [c for c in cat_cols if df[c].nunique() >= 50]
    le_map = {}
    for c in low_card:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_map[c] = le
    te = TargetEncoder(cols=high_card, smoothing=0.3)
    if target is not None:
        df[high_card] = te.fit_transform(df[high_card], target)
    else:
        df[high_card] = te.transform(df[high_card])
    return df, le_map, te, low_card, high_card

def train(df):
    features = [c for c in df.columns if c not in ("TransactionID","isFraud")]
    X = df[features]; y = df["isFraud"].astype(int)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "device": "gpu",         # <-- GPU enabled
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbosity": -1,
        "seed": RND
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    oof = np.zeros(len(X)); models=[]
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print("Fold", fold+1)
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        bst = lgb.train(
            params,
            dtr,
            num_boost_round=20000,
            valid_sets=[dtr, dva],
            callbacks=[
                early_stopping(stopping_rounds=200),
                log_evaluation(period=200)
            ]
        )
        models.append(bst)
        oof[va] = bst.predict(Xva, num_iteration=bst.best_iteration)
        del Xtr, Xva, ytr, yva, dtr, dva; gc.collect()

    print("IEEE OOF AUC:", roc_auc_score(y, oof))
    os.makedirs(OUTDIR, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(os.path.join(OUTDIR, f"ieee_lgb_fold{i}.txt"))
    return models

def check_gpu_support():
    print("LightGBM version:", lgb.__version__)
    try:
        # tiny test train to see if LightGBM accepts 'device': 'gpu'
        X_dummy = pd.DataFrame(np.random.randn(50, 4), columns=list("ABCD"))
        y_dummy = np.random.randint(0, 2, size=50)
        dtrain = lgb.Dataset(X_dummy, label=y_dummy)
        params_test = {"objective":"binary","metric":"auc","device":"gpu","verbosity":-1}
        lgb.train(params_test, dtrain, num_boost_round=1)
        print("GPU test passed: LightGBM build accepted 'device=gpu'.")
    except Exception as e:
        print("GPU test failed — LightGBM may be CPU-only or GPU not configured.")
        print("Test error:", e)

if __name__ == "__main__":
    check_gpu_support()
    df = load_merge()
    df = features(df)
    target = df["isFraud"]
    df, le_map, te, low_card, high_card = encode(df, target)
    models = train(df)
    joblib.dump({"le_map":le_map, "te":te, "low_card":low_card, "high_card":high_card}, os.path.join(OUTDIR,"ieee_encoders.pkl"))
    print("Saved IEEE models + encoders")
