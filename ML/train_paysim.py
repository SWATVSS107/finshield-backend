# train_paysim.py
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb

RND=42
N_SPLITS=5
DATA="Dataset/PS_20174392719_1491204439457_log.csv"
OUTDIR="models"

def prepare(df):
    # Expected PaySim columns: type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    # if headers differ, inspect df.columns and adjust
    # create amount log
    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"].astype(float))
    # create simple features
    if "nameOrig" in df.columns and "nameDest" in df.columns:
        df["same_account"] = (df["nameOrig"]==df["nameDest"]).astype(int)
    # basic fill
    df = df.fillna(-999)
    return df

def encode_and_train(df):
    if "isFraud" not in df.columns:
        raise ValueError("No isFraud column found in PaySim file")
    # one-hot or label encode 'type' column
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="t")
    features = [c for c in df.columns if c not in ("isFraud","nameOrig","nameDest")]
    X = df[features]; y = df["isFraud"].astype(int)
    scaler = StandardScaler()
    numcols = X.select_dtypes(include=[np.number]).columns
    X[numcols] = scaler.fit_transform(X[numcols])
    params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":64,"seed":RND,"verbosity":-1}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    oof = np.zeros(len(X)); models=[]
    for fold,(tr,va) in enumerate(skf.split(X,y)):
        Xtr,Xva = X.iloc[tr], X.iloc[va]
        ytr,yva = y.iloc[tr], y.iloc[va]
        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)
        bst = lgb.train(
    params,
    dtr,
    num_boost_round=5000,
    valid_sets=[dtr, dva],
    callbacks=[
        early_stopping(stopping_rounds=200),
        log_evaluation(200)
    ]
)
        models.append(bst)
        oof[va] = bst.predict(Xva, num_iteration=bst.best_iteration)
    print("PaySim OOF AUC:", roc_auc_score(y,oof))
    os.makedirs(OUTDIR, exist_ok=True)
    for i,m in enumerate(models):
        m.save_model(os.path.join(OUTDIR, f"paysim_lgb_fold{i}.txt"))
    joblib.dump(scaler, os.path.join(OUTDIR, "paysim_scaler.pkl"))
    print("Saved PaySim models & scaler")
    return

if __name__=="__main__":
    df = pd.read_csv(DATA)
    df = prepare(df)
    encode_and_train(df)
