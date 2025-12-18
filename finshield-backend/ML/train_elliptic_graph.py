# train_elliptic_graph.py
import os, joblib, numpy as np, pandas as pd
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb

from tqdm import tqdm
import community as community_louvain   # python-louvain

RND=42
N_SPLITS=5
DIR="Dataset/elliptic_bitcoin_dataset"
EDG=os.path.join(DIR,"elliptic_txs_edgelist.csv")
FEAT=os.path.join(DIR,"elliptic_txs_features.csv")
LAB=os.path.join(DIR,"elliptic_txs_classes.csv")
OUTDIR="models"

def load():
    edges = pd.read_csv(EDG)
    feats = pd.read_csv(FEAT, index_col=0)  # first col is tx id
    labels = pd.read_csv(LAB)
    return edges, feats, labels

def build_graph(edges):
    G = nx.DiGraph()
    # edges file typically has columns: txId1, txId2
    for _,r in edges.iterrows():
        G.add_edge(r[0], r[1])
    return G

def graph_features(G, feats):
    nodes = list(G.nodes())
    # degree features
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    und = G.to_undirected()
    partition = community_louvain.best_partition(und)
    # compile df
    rows=[]
    for n in tqdm(nodes):
        row = {}
        row["txid"]=n
        row["deg_in"] = deg_in.get(n,0)
        row["deg_out"] = deg_out.get(n,0)
        row["pagerank"] = pagerank.get(n,0)
        row["community"] = partition.get(n, -1)
        # merge with provided features if exist
        if n in feats.index:
            for c in feats.columns:
                row[c] = feats.loc[n, c]
        rows.append(row)
    gdf = pd.DataFrame(rows).set_index("txid")
    # fillna
    gdf = gdf.fillna(-999)
    return gdf

def train(gdf, labels):
    # labels has columns: txId, class (1 fraud, 0 licit, 2 unknown) - keep 1 vs 0, drop unknown (2)
    labels = labels.rename(columns={labels.columns[0]:"txid", labels.columns[1]:"class"})
    lab_df = labels[labels["class"]!=2].set_index("txid")
    data = gdf.join(lab_df, how="inner")
    X = data.drop(columns=["class"])
    y = (data["class"]==1).astype(int)
    # basic numeric casting
    X = X.select_dtypes(include=[np.number]).fillna(-999)
    params = {"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":128,"seed":RND,"verbosity":-1}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)
    oof = np.zeros(len(X)); models=[]
    for fold,(tr,va) in enumerate(skf.split(X,y)):
        print("Fold",fold+1)
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
    print("Elliptic graph OOF AUC:", roc_auc_score(y, oof))
    os.makedirs(OUTDIR, exist_ok=True)
    for i,m in enumerate(models):
        m.save_model(os.path.join(OUTDIR, f"elliptic_graph_lgb_fold{i}.txt"))
    joblib.dump({"features":list(X.columns)}, os.path.join(OUTDIR,"elliptic_graph_meta.pkl"))
    print("Saved elliptic graph models")
    return

if __name__=="__main__":
    edges, feats, labels = load()
    print("Building graph (this may take a while)...")
    G = build_graph(edges)
    print("Computing graph features...")
    gdf = graph_features(G, feats)
    train(gdf, labels)
