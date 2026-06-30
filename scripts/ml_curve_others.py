"""Data-efficiency curve for KNN, Linear Reg., SVM (sklearn; no torch/xgboost) at N=100..6000,
20 seeds, both targets. Reuses ml_features.npz base block; recomputes port encoding per subsample.
Records RMSE, R2, train, infer."""
import time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
def log(*a): print(*a,flush=True)
ROOT=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM")
SCR=Path("/private/tmp/claude-501/-Users-eduardopacheco-Desktop-Documentos-USP-Portos/d17e2190-43fc-486a-8398-986e9d98c9c7/scratchpad")
METRICS=ROOT/"metrics/ML_models"
SIZES=[100,500,1500,3000,6000]
SEEDS=[42,137,256,512,1024,2048,3141,4096,5000,6273,7777,8192,9001,10240,11111,12345,13579,14641,15360,16384]
TARGETS=["TOperacao","TAtracado"]
d=np.load(SCR/"ml_features.npz"); base_tr=d["Xtr"][:,:-1]; base_te=d["Xte"][:,:-1]; Ntr=base_tr.shape[0]
tr=pd.read_parquet(ROOT/"data_input/train_strings.parquet",columns=["Porto Atracação","TOperacao","TAtracado"])
te=pd.read_parquet(ROOT/"data_input/test_strings.parquet",columns=["Porto Atracação","TOperacao","TAtracado"])
port_tr=tr["Porto Atracação"].to_numpy(); port_te=te["Porto Atracação"].to_numpy()
def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    rmse=float(np.sqrt(np.mean((y-p)**2)));mae=float(np.mean(np.abs(y-p)))
    ss=float(np.sum((y-y.mean())**2));r2=float(1-np.sum((y-p)**2)/ss) if ss>0 else float("nan")
    return rmse,r2,mae
csv=METRICS/"curve_others.csv"
done=set()
if csv.exists(): done={tuple(x) for x in pd.read_csv(csv)[["model","target","size","seed"]].to_numpy()}
def feats(idx,y,target):
    Xb=base_tr[idx]; port=port_tr[idx]; gm=float(np.median(y)); med=pd.Series(y,index=port).groupby(level=0).median().to_dict()
    ptr=pd.Series(port).map(med).fillna(gm).to_numpy(np.float32); pte=pd.Series(port_te).map(med).fillna(gm).to_numpy(np.float32)
    Xtr=np.column_stack([Xb,ptr]); Xte=np.column_stack([base_te,pte])
    sc=StandardScaler().fit(Xtr); return np.nan_to_num(sc.transform(Xtr).astype(np.float32)), np.nan_to_num(sc.transform(Xte).astype(np.float32))
for target in TARGETS:
    yall=tr[target].to_numpy(np.float32); yte=te[target].to_numpy(np.float32)
    for size in SIZES:
        for seed in SEEDS:
            rng=np.random.default_rng(seed); idx=rng.choice(Ntr,size=size,replace=False); y=yall[idx]
            need=[m for m in ["knn","linear_regression","svm"] if (m,target,size,seed) not in done]
            if not need: continue
            Xtr,Xte=feats(idx,y,target)
            res={}
            if "knn" in need:
                t=time.time(); m=KNeighborsRegressor(n_neighbors=min(10,size),n_jobs=-1).fit(Xtr,y); tr_s=time.time()-t
                ti=time.time(); p=m.predict(Xte); inf=time.time()-ti; res["knn"]=(tr_s,inf,p)
            if "linear_regression" in need:
                t=time.time(); XtX=Xtr.T@Xtr; XtX[np.diag_indices_from(XtX)]+=1e-2; w=np.linalg.lstsq(XtX,Xtr.T@y,rcond=None)[0]; tr_s=time.time()-t
                ti=time.time(); p=Xte@w; inf=time.time()-ti; res["linear_regression"]=(tr_s,inf,p)
            if "svm" in need:
                t=time.time(); m=SGDRegressor(loss="epsilon_insensitive",epsilon=0.0,penalty="l2",alpha=1e-4,max_iter=10,tol=None,random_state=seed).fit(Xtr,y); tr_s=time.time()-t
                ti=time.time(); p=m.predict(Xte); inf=time.time()-ti; res["svm"]=(tr_s,inf,p)
            for mdl,(tr_s,inf,p) in res.items():
                rmse,r2,mae=metrics(yte,p)
                pd.DataFrame([{"model":mdl,"target":target,"size":size,"seed":seed,"rmse":rmse,"r2":r2,"mae":mae,"train_s":round(tr_s,3),"infer_s":round(inf,5)}]).to_csv(csv,mode="a",header=not csv.exists(),index=False)
        log(f"{target} N={size}: done seeds")
log("OTHERS CURVE DONE")
