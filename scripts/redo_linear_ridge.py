"""Recompute the linear-regression curve with RidgeCV (regularized) -> sensible at small N.
Unregularized OLS explodes when N << p (1606 features). Replaces 'linear_regression' rows in curve_others.csv."""
import time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
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
ALPHAS=[0.1,1,10,100,1000,10000]
out=[]
for target in TARGETS:
    yall=tr[target].to_numpy(np.float32); yte=te[target].to_numpy(np.float32)
    for size in SIZES:
        for seed in SEEDS:
            rng=np.random.default_rng(seed); idx=rng.choice(Ntr,size=size,replace=False); y=yall[idx]
            Xb=base_tr[idx]; port=port_tr[idx]; gm=float(np.median(y)); med=pd.Series(y,index=port).groupby(level=0).median().to_dict()
            ptr=pd.Series(port).map(med).fillna(gm).to_numpy(np.float32); pte=pd.Series(port_te).map(med).fillna(gm).to_numpy(np.float32)
            Xtr=np.column_stack([Xb,ptr]); Xte=np.column_stack([base_te,pte])
            sc=StandardScaler().fit(Xtr); Xtr=np.nan_to_num(sc.transform(Xtr).astype(np.float32)); Xte=np.nan_to_num(sc.transform(Xte).astype(np.float32))
            t=time.time(); m=RidgeCV(alphas=ALPHAS).fit(Xtr,y); tr_s=time.time()-t
            ti=time.time(); p=m.predict(Xte); inf=time.time()-ti
            rmse,r2,mae=metrics(yte,p)
            out.append({"model":"linear_regression","target":target,"size":size,"seed":seed,"rmse":rmse,"r2":r2,"mae":mae,"train_s":round(tr_s,3),"infer_s":round(inf,5)})
        log(f"ridge {target} N={size}: R2={np.mean([r['r2'] for r in out if r['target']==target and r['size']==size]):+.3f}")
new=pd.DataFrame(out)
csv=METRICS/"curve_others.csv"
old=pd.read_csv(csv); old=old[old.model!="linear_regression"]
pd.concat([old,new],ignore_index=True).to_csv(csv,index=False)
log("RIDGE REDONE; replaced linear_regression rows")
