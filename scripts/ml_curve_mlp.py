"""Data-efficiency curve for the MLP (torch; no xgboost import) at N=100..6000, 20 seeds, both targets."""
import time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
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
class MLP(nn.Module):
    def __init__(s,n):
        super().__init__(); s.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(0.1),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Linear(128,1))
    def forward(s,x): return s.net(x).squeeze(-1)
def train_mlp(X,y,seed):
    torch.manual_seed(seed); m=MLP(X.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.MSELoss()
    n=len(X); nval=max(2,int(0.1*n)); ntr=n-nval; rng=np.random.default_rng(seed); best=1e9; noimp=0; B=min(256,max(8,ntr//4))
    Xt=torch.from_numpy(X); yt=torch.from_numpy(y); Xv=Xt[ntr:]; yv=yt[ntr:]
    for ep in range(300):
        m.train(); order=rng.permutation(ntr)
        for s0 in range(0,ntr,B):
            idx=order[s0:s0+B]
            if len(idx)<2: continue
            opt.zero_grad(); lf(m(Xt[idx]),yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v=lf(m(Xv),yv).item()
        if v<best-1e-4: best=v; noimp=0
        else:
            noimp+=1
            if noimp>=20: break
    return m
csv=METRICS/"curve_mlp.csv"
done=set()
if csv.exists(): done={tuple(x) for x in pd.read_csv(csv)[["target","size","seed"]].to_numpy()}
for target in TARGETS:
    yall=tr[target].to_numpy(np.float32); yte=te[target].to_numpy(np.float32)
    for size in SIZES:
        for seed in SEEDS:
            if (target,size,seed) in done: continue
            rng=np.random.default_rng(seed); idx=rng.choice(Ntr,size=size,replace=False); y=yall[idx]
            Xb=base_tr[idx]; port=port_tr[idx]; gm=float(np.median(y)); med=pd.Series(y,index=port).groupby(level=0).median().to_dict()
            ptr=pd.Series(port).map(med).fillna(gm).to_numpy(np.float32); pte=pd.Series(port_te).map(med).fillna(gm).to_numpy(np.float32)
            Xtr=np.column_stack([Xb,ptr]); Xte=np.column_stack([base_te,pte])
            sc=StandardScaler().fit(Xtr); Xtr=np.nan_to_num(sc.transform(Xtr).astype(np.float32)); Xte=np.nan_to_num(sc.transform(Xte).astype(np.float32))
            t=time.time(); m=train_mlp(Xtr,y,seed); tr_s=time.time()-t
            m.eval()
            with torch.no_grad():
                ti=time.time(); p=m(torch.from_numpy(Xte)).numpy(); inf=time.time()-ti
            rmse,r2,mae=metrics(yte,p)
            pd.DataFrame([{"model":"mlp","target":target,"size":size,"seed":seed,"rmse":rmse,"r2":r2,"mae":mae,"train_s":round(tr_s,3),"infer_s":round(inf,5)}]).to_csv(csv,mode="a",header=not csv.exists(),index=False)
        log(f"mlp {target} N={size} done")
log("MLP CURVE DONE")
