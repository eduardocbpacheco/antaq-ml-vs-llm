"""Data-efficiency curve for XGBoost (the standard/best ML), at the SAME sizes as the encoder
(100..12000), 20 seeds. Trains on N random subsamples of the full train set, using the shared
feature block (tabular + embeddings) precomputed in ml_features.npz, recomputing the port
median-target-encoding on each subsample (fair). Records RMSE, R2, train time, inference time.
No torch import -> no OpenMP deadlock."""
import json, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
def log(*a): print(*a,flush=True)
ROOT=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM")
SCR=Path("/private/tmp/claude-501/-Users-eduardopacheco-Desktop-Documentos-USP-Portos/d17e2190-43fc-486a-8398-986e9d98c9c7/scratchpad")
HP=ROOT/"hyperparams"; METRICS=ROOT/"metrics/ML_models"
SIZES=[100,500,1500,3000,6000,12000]
SEEDS=[42,137,256,512,1024,2048,3141,4096,5000,6273,7777,8192,9001,10240,11111,12345,13579,14641,15360,16384]
TARGETS=["TOperacao","TAtracado"]

d=np.load(SCR/"ml_features.npz")
base_tr=d["Xtr"][:,:-1]   # shared block (drop the TOperacao port col)
base_te=d["Xte"][:,:-1]
Ntr=base_tr.shape[0]
# targets + port in the SAME row order as the npz (built from train/test_strings in order)
tr=pd.read_parquet(ROOT/"data_input/train_strings.parquet",columns=["Porto Atracação","TOperacao","TAtracado"])
te=pd.read_parquet(ROOT/"data_input/test_strings.parquet",columns=["Porto Atracação","TOperacao","TAtracado"])
port_tr_all=tr["Porto Atracação"].to_numpy(); port_te_all=te["Porto Atracação"].to_numpy()
log(f"base train {base_tr.shape}, test {base_te.shape}, Ntr={Ntr}")

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    rmse=float(np.sqrt(np.mean((y-p)**2)));mae=float(np.mean(np.abs(y-p)))
    ss=float(np.sum((y-y.mean())**2));r2=float(1-np.sum((y-p)**2)/ss) if ss>0 else float("nan")
    return rmse,r2,mae

rows=[]
csv=METRICS/"curve_xgboost.csv"
done=set()
if csv.exists():
    done={tuple(x) for x in pd.read_csv(csv)[["target","size","seed"]].to_numpy()}
for target in TARGETS:
    params=json.load(open(HP/f"best_params_{target}.json")); params.update(dict(objective="reg:squarederror",tree_method="hist",eval_metric="rmse",seed=0)); params.pop("n_estimators",None)
    y_all=tr[target].to_numpy(np.float32); yte=te[target].to_numpy(np.float32)
    for size in SIZES:
        for seed in SEEDS:
            if (target,size,seed) in done: continue
            rng=np.random.default_rng(seed); idx=rng.choice(Ntr,size=size,replace=False)
            Xb=base_tr[idx]; y=y_all[idx]; port=port_tr_all[idx]
            gm=float(np.median(y)); med=pd.Series(y,index=port).groupby(level=0).median().to_dict()
            ptr=pd.Series(port).map(med).fillna(gm).to_numpy(np.float32)
            pte=pd.Series(port_te_all).map(med).fillna(gm).to_numpy(np.float32)
            Xtr=np.column_stack([Xb,ptr]); Xte=np.column_stack([base_te,pte])
            vfrac=0.1 if size>=50 else 0.0
            ci,vi=train_test_split(np.arange(size),test_size=max(2,int(vfrac*size)),random_state=seed)
            t=time.time()
            dtr=xgb.QuantileDMatrix(Xtr[ci],label=y[ci]); dval=xgb.QuantileDMatrix(Xtr[vi],label=y[vi],ref=dtr)
            bst=xgb.train(params,dtr,num_boost_round=3000,evals=[(dval,"v")],early_stopping_rounds=50,verbose_eval=False)
            tr_s=time.time()-t
            dte=xgb.QuantileDMatrix(Xte); ti=time.time(); pred=bst.predict(dte); inf_s=time.time()-ti
            rmse,r2,mae=metrics(yte,pred)
            rows.append({"model":"xgboost","target":target,"size":size,"seed":seed,"rmse":rmse,"r2":r2,"mae":mae,
                         "train_s":round(tr_s,3),"infer_s":round(inf_s,5),"best_iter":int(bst.best_iteration)})
            pd.DataFrame([rows[-1]]).to_csv(csv,mode="a",header=not csv.exists(),index=False)
        sub=[r for r in rows if r["target"]==target and r["size"]==size]
        if sub:
            log(f"{target} N={size:5d}: RMSE={np.mean([r['rmse'] for r in sub]):.2f} R2={np.mean([r['r2'] for r in sub]):+.3f} train={np.mean([r['train_s'] for r in sub]):.1f}s")
log("ML CURVE DONE")
