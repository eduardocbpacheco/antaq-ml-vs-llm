"""Encoder-only model (multilingual MiniLM, same backbone as the ML embeddings) with a
2-output regression head, fine-tuned on serialized berthing descriptions to predict
[TOperacao, TAtracado] jointly. Conditions: base / ft100 / ft500 / ft1500, 20 seeds.
Same per-seed subsets as the generative fine-tuning. Resumable. MPS / bf16-free (fp32 small)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
def log(*a): print(*a,flush=True)

SMOKE="--smoke" in sys.argv
ROOT=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM")
SER=Path("/private/tmp/claude-501/-Users-eduardopacheco-Desktop-Documentos-USP-Portos/d17e2190-43fc-486a-8398-986e9d98c9c7/scratchpad/serialized")
if SMOKE:
    PREDS=Path("/private/tmp/claude-501/-Users-eduardopacheco-Desktop-Documentos-USP-Portos/d17e2190-43fc-486a-8398-986e9d98c9c7/scratchpad/smoke_enc"); PREDS.mkdir(exist_ok=True,parents=True); METRICS=PREDS
else:
    PREDS=ROOT/"preds"; METRICS=ROOT/"metrics"/"LLM"
MODEL_ID="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"; TAG="encoder_minilm"
DEVICE=("cpu" if SMOKE else ("mps" if torch.backends.mps.is_available() else "cpu"))
TARGETS=["TOperacao","TAtracado"]
SEEDS=[42,137,256,512,1024,2048,3141,4096,5000,6273,7777,8192,9001,10240,11111,12345,13579,14641,15360,16384]
SIZES=[100,500,1500,3000,6000,12000,24000]
EPOCHS={100:20,500:10,1500:6,3000:4,6000:3,12000:2,24000:2}
BATCH=16; LR=2e-5; MAXLEN=256
if SMOKE:
    SIZES=[100]; EPOCHS={100:1}

test_ser=pd.read_parquet(SER/"test.parquet"); pool_ser=pd.read_parquet(SER/"train_pool.parquet")
test_tab=pd.read_parquet(ROOT/"data_input/test_strings.parquet",columns=["IDAtracacao"]+TARGETS)
train_tab=pd.read_parquet(ROOT/"data_input/train_strings.parquet",columns=["IDAtracacao"]+TARGETS)
test=test_ser.merge(test_tab,on="IDAtracacao",how="left")
pool=pool_ser.merge(train_tab,on="IDAtracacao",how="left")
pool=pool[~pool["description"].str.startswith("ERR")].reset_index(drop=True)
test=test[~test["description"].str.startswith("ERR")].reset_index(drop=True)
# target standardization (full-train stats)
MU=train_tab[TARGETS].mean().to_numpy(np.float32); SD=train_tab[TARGETS].std().to_numpy(np.float32)
log(f"device={DEVICE} pool={len(pool)} test={len(test)} mu={MU} sd={SD}")

tok=AutoTokenizer.from_pretrained(MODEL_ID)
def load_model():
    m=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,num_labels=2,problem_type="regression")
    return m.to(DEVICE)

def enc(texts):
    return tok(texts,return_tensors="pt",padding="max_length",truncation=True,max_length=MAXLEN)

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    rmse=float(np.sqrt(np.mean((y-p)**2))); mae=float(np.mean(np.abs(y-p)))
    ss=float(np.sum((y-y.mean())**2)); r2=float(1-np.sum((y-p)**2)/ss) if ss>0 else float("nan")
    return rmse,r2,mae

@torch.no_grad()
def predict(model,descs,bs=32):
    model.eval(); outs=[]
    for i in range(0,len(descs),bs):
        e=enc(descs[i:i+bs]); e={k:v.to(DEVICE) for k,v in e.items()}
        z=model(**e).logits.float().cpu().numpy()   # standardized [n,2]
        outs.append(z)
    z=np.concatenate(outs,0)
    return z*SD+MU   # un-standardize -> hours, shape [n,2]

def train_model(tdf,seed):
    torch.manual_seed(seed)
    model=load_model(); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    y=((tdf[TARGETS].to_numpy(np.float32)-MU)/SD)
    descs=tdf["description"].tolist(); n=len(descs); ep=EPOCHS[len(descs)] if len(descs) in EPOCHS else 8
    rng=np.random.default_rng(seed)
    for e in range(ep):
        order=rng.permutation(n)
        for s in range(0,n,BATCH):
            idx=order[s:s+BATCH]
            enc_b=enc([descs[i] for i in idx]); enc_b={k:v.to(DEVICE) for k,v in enc_b.items()}
            lab=torch.tensor(y[idx]).to(DEVICE)
            out=model(**enc_b,labels=lab)
            out.loss.backward(); opt.step(); opt.zero_grad()
    return model

def append(strategy,target,seed,rmse,r2,mae):
    csv=METRICS/"results_encoder.csv"
    pd.DataFrame([{"model":TAG,"strategy":strategy,"target":target,"seed":seed,"rmse":rmse,"r2":r2,"mae":mae}]).to_csv(
        csv,mode="a",header=not csv.exists(),index=False)
def append_t(strategy,seed,train_s,infer_s):
    csv=METRICS/"timing_encoder.csv"
    pd.DataFrame([{"model":TAG,"strategy":strategy,"seed":seed,"train_s":round(train_s,3),"infer_s":round(infer_s,3),"n_test":len(test)}]).to_csv(
        csv,mode="a",header=not csv.exists(),index=False)
def free(*o):
    for x in o:
        try: del x
        except: pass
    if DEVICE=="mps": torch.mps.empty_cache()
    gc.collect()

def run():
    test_desc=test["description"].tolist(); yt=test[TARGETS].to_numpy(np.float32)
    CHUNK=int(os.environ.get("CHUNK","6")); done=0
    # BASE: untrained head, deterministic(seed0), replicate
    if not (PREDS/f"{TAG}_base_{TARGETS[0]}_seed{SEEDS[0]}.npy").exists():
        torch.manual_seed(0); m=load_model()
        ti=time.time(); P=predict(m,test_desc); infer_s=time.time()-ti
        for j,t in enumerate(TARGETS):
            rmse,r2,mae=metrics(yt[:,j],P[:,j])
            for sd in SEEDS:
                np.save(PREDS/f"{TAG}_base_{t}_seed{sd}.npy",P[:,j]); append("base",t,sd,rmse,r2,mae)
            log(f"[base] {t} RMSE={rmse:.3f} R2={r2:+.4f}")
        append_t("base",SEEDS[0],0.0,infer_s); free(m)
    for size in SIZES:
        for seed in SEEDS:
            if (PREDS/f"{TAG}_ft{size}_{TARGETS[0]}_seed{seed}.npy").exists(): continue
            rng=np.random.default_rng(seed); idx=rng.choice(len(pool),size=size,replace=False)
            tdf=pool.iloc[idx].reset_index(drop=True)
            ttr=time.time(); m=train_model(tdf,seed); train_s=time.time()-ttr
            ti=time.time(); P=predict(m,test_desc); infer_s=time.time()-ti
            for j,t in enumerate(TARGETS):
                rmse,r2,mae=metrics(yt[:,j],P[:,j])
                np.save(PREDS/f"{TAG}_ft{size}_{t}_seed{seed}.npy",P[:,j]); append(f"ft_{size}",t,seed,rmse,r2,mae)
            append_t(f"ft_{size}",seed,train_s,infer_s)
            log(f"[ft{size}] seed={seed:5d} TOp_RMSE={metrics(yt[:,0],P[:,0])[0]:.2f} TAtr_RMSE={metrics(yt[:,1],P[:,1])[0]:.2f} train={train_s:.0f}s infer={infer_s:.1f}s")
            free(m); done+=1
            if done>=CHUNK: log(f"[chunk] {done} runs, exit for fresh restart"); return
    log("ENCODER ALL DONE")

if __name__=="__main__":
    run()
