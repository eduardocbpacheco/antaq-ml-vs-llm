"""Fine-tune a small BASE LLM (Qwen2.5-1.5B) on serialized ANTAQ descriptions to predict
numeric turnaround targets. Regimes: base (0 ex), ft_100, ft_500. 20 seeds. MPS / bf16.
Resumable: skips runs whose preds .npy already exists. Fresh base reload per run (robust)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import re, json, gc, sys, time
import numpy as np, pandas as pd
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def log(*a): print(*a, flush=True)

ROOT = Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM")
SCR  = Path("/Users/eduardopacheco/antaq_work")
SER  = SCR/"serialized"
PREDS = ROOT/"preds"; METRICS = ROOT/"metrics"/"LLM"
PREDS.mkdir(exist_ok=True, parents=True); METRICS.mkdir(exist_ok=True, parents=True)

MODEL_ID = "Qwen/Qwen2.5-1.5B"; MODEL_TAG = "qwen25_1p5b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
TARGETS = ["TOperacao","TAtracado"]
TARGET_DESC = {"TOperacao":"tempo de operação de carga/descarga (em horas)",
               "TAtracado":"tempo total atracado no berço (em horas)"}
SEEDS = [42,137,256,512,1024,2048,3141,4096,5000,6273,7777,8192,9001,10240,11111,12345,13579,14641,15360,16384]
SIZES = [100,500,1500,3000,6000]
EPOCHS = {100:8, 500:4, 1500:2, 3000:1, 6000:1}
BATCH = 4
LR = 2e-4
MAXLEN = 384
FIXED_LEN = 240   # fixed pad length -> constant MPS graph shape (max real len ~235)
GEN_BS = 20

def prompt_for(desc, target):
    return ("Você é um especialista em operações portuárias brasileiras.\n"
            f"Com base na descrição da atracação, estime o {TARGET_DESC[target]}.\n"
            f"Descrição: {desc}\n"
            "Responda apenas com um número em horas.\nResposta:")

def metrics(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    rmse=float(np.sqrt(np.mean((y-p)**2))); mae=float(np.mean(np.abs(y-p)))
    ss=float(np.sum((y-y.mean())**2)); r2=float(1-np.sum((y-p)**2)/ss) if ss>0 else float("nan")
    return rmse,r2,mae

NUM_RE=re.compile(r"-?\d+(?:[.,]\d+)?")
def parse(txt, fb):
    if not txt: return fb
    m=NUM_RE.search(txt.replace(" ",""))
    if not m: return fb
    try:
        v=float(m.group().replace(",",".")); return v if (0<=v<2000) else fb
    except: return fb

# ---------- data ----------
test_ser = pd.read_parquet(SER/"test.parquet")
pool_ser = pd.read_parquet(SER/"train_pool.parquet")
test_tab = pd.read_parquet(ROOT/"data_input/test_strings.parquet", columns=["IDAtracacao"]+TARGETS)
train_tab= pd.read_parquet(ROOT/"data_input/train_strings.parquet", columns=["IDAtracacao"]+TARGETS)
test = test_ser.merge(test_tab, on="IDAtracacao", how="left")
pool = pool_ser.merge(train_tab, on="IDAtracacao", how="left")
pool = pool[~pool["description"].str.startswith("ERR")].reset_index(drop=True)
test = test[~test["description"].str.startswith("ERR")].reset_index(drop=True)
FALLBACK = {t: float(train_tab[t].median()) for t in TARGETS}

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token
log(f"device={DEVICE} dtype={DTYPE} pool={len(pool)} test={len(test)} fallback={FALLBACK}")

def load_base():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    m.config.pad_token_id = tok.pad_token_id
    return m

@torch.no_grad()
def generate(model, descs, target, bs=GEN_BS):
    model.eval(); out=[]; tok.padding_side="left"
    for i in range(0,len(descs),bs):
        prompts=[prompt_for(d,target) for d in descs[i:i+bs]]
        enc=tok(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=FIXED_LEN).to(DEVICE)
        gen=model.generate(**enc, max_new_tokens=10, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(gen[:,enc["input_ids"].shape[1]:], skip_special_tokens=True))
    tok.padding_side="right"
    return out

def build_batch(rows, target):
    input_ids=[]; labels=[]
    for _,r in rows.iterrows():
        p=prompt_for(r["description"], target); c=f" {float(r[target]):.1f}"+tok.eos_token
        pid=tok(p, add_special_tokens=False)["input_ids"]; cid=tok(c, add_special_tokens=False)["input_ids"]
        ids=(pid+cid)[:FIXED_LEN]; lab=([-100]*len(pid)+cid)[:FIXED_LEN]
        input_ids.append(ids); labels.append(lab)
    m=FIXED_LEN; pad=tok.pad_token_id   # fixed length -> constant MPS graph shape
    att=[[1]*len(x)+[0]*(m-len(x)) for x in input_ids]
    input_ids=[x+[pad]*(m-len(x)) for x in input_ids]; labels=[x+[-100]*(m-len(x)) for x in labels]
    return (torch.tensor(input_ids).to(DEVICE), torch.tensor(att).to(DEVICE), torch.tensor(labels).to(DEVICE))

def train_lora(train_df, target, seed):
    torch.manual_seed(seed)
    model=load_base()
    lcfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model=get_peft_model(model, lcfg); model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    n=len(train_df); ep=EPOCHS.get(n,5); rng=np.random.default_rng(seed)
    for e in range(ep):
        order=rng.permutation(n)
        for s in range(0,n,BATCH):
            idx=order[s:s+BATCH]
            ii,att,lab=build_batch(train_df.iloc[idx], target)
            out=model(input_ids=ii, attention_mask=att, labels=lab)
            out.loss.backward(); opt.step(); opt.zero_grad()
    return model

def append_metric(strategy, target, seed, rmse, r2, mae):
    csv=METRICS/"results_finetune.csv"
    row=pd.DataFrame([{"model":MODEL_TAG,"strategy":strategy,"target":target,"seed":seed,
                       "rmse":rmse,"r2":r2,"mae":mae}])
    row.to_csv(csv, mode="a", header=not csv.exists(), index=False)

def append_timing(strategy, target, seed, train_s, infer_s):
    csv=METRICS/"timing_finetune.csv"
    row=pd.DataFrame([{"model":MODEL_TAG,"strategy":strategy,"target":target,"seed":seed,
                       "train_s":round(train_s,3),"infer_s":round(infer_s,3),"n_test":len(test)}])
    row.to_csv(csv, mode="a", header=not csv.exists(), index=False)

def free(*objs):
    for o in objs:
        try: del o
        except: pass
    if DEVICE=="mps": torch.mps.empty_cache()
    gc.collect()

def run():
    test_desc=test["description"].tolist()
    CHUNK=int(os.environ.get("CHUNK","5"))   # runs per process before fresh restart (avoids MPS frag)
    done_count=0
    # BASE (0 ex), deterministic -> replicate across seeds
    for target in TARGETS:
        if (PREDS/f"{MODEL_TAG}_base_{target}_seed{SEEDS[0]}.npy").exists(): continue
        t0=time.time(); m=load_base()
        ti=time.time(); txts=generate(m, test_desc, target); infer_s=time.time()-ti
        preds=np.array([parse(t,FALLBACK[target]) for t in txts])
        rmse,r2,mae=metrics(test[target].values, preds)
        for sd in SEEDS:
            np.save(PREDS/f"{MODEL_TAG}_base_{target}_seed{sd}.npy", preds)
            append_metric("base", target, sd, rmse, r2, mae)
        append_timing("base", target, SEEDS[0], 0.0, infer_s)
        log(f"[base] {target} RMSE={rmse:.3f} R2={r2:+.4f} ({time.time()-t0:.0f}s)")
        free(m)
    # FINE-TUNE
    for size in SIZES:
        for target in TARGETS:
            for seed in SEEDS:
                npy=PREDS/f"{MODEL_TAG}_ft{size}_{target}_seed{seed}.npy"
                if npy.exists(): continue
                t0=time.time(); rng=np.random.default_rng(seed)
                idx=rng.choice(len(pool), size=size, replace=False)
                tdf=pool.iloc[idx].reset_index(drop=True)
                ttr=time.time(); model=train_lora(tdf, target, seed); train_s=time.time()-ttr
                tin=time.time(); txts=generate(model, test_desc, target); infer_s=time.time()-tin
                preds=np.array([parse(t,FALLBACK[target]) for t in txts])
                rmse,r2,mae=metrics(test[target].values, preds)
                np.save(npy, preds); append_metric(f"ft_{size}", target, seed, rmse, r2, mae)
                append_timing(f"ft_{size}", target, seed, train_s, infer_s)
                log(f"[ft{size}] {target} seed={seed:5d} RMSE={rmse:7.3f} R2={r2:+.4f} train={train_s:.0f}s infer={infer_s:.0f}s")
                free(model)
                done_count+=1
                if done_count>=CHUNK:
                    log(f"[chunk] processed {done_count} runs, exiting for fresh restart"); return

if __name__=="__main__":
    run(); log("ALL DONE")
