"""Data-efficiency curves: XGBoost vs Encoder (R2 vs #train examples, log-x), both targets,
with XGBoost's full-data point (391,460) and the generative-LLM points for reference."""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ML=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM/metrics/ML_models")
LLM=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM/metrics/LLM")
TAR=["TOperacao","TAtracado"]
cur=pd.read_csv(ML/"curve_xgboost.csv")
enc=pd.read_csv(LLM/"results_encoder.csv")
xgb_full=pd.read_csv(ML/"results_xgboost.csv")
gen=pd.read_csv(LLM/"results_finetune.csv")
NFULL=391460
encmap={"ft_100":100,"ft_500":500,"ft_1500":1500,"ft_3000":3000,"ft_6000":6000,"ft_12000":12000}
genmap=encmap
fig,axes=plt.subplots(1,2,figsize=(7.4,3.1),sharey=True)
for ax,t in zip(axes,TAR):
    # XGBoost curve
    c=cur[cur.target==t].groupby("size")["r2"].mean()
    xs=list(c.index); ys=list(c.values)
    full=xgb_full[xgb_full.target==t]["r2"].mean()
    ax.plot(xs+[NFULL], ys+[full], "^-", color="#2c7fb8", lw=2, ms=5, label="XGBoost (ML)")
    ax.scatter([NFULL],[full], color="#2c7fb8", s=70, zorder=5, marker="*")
    # Encoder curve
    e=enc[enc.strategy.isin(encmap)].copy(); e["N"]=e.strategy.map(encmap)
    eg=e[e.target==t].groupby("N")["r2"].mean()
    ax.plot(list(eg.index), list(eg.values), "o-", color="#31a354", lw=2, ms=5, label="Encoder+head")
    # Generative LLM
    g=gen[gen.strategy.isin(genmap)].copy(); g["N"]=g.strategy.map(genmap)
    gg=g[g.target==t].groupby("N")["r2"].mean()
    ax.plot(list(gg.index), list(gg.values), "s--", color="#d95f0e", lw=1.5, ms=4, label="Generative LLM")
    ax.axhline(0,color="k",lw=0.5,alpha=0.4)
    ax.set_xscale("log"); ax.set_title(t,fontsize=10)
    ax.set_xlabel("# training examples (log)",fontsize=9); ax.grid(ls=":",lw=0.5,alpha=0.5)
    ax.tick_params(labelsize=8); ax.set_ylim(-0.7,0.8)
axes[0].set_ylabel("$R^2$ (test, mean of seeds)",fontsize=9)
axes[0].legend(fontsize=7.5,loc="lower right",frameon=True)
plt.tight_layout()
out="/Users/eduardopacheco/Desktop/Documentos/USP/Portos/paper_kdmile/fig_curves2.pdf"
plt.savefig(out,bbox_inches="tight"); print("saved",out)
# print the data-matched table
print("\n=== R2 by N (data-matched) ===")
for t in TAR:
    print(f"\n{t}:")
    c=cur[cur.target==t].groupby("size")["r2"].mean()
    e=enc[enc.strategy.isin(encmap)].copy(); e["N"]=e.strategy.map(encmap); eg=e[e.target==t].groupby("N")["r2"].mean()
    for N in [100,500,1500,3000,6000,12000]:
        print(f"  N={N:6d}: XGB={c.get(N,float('nan')):+.3f}  Enc={eg.get(N,float('nan')):+.3f}")
    print(f"  N={NFULL} (full): XGB={xgb_full[xgb_full.target==t]['r2'].mean():+.3f}")
