import pandas as pd, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ML=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM/metrics/ML_models")
LLM=Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM/metrics/LLM")
SIZES=[100,500,1500,3000,6000]; TAR=["TOperacao","TAtracado"]
ml=pd.concat([pd.read_csv(ML/"curve_xgboost.csv"),pd.read_csv(ML/"curve_others.csv"),pd.read_csv(ML/"curve_mlp.csv")],ignore_index=True)
enc=pd.read_csv(LLM/"results_encoder.csv"); enc=enc[enc.strategy.str.startswith("ft_")].copy(); enc["size"]=enc.strategy.str[3:].astype(int); enc["model"]="encoder"
dec=pd.read_csv(LLM/"results_finetune.csv"); dec=dec[dec.strategy.str.startswith("ft_")].copy(); dec["size"]=dec.strategy.str[3:].astype(int); dec["model"]="decoder"
allc=pd.concat([ml[["model","target","size","seed","r2"]],enc[["model","target","size","seed","r2"]],dec[["model","target","size","seed","r2"]]],ignore_index=True)
xgbf=pd.read_csv(ML/"results_xgboost.csv")
style={"xgboost":("XGBoost","#1f4e79","^-"),"mlp":("MLP","#7fb0d6",":"),"linear_regression":("Linear (ridge)","#9467bd","-."),
       "knn":("KNN","#17becf","--"),"svm":("SVM","#8c8c8c",":"),"encoder":("Encoder+head","#2ca02c","o-"),"decoder":("Decoder (gen-LLM)","#d62728","s--")}
order=["xgboost","mlp","linear_regression","knn","svm","encoder","decoder"]
fig,axes=plt.subplots(1,2,figsize=(7.6,3.2),sharey=True)
for ax,t in zip(axes,TAR):
    for m in order:
        g=allc[(allc.target==t)&(allc.model==m)].groupby("size")["r2"].mean()
        lab,c,ls=style[m]
        ax.plot(list(g.index),[max(v,-0.6) for v in g.values],ls,color=c,label=lab,lw=1.8,ms=4)
    ax.scatter([391460],[xgbf[xgbf.target==t]["r2"].mean()],marker="*",s=90,color="#1f4e79",zorder=6)
    ax.axhline(0,color="k",lw=0.5,alpha=0.4); ax.set_xscale("log"); ax.set_title(t,fontsize=10)
    ax.set_xlabel("# training examples (log)",fontsize=9); ax.grid(ls=":",lw=0.4,alpha=0.5); ax.set_ylim(-0.62,0.82); ax.tick_params(labelsize=8)
axes[0].set_ylabel("$R^2$ (test, mean of seeds)",fontsize=9)
axes[1].legend(fontsize=6.6,loc="lower right",ncol=1,frameon=True)
plt.tight_layout()
out="/Users/eduardopacheco/Desktop/Documentos/USP/Portos/paper_kdmile/fig_allfamilies.pdf"
plt.savefig(out,bbox_inches="tight"); print("saved",out)
