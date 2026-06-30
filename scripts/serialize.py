"""Serialize ANTAQ berthing rows into fluent Portuguese descriptions via Bedrock DeepSeek.
Caches to parquet so it is reproducible and re-runnable. Targets are NEVER included."""
import os, sys, json, time
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from dotenv import load_dotenv

ROOT = Path("/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM")
load_dotenv(ROOT/".env")
REGION = os.environ.get("AWS_DEFAULT_REGION","us-east-1")
MODEL_ID = "deepseek.v3.2"
OUT = Path("/private/tmp/claude-501/-Users-eduardopacheco-Desktop-Documentos-USP-Portos/d17e2190-43fc-486a-8398-986e9d98c9c7/scratchpad/serialized")
OUT.mkdir(exist_ok=True, parents=True)

TARGETS = ["TOperacao","TAtracado","TEstadia","TEsperaAtracacao","TEsperaInicioOp","TEsperaDesatracacao"]
DOW = {0:"segunda-feira",1:"terça-feira",2:"quarta-feira",3:"quinta-feira",4:"sexta-feira",5:"sábado",6:"domingo"}
MES = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}

def first(s):
    return str(s).split(",")[0].strip() if pd.notna(s) else None

def na(v):
    if v is None: return False
    s=str(v).strip().lower()
    return s not in ("não se aplica","nan","none","","0","0.0")

def row_facts(r):
    """Build a clean dict of human facts for the row (no targets)."""
    f={}
    f["porto"]=r.get("Porto Atracação"); f["municipio_uf"]=f'{r.get("Município")}/{r.get("SGUF")}'
    f["regiao"]=r.get("Região Geográfica")
    f["tipo_operacao"]=r.get("Tipo de Operação"); f["navegacao"]=r.get("Tipo de Navegação da Atracação")
    f["natureza_carga"]=r.get("Natureza da Carga")
    o,d=first(r.get("Origem")),first(r.get("Destino"))
    if na(r.get("Origem")): f["origem"]=o
    if na(r.get("Destino")): f["destino"]=d
    if na(r.get("Grupo de Mercadoria")): f["mercadoria"]=r.get("Grupo de Mercadoria")
    if na(r.get("Grupo Mercadoria Conteinerizada")): f["mercadoria_conteinerizada"]=r.get("Grupo Mercadoria Conteinerizada")
    peso=r.get("VLPesoCargaBruta")
    if peso and peso>0: f["peso_bruto_toneladas"]=round(float(peso),1)
    teu=r.get("TEU")
    if teu and teu>0: f["TEU"]=round(float(teu),1)
    qt=r.get("QTCarga")
    if qt and qt>0: f["qtd_unidades_carga"]=int(qt)
    try: f["periodo"]=f'{MES.get(int(r.get("Mes_num")),"")} de {int(r.get("Ano"))}, {DOW.get(int(r.get("DiaSemana")),"")}'
    except Exception: pass
    if na(r.get("Carga Geral Acondicionamento")): f["acondicionamento"]=r.get("Carga Geral Acondicionamento")
    return {k:v for k,v in f.items() if v is not None and str(v).strip()!=""}

SYS = ("Você reescreve dados estruturados de uma atracação portuária brasileira em UMA única "
       "descrição fluente em português, com 2 a 4 frases. Use TODOS os campos fornecidos e NÃO invente "
       "nenhuma informação que não esteja nos dados. Não mencione tempos de operação ou de atracação. "
       "Responda APENAS com a descrição, sem rótulos nem aspas.")

runtime = boto3.client("bedrock-runtime", region_name=REGION)

def template_desc(f):
    """Deterministic fluent PT fallback (used only when Bedrock fails after retries)."""
    parts=[]
    loc=f"no porto de {f.get('porto','?')}"
    if f.get("municipio_uf"): loc+=f" ({f['municipio_uf']}"+(f", região {f['regiao']}" if f.get("regiao") else "")+")"
    op=f"operação de {f.get('tipo_operacao','movimentação')}"
    if f.get("navegacao"): op+=f" em navegação de {f['navegacao']}"
    parts.append(f"Atracação {loc}, {op}.")
    carga=[]
    if f.get("natureza_carga"): carga.append(f"natureza {f['natureza_carga']}")
    if f.get("mercadoria"): carga.append(f"principalmente {f['mercadoria']}")
    if f.get("mercadoria_conteinerizada"): carga.append(f"mercadoria conteinerizada {f['mercadoria_conteinerizada']}")
    if carga: parts.append("Carga de "+", ".join(carga)+".")
    od=[]
    if f.get("origem"): od.append(f"origem em {f['origem']}")
    if f.get("destino"): od.append(f"destino a {f['destino']}")
    extra=[]
    if f.get("peso_bruto_toneladas"): extra.append(f"peso bruto de {f['peso_bruto_toneladas']} toneladas")
    if f.get("TEU"): extra.append(f"{f['TEU']} TEU")
    if f.get("qtd_unidades_carga"): extra.append(f"{f['qtd_unidades_carga']} unidades")
    tail=", ".join(od+extra)
    if tail: parts.append(tail[0].upper()+tail[1:]+(f", em {f['periodo']}." if f.get("periodo") else "."))
    elif f.get("periodo"): parts.append(f"Em {f['periodo']}.")
    return " ".join(parts)

def call(facts, max_retries=8):
    prompt = SYS+"\n\nDados (JSON):\n"+json.dumps(facts, ensure_ascii=False)+"\n\nDescrição:"
    for a in range(max_retries):
        try:
            resp=runtime.converse(modelId=MODEL_ID,
                messages=[{"role":"user","content":[{"text":prompt}]}],
                inferenceConfig={"maxTokens":300,"temperature":0.0})
            return resp["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            # ThrottlingException, ServiceUnavailableException, transient errors -> backoff & retry
            if a==max_retries-1: return None
            time.sleep(min(2**a,30))
    return None

def serialize_df(df, tag, n=None, workers=8):
    cache=OUT/f"{tag}.parquet"
    if cache.exists():
        c=pd.read_parquet(cache)
        if n is None or len(c)>=n:
            print(f"[{tag}] cached {len(c)}",flush=True); return c
    sub=(df if n is None else df.iloc[:n]).reset_index(drop=True)
    ckpt=OUT/f"{tag}_ckpt.parquet"
    done={}
    if ckpt.exists():
        cdf=pd.read_parquet(ckpt)
        done={r.IDAtracacao:r.description for r in cdf.itertuples()}
        print(f"[{tag}] resume: {len(done)} already done",flush=True)
    facts_all=[row_facts(r) for _,r in sub.iterrows()]
    ids=sub["IDAtracacao"].tolist()
    todo=[(i,facts_all[i]) for i in range(len(sub)) if ids[i] not in done]
    texts={ids[i]:done.get(ids[i]) for i in range(len(sub))}
    nfail=0; cnt=0
    def work(item):
        i,f=item; r=call(f)
        return ids[i], (r if r else template_desc(f)), (0 if r else 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idd,desc,fail in ex.map(work, todo):
            texts[idd]=desc; nfail+=fail; cnt+=1
            if cnt%200==0:
                # checkpoint
                cur=pd.DataFrame({"IDAtracacao":list(texts.keys()),"description":list(texts.values())})
                cur=cur[cur.description.notna()]; cur.to_parquet(ckpt)
                print(f"[{tag}] {cnt}/{len(todo)} (template fallbacks so far: {nfail})",flush=True)
    out=pd.DataFrame({"IDAtracacao":ids})
    out["facts"]=[json.dumps(f,ensure_ascii=False) for f in facts_all]
    out["description"]=[texts[idd] for idd in ids]
    out.to_parquet(cache)
    if ckpt.exists(): ckpt.unlink()
    print(f"[{tag}] saved {len(out)}  template_fallbacks={nfail}",flush=True)
    return out

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "test"
    test=pd.read_parquet(ROOT/"data_input/test_strings.parquet")
    if mode=="test":
        r=test.iloc[0]; f=row_facts(r)
        print("FACTS:",json.dumps(f,ensure_ascii=False,indent=2))
        print("\nDESCRIPTION:\n",call(f))
        r=test.iloc[5]; f=row_facts(r)
        print("\n---FACTS2:",json.dumps(f,ensure_ascii=False))
        print("DESC2:\n",call(f))
    elif mode=="run":
        POOL=int(os.environ.get("POOL","1500"))
        train=pd.read_parquet(ROOT/"data_input/train_strings.parquet")
        tr_pool=train.sample(n=POOL,random_state=42).reset_index(drop=True)
        serialize_df(test,"test")
        serialize_df(tr_pool,"train_pool")
