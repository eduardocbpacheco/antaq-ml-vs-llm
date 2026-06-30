#!/bin/zsh
cd ~/antaq_work
PREDS="/Users/eduardopacheco/Desktop/Documentos/USP/Portos/ANTAQ_ML_vs_LLM/preds"
PY="$HOME/antaq_work/ftvenv/bin/python"
i=0
while true; do
  n=$(ls "$PREDS" | grep -c qwen25_1p5b_ft)
  if [ "$n" -ge 200 ]; then echo "[loop] ALL DONE — $n/200" >> ~/antaq_work/decoder.log; break; fi
  i=$((i+1)); echo "[loop] iter $i ($n/200) $(date '+%H:%M:%S')" >> ~/antaq_work/decoder.log
  CHUNK=5 $PY -u finetune.py >> ~/antaq_work/decoder.log 2>&1
  sleep 2
done
echo "[loop] FINISHED $(date '+%H:%M:%S')" >> ~/antaq_work/decoder.log
