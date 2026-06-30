# Traditional ML vs. LLMs for Ship Turnaround Time at Brazilian Ports

Code, metrics, predictions and the KDMiLe paper for a controlled comparison of **traditional
machine learning** against **large language models** on predicting ship turnaround times
(operation time `TOperacao` and total moored time `TAtracado`) from Brazilian National Waterway
Transportation Agency (ANTAQ) data, using a leakage-controlled, berthing-level pipeline.

**Paper:** `paper/paper.pdf` (KDMiLe). Authors: E. C. B. Pacheco and J. A. Ramos.

## TL;DR results (200-berthing held-out test, 20 seeds, paired Wilcoxon + Cohen's d)

- **Tuned XGBoost wins decisively:** R² ≈ **0.73** on both targets.
- **Every LLM regime stays at R² ≤ 0** (no better than the mean): frontier LLMs (Qwen3-32B,
  DeepSeek-V3) prompted **zero-/5-/20-shot** (in-context examples do **not** help), and a small
  **generative** LLM (Qwen2.5-1.5B) fine-tuned on 100–1500 serialized records.
- **An encoder + regression head** (the *same* multilingual MiniLM backbone used for the ML
  embeddings, fine-tuned end-to-end) is the **only text regime to exceed R² = 0**, rising with
  data (without saturating) to R² ≈ **0.41–0.43** at 12 000 examples — competitive with KNN/SVM
  at matched data, but still below XGBoost.
- **Data-matched curves (N = 100–6000)** for all 7 families: XGBoost dominates at every budget
  and is ~8× more data-efficient than the encoder; linear/MLP overfit catastrophically at small
  N in the 1606-dim feature space; the decoder stays ≈ 0.
- **Cost:** XGBoost trains in minutes and infers in ~3 ms / 200 cases; the generative LLM costs
  ~18 min to train and ~45 s to infer 200 cases — the most expensive and least accurate.

## Repository layout

```
paper/        paper.tex, refs.bib, kdmile.cls/.bst, figures, paper.pdf
notebooks/    2_eda... (build leakage-free train/test), 3_train_ml..., 4_llm_inference, 5_comparisons
scripts/      finetune.py (generative LLM LoRA), encoder_ft.py (encoder+reg head),
              serialize.py (case -> PT description via Bedrock), ml_curve*.py (data-matched ML curves),
              make_fig*.py (figures), run_decoder_durable.sh (chunked-restart loop)
metrics/      per-seed RMSE/R²/MAE CSVs (ML_models/, LLM/) + timing + data-efficiency curves
preds/        per-seed test predictions (.npy) for every model/regime/seed
serialized/   train_pool.parquet (12k LLM-serialized descriptions) + test.parquet (200)
consolidated_table.csv, datamatched_full.csv   summary tables
```

## Reproducing

1. **Data.** The cleaned ANTAQ dataset (2018–2024) and the leakage-controlled pipeline are
   released via Zenodo (see the paper / the prior audit work). `notebooks/2_*.ipynb` rebuilds
   the berthing-level `train_strings.parquet` / `test_strings.parquet` (the 29 MB train file is
   not committed here — regenerate it or fetch from Zenodo).
2. **ML models + data-efficiency curves.** `notebooks/3_*.ipynb`, then `scripts/ml_curve*.py`.
3. **Prompted LLMs.** `notebooks/4_*.ipynb` (Amazon Bedrock; set AWS credentials in a local
   `.env` — never commit it).
4. **Generative-LLM fine-tuning.** `scripts/serialize.py` (descriptions) then
   `scripts/finetune.py` (Qwen2.5-1.5B + LoRA, Apple MPS / CUDA).
5. **Encoder + regression head.** `scripts/encoder_ft.py`.
6. **Comparison + stats + figures.** `notebooks/5_*.ipynb`, `scripts/make_fig*.py`.

Environment: Python 3.11, `torch`, `transformers`, `peft`, `xgboost`, `scikit-learn`,
`sentence-transformers`, `pandas`, `pyarrow`, `boto3` (for Bedrock).

## Notes

- **No secrets are committed.** AWS credentials live only in a local `.env` (git-ignored).
- Fine-tuning was run on an Apple M5 Pro (MPS, bf16); prompted LLMs via Amazon Bedrock.
- All metrics are reported per seed; comparisons use the paired Wilcoxon signed-rank test and
  pooled Cohen's d.
