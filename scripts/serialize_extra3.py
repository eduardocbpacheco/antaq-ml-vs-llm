"""Serialize +12000 additional training berthings (disjoint from existing 12000-pool) -> 24000-pool."""
import serialize as S
import pandas as pd
train=pd.read_parquet(S.ROOT/"data_input/train_strings.parquet")
existing=set(pd.read_parquet(S.OUT/"train_pool.parquet")["IDAtracacao"].tolist())
extra=train[~train["IDAtracacao"].isin(existing)].sample(n=12000,random_state=45).reset_index(drop=True)
print(f"serializing {len(extra)} extra (disjoint from {len(existing)} existing)",flush=True)
S.serialize_df(extra,"train_pool_extra3",workers=10)
print("EXTRA3 DONE",flush=True)
