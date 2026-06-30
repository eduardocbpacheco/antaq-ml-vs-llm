import serialize as S
from pathlib import Path
import pandas as pd
S.OUT = Path("/Users/eduardopacheco/antaq_work/serialized"); S.OUT.mkdir(exist_ok=True,parents=True)
test = pd.read_parquet(S.ROOT/"data_input/test_strings.parquet")
print(f"regenerating {len(test)} test descriptions...", flush=True)
out = S.serialize_df(test, "test", workers=8)
print("TEST REGEN DONE; fallbacks:", int(out.description.isna().sum()), "rows:", len(out), flush=True)
