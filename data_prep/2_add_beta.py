# Step 2 of the data build: add the 9th feature, beta_252d = JKP's CAPM beta on the past year of daily returns,
# known at each month end. The sealed holdout only gets the column merged; nothing in it is analysed.
# Run from the repo folder:  python data_prep/2_add_beta.py
# Needs a WRDS account with JKP access: set the environment variable WRDS_USERNAME.
# Output: data/model_panel_beta.parquet and sealed/holdout_2021_2025_beta.parquet
import os
import pandas as pd
import wrds

db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
frames = []
for y in range(2003, 2026):
    q = f"""select permno, eom, beta_252d from contrib_global_factor.global_factor
            where excntry='USA' and common=1 and obs_main=1 and primary_sec=1 and permno is not null
              and eom >= '{y}-01-01' and eom <= '{y}-12-31'"""
    frames.append(db.raw_sql(q))
db.close()
b = pd.concat(frames, ignore_index=True)
b["eom"] = pd.to_datetime(b["eom"])
b["permno"] = b["permno"].astype("int64")
b["beta_252d"] = b["beta_252d"].astype("float64")
b = b.drop_duplicates(["permno", "eom"])
for src, dst in [("data/model_panel.parquet", "data/model_panel_beta.parquet"),
                 ("sealed/holdout_2021_2025.parquet", "sealed/holdout_2021_2025_beta.parquet")]:
    p = pd.read_parquet(src)
    out = p.merge(b, on=["permno", "eom"], how="left", validate="one_to_one")
    assert len(out) == len(p)
    out.to_parquet(dst, index=False)
    print(dst, out.shape, "beta coverage", round(out["beta_252d"].notna().mean(), 3))
