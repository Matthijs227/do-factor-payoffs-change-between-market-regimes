# Step 3 of the data build: the three regime inputs, one row per month end, each value known ON that date.
# Run from the repo folder:  python data_prep/3_regime_inputs.py      (public data only, no account needed)
#   vix      VIX close on the last trading day of the month                    (Cboe)
#   credit   Moody's Baa yield minus the 10-year Treasury yield, last day      (FRED series BAA10Y, Fed H.15 data)
#   mkt_12m  total US market return over the 12 months up to this month       (Kenneth French: Mkt-RF + RF, compounded)
#   mkt_1m   the month's own market return (spare column, not used in the test)
#   target_only_mkt_rf_next = NEXT month's market excess return. A FUTURE value: used only to build the
#            beta-adjusted return in step 5 of regime_test.py, never as a feature or a regime input.
# Output: data/regime_inputs.parquet
import io
import os
import urllib.request
import zipfile
import pandas as pd

def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()

CBOE = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cboe.com/"}   # Cboe refuses requests without these

v = pd.read_csv(io.BytesIO(get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", CBOE)),
                parse_dates=["DATE"], date_format="%m/%d/%Y")
vix = v.set_index("DATE")["CLOSE"].resample("ME").last().rename("vix")

c = pd.read_csv(io.BytesIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y")),
                parse_dates=["observation_date"], na_values=".")
credit = c.dropna().set_index("observation_date")["BAA10Y"].resample("ME").last().rename("credit")

z = zipfile.ZipFile(io.BytesIO(get("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")))
lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
rows = []
for ln in lines:
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) == 5 and len(parts[0]) == 6 and parts[0].isdigit():
        rows.append((parts[0], float(parts[1]), float(parts[4])))
    if "Annual" in ln:
        break                                   # the monthly block ends where the annual block starts
ff = pd.DataFrame(rows, columns=["ym", "mkt_rf", "rf"])
ff["eom"] = pd.to_datetime(ff["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
ff = ff.set_index("eom")
mkt = (ff["mkt_rf"] + ff["rf"]) / 100
mkt_1m = mkt.rename("mkt_1m")
mkt_12m = ((1 + mkt).rolling(12).apply(lambda x: x.prod(), raw=True) - 1).rename("mkt_12m")
nxt = (ff["mkt_rf"] / 100).shift(-1).rename("target_only_mkt_rf_next")

out = pd.concat([vix, credit, mkt_12m, mkt_1m, nxt], axis=1)
out.index.name = "eom"
out = out[out.index >= "1990-01-31"]
os.makedirs("data", exist_ok=True)
out.reset_index().to_parquet("data/regime_inputs.parquet", index=False)
print(out.shape, out.index.min().date(), "->", out.index.max().date())
print(out.loc[["2008-10-31", "2009-03-31", "2017-06-30", "2020-03-31"]].round(3).to_string())
