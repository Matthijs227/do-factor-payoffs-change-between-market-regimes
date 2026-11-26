# Do factor payoffs change between market regimes? 2003-2020, US stocks.
#
# Hypothesis (written before any result): news hits stocks differently depending on the regime.
#   stress = uncertainty: money flees to safe, defensive stocks (quality, profitability, low beta)
#   calm   = risk-on: money chases momentum and high beta
#
# Pre-registration (2026-11-25, before any result; full table in the README):
#   regime   2-state Gaussian HMM on monthly VIX, Baa-10y credit spread, US market 12-month return (from 1990),
#            refit every January on past data only, filtered probabilities only, stress = state with higher mean VIX,
#            a month is stress if p(stress) > 0.5
#   cells    stress: quality > 0, profitability > 0, value < 0, beta < 0 | calm: momentum > 0, value < 0, beta > 0
#            regime claim = stress minus calm: quality, profitability stress > calm; momentum, beta calm > stress
#   pass     raw signals, all 215 months: difference t >= 2 over months AND right sign in every stress episode but one
#   trap     beta, quality and value stress cells must survive a beta-adjusted return
#   model    XGBoost 9 features + p(stress) vs 9 features, valid 2016-2020, paired t >= 2
#            (the analysis plan, same day, also ran linear 9 features vs 9 features + 9 feature x p(stress) interactions)
#   risk     risk report on each model's decile long/short; reported, not a pass/fail on its own
#   sealed   2021-2025 holdout not opened
#
# Data: built by the scripts in data_prep/ (see README).
#   data/model_panel_beta.parquet  one row = one US stock at one month end (eom), 2003-01 to 2020-11, 608,633 rows;
#                                  ret_next = the stock's excess return over the NEXT month
#   data/regime_inputs.parquet     one row per month end from 1990, each value known on that date

import numpy as np
import pandas as pd
import xgboost as xgb
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression

panel = pd.read_parquet("data/model_panel_beta.parquet")
regime = pd.read_parquet("data/regime_inputs.parquet")


def ic_by_month(df,pred_col,ret_col="ret_next"):
    out= {}
    for month, group in df.groupby("eom"):
        out[month] = group[pred_col].corr(group[ret_col], method="spearman")
    return pd.Series(out)

def report(name, ic):
    t = ic.mean() / (ic.std() / len(ic) ** 0.5)
    print(f"{name:12s} mean IC {ic.mean():.4f}   sd {ic.std():.4f}   t {t:.2f}   months {len(ic)}   share of months > 0: {(ic > 0).mean():.2f}")

def risk_report(name, pnl):
    mean= pnl.mean()
    sd= pnl.std()
    sharpe= mean/sd*np.sqrt(12)
    t= mean/(sd/len(pnl)**0.5)
    worst_month= pnl.min()
    which_month= pnl.idxmin()
    k= round(0.05*len(pnl))
    es= pnl.nsmallest(k).mean()
    pnl_total= pnl.cumsum()
    max_pnl= pnl_total.cummax()
    drawdown= pnl_total-max_pnl
    max_drawdown= drawdown.min()
    pnl_year= pnl.groupby(pnl.index.year).sum()
    best_pnl_year= pnl_year.idxmax()
    mean_pnl_exbest= pnl[pnl.index.year!=best_pnl_year].mean()
    # block bootstrap: draw whole years with replacement (neighbouring months are not independent)
    rng= np.random.default_rng(0)
    years= pnl_year.index
    draws=[]
    for i in range(2000):
        pick= rng.choice(years,size=len(years),replace=True)
        parts= []
        for y in pick:
            parts.append(pnl[pnl.index.year==y])
        draws.append(pd.concat(parts).mean())
    share_neg= (np.array(draws)<=0).mean()
    print(f"{name:10s} mean {mean:.4f} sd {sd:.4f} Sharpe {sharpe:.2f} t {t:.2f}")
    print(f"   worst {worst_month:.4f} ({which_month.date()}) | ES5% {es:.4f} | maxDD {max_drawdown:.4f}")
    print(f"   ex best {best_pnl_year} {mean_pnl_exbest:.4f} | boot share<=0 {share_neg:.3f}")


# ---------------------------------------------------------------
# Step 1: target and 9 features, all ranked within each month (-0.5 worst .. +0.5 best)
# ---------------------------------------------------------------
panel["size"]= np.log(panel["me"])
features= ["be_me","gp_at","qmj","ret_12_1","niq_su","at_gr1","ope_be","size","beta_252d"]
panel["target"]= panel.groupby("eom")["ret_next"].rank(pct=True)-0.5
panel["beta_real"]=panel["beta_252d"]          # raw beta, kept for the beta-adjusted return in step 5
for f in features:
    panel[f]= panel.groupby("eom")[f].rank(pct=True)-0.5
print("beta coverage:", panel["beta_252d"].notna().mean())

# ---------------------------------------------------------------
# Step 2: one HMM fit on 1990-2002, to see what the two states are
# ---------------------------------------------------------------
early= regime[regime["eom"]<"2003-01-01"]
X= early[["vix","credit","mkt_12m"]].dropna()
hmm= GaussianHMM(n_components=2,covariance_type="full",n_iter=200, random_state=0)
hmm.fit(X.values)
print("\nstate means (vix, credit, mkt_12m):\n", hmm.means_)
vix_stress= np.argmax(hmm.means_[:,0])
print("stress state:", vix_stress)

# ---------------------------------------------------------------
# Step 3: p(stress) for every month 2003-2020 from the PAST only
#   refit each January on all months before that year; for each month, predict_proba on the rows up to and
#   including that month and keep the last row (= the filtered probability, what was knowable at that month)
# ---------------------------------------------------------------
full= regime.set_index("eom")[["vix","credit","mkt_12m"]].dropna()
p= {}
for year in range(2003,2021):
    past= full[full.index.year<year]
    hmm= GaussianHMM(n_components=2,covariance_type="full",n_iter=200, random_state=0)
    hmm.fit(past.values)
    vix_stress= np.argmax(hmm.means_[:,0])
    months= full.index[full.index.year== year]
    for m in months:
        upto = full[full.index<=m]
        p[m]= hmm.predict_proba(upto.values)[-1,vix_stress]
p_stress= pd.Series(p)
stress= p_stress>0.5
start_stress= stress&~stress.shift(1, fill_value=False)
episode= start_stress.cumsum().where(stress)
print("\nstress episodes:")
for e in sorted(episode.dropna().unique()):
    ep_months= episode[episode==e].index
    print(f"   {int(e)}  {ep_months.min().date()} to {ep_months.max().date()}  ({len(ep_months)} months)")

# ---------------------------------------------------------------
# Step 4: the pre-registered test on raw signals (no model), all 215 months
# ---------------------------------------------------------------
ics= {}
for f in features:
    ics[f]= ic_by_month(panel, f)
ic= pd.DataFrame(ics)
ic["p_stress"]= p_stress
ic["stress"]= stress
ic["episode"]=episode
stress_months= ic[ic["stress"]]
calm_months= ic[~ic["stress"]]
mean_ic_stress= stress_months[features].mean()
mean_ic_calm= calm_months[features].mean()
difference_ic= mean_ic_stress-mean_ic_calm
t= (mean_ic_stress-mean_ic_calm)/np.sqrt(stress_months[features].var()/stress_months[features].count()+calm_months[features].var()/calm_months[features].count())
df_ic= pd.DataFrame({"stress":mean_ic_stress,"calm":mean_ic_calm,"diff":difference_ic,"t":t})
print("\nTable 1: mean IC in stress and calm months, difference and its t")
print(df_ic.round(4))
cells=["qmj","gp_at","be_me","beta_252d","ret_12_1"]
mean_episode= stress_months.groupby("episode")[cells].mean()-mean_ic_calm[cells]
print("\nTable 2: each stress episode's mean IC minus the calm mean")
print(mean_episode.round(4))

# ---------------------------------------------------------------
# Step 5: the beta trap. A rank target is not beta-adjusted: in a falling month low beta ranks top for free.
#   bet_adj_ret = ret_next - raw beta x next month's market excess return
# ---------------------------------------------------------------
mkt_next= regime.set_index("eom")["target_only_mkt_rf_next"]
panel["bet_adj_ret"]= panel["ret_next"]-panel["beta_real"]*panel["eom"].map(mkt_next)
print("\nbeta-adjusted return coverage:", panel["bet_adj_ret"].notna().mean())
cells_adj= ["beta_252d","qmj","be_me"]
ics_adj={}
for c in cells_adj:
    ics_adj[c]= ic_by_month(panel,c,"bet_adj_ret")
ic_adj= pd.DataFrame(ics_adj)
ic_adj["p_stress"]= p_stress
ic_adj["stress"]= stress
ic_adj["episode"]=episode
stress_months_adj= ic_adj[ic_adj["stress"]]
calm_months_adj= ic_adj[~ic_adj["stress"]]
mean_ic_stress_adj= stress_months_adj[cells_adj].mean()
mean_ic_calm_adj= calm_months_adj[cells_adj].mean()
difference_ic_adj= mean_ic_stress_adj-mean_ic_calm_adj
t_adj= (mean_ic_stress_adj-mean_ic_calm_adj)/np.sqrt(stress_months_adj[cells_adj].var()/stress_months_adj[cells_adj].count()+calm_months_adj[cells_adj].var()/calm_months_adj[cells_adj].count())
df_ic_adj= pd.DataFrame({"stress":mean_ic_stress_adj,"calm":mean_ic_calm_adj,"diff":difference_ic_adj,"t":t_adj})
print("\nTable 3: Table 1 on the beta-adjusted return")
print(df_ic_adj.round(4))

# ---------------------------------------------------------------
# Step 6: does the regime help a MODEL? train 2003-2012, early stopping 2013-2015, valid 2016-2020
#   p(stress) goes in raw (it is the same for every stock in a month, so a within-month rank would be constant).
#   A linear model cannot use it as a plain column (it shifts every stock in a month equally), so the linear
#   version gets interactions: feature x p(stress). Trees build interactions themselves.
# ---------------------------------------------------------------
panel["p_stress"]= panel["eom"].map(p_stress)
inter=[]
for f in features:
    panel[f+"_x"]= panel[f]*panel["p_stress"]
    inter.append(f+"_x")
train= panel[panel["eom"]<"2013-01-01"].copy()
stop= panel[panel["eom"].between("2013-01-01","2016-01-01")].copy()
valid=panel[panel["eom"]>"2016-01-01"].copy()

print()
lin9= LinearRegression()
lin9.fit(train[features].fillna(0),train["target"])
valid["pred_lin9"]= lin9.predict(valid[features].fillna(0))
print("lin9 coefficients:", dict(zip(features, lin9.coef_.round(4))))
report("linear9", ic_by_month(valid, "pred_lin9"))

lin9r= LinearRegression()
lin9r.fit(train[features+inter].fillna(0),train["target"])
valid["pred_lin9r"]=lin9r.predict(valid[features+inter].fillna(0))
print("lin9r coefficients:",dict(zip(features+inter,lin9r.coef_.round(4))))
report("linear9r",ic_by_month(valid,"pred_lin9r"))

model9= xgb.XGBRegressor(n_estimators=2000,max_depth=3, learning_rate=0.05, early_stopping_rounds=50,random_state=0)
model9.fit(train[features],train["target"], eval_set=[(stop[features],stop["target"])],verbose=False)
print("xgb9 trees kept:", model9.best_iteration)
valid["pred_xgb9"] = model9.predict(valid[features])
report("xgboost9", ic_by_month(valid, "pred_xgb9"))
print("importance:", dict(zip(features, model9.feature_importances_.round(3))))

model9r= xgb.XGBRegressor(n_estimators=2000, max_depth=3,learning_rate=0.05,early_stopping_rounds=50,random_state=0)
model9r.fit(train[features+["p_stress"]],train["target"],eval_set=[(stop[features+["p_stress"]],stop["target"])],verbose=False)
print("xgb9r trees kept:",model9r.best_iteration)
valid["pred_xgb9r"]=model9r.predict(valid[features+["p_stress"]])
report("xgboost9r", ic_by_month(valid,"pred_xgb9r"))
print("importance:",dict(zip(features+["p_stress"],model9r.feature_importances_.round(3))))

print("\nTable 4: paired test (with regime minus without), then each valid year left out in turn")
paired_xgb= ic_by_month(valid,"pred_xgb9r")-ic_by_month(valid,"pred_xgb9")
paired_lin= ic_by_month(valid,"pred_lin9r")-ic_by_month(valid,"pred_lin9")
report("xgb paired", paired_xgb)
report("lin paired", paired_lin)
for year in range(2016,2021):
    report(f"lin ex {year}",paired_lin[paired_lin.index.year !=year])
    report(f"xgb ex {year}",paired_xgb[paired_xgb.index.year !=year])

# ---------------------------------------------------------------
# Step 7: risk of each model's decile long/short (top 10% of the prediction minus bottom 10%, equal weight,
#   per month, before costs). The date shown for the worst month is the formation month; the loss is the month after.
# ---------------------------------------------------------------
valid["xgb9_ranked"]= valid.groupby("eom")["pred_xgb9"].rank(pct=True)
top= valid[valid["xgb9_ranked"]>0.9]
valid["xgb9_bottom"]=valid.groupby("eom")["pred_xgb9"].rank(pct=True)
bottom= valid[valid["xgb9_bottom"]<=0.1]
top_ret= top.groupby("eom")["ret_next"].mean()
bottom_ret= bottom.groupby("eom")["ret_next"].mean()
spread9= top_ret-bottom_ret

valid["xgb9r_ranked"]= valid.groupby("eom")["pred_xgb9r"].rank(pct=True)
top= valid[valid["xgb9r_ranked"]>0.9]
valid["xgb9r_bottom"]=valid.groupby("eom")["pred_xgb9r"].rank(pct=True)
bottom= valid[valid["xgb9r_bottom"]<=0.1]
top_ret= top.groupby("eom")["ret_next"].mean()
bottom_ret= bottom.groupby("eom")["ret_next"].mean()
spread9r= top_ret-bottom_ret

valid["lin9_ranked"]= valid.groupby("eom")["pred_lin9"].rank(pct=True)
top= valid[valid["lin9_ranked"]>0.9]
valid["lin9_bottom"]=valid.groupby("eom")["pred_lin9"].rank(pct=True)
bottom= valid[valid["lin9_bottom"]<=0.1]
top_ret= top.groupby("eom")["ret_next"].mean()
bottom_ret= bottom.groupby("eom")["ret_next"].mean()
spread9lin= top_ret-bottom_ret

valid["lin9r_ranked"]= valid.groupby("eom")["pred_lin9r"].rank(pct=True)
top= valid[valid["lin9r_ranked"]>0.9]
valid["lin9r_bottom"]=valid.groupby("eom")["pred_lin9r"].rank(pct=True)
bottom= valid[valid["lin9r_bottom"]<=0.1]
top_ret= top.groupby("eom")["ret_next"].mean()
bottom_ret= bottom.groupby("eom")["ret_next"].mean()
spread9rlin= top_ret-bottom_ret

print("\nTable 5: risk report of each decile long/short, valid 2016-2020")
risk_report("xgb9",spread9)
risk_report("xgb9r",spread9r)
risk_report("lin9",spread9lin)
risk_report("lin9r",spread9rlin)
