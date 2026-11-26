# Do factor payoffs change between market regimes?

**Result:** no, not in the way I predicted. A hidden Markov model on VIX, the credit spread and the market's 12-month return splits 2003-2020 into stress and calm months, but no factor's payoff differs between them with |t| >= 2, and quality and beta moved the opposite way to my prediction in all 6 stress episodes. The main reason: the model's "stress" state measures **high volatility, not falling markets**, and many of its stress months are rebounds, when high-beta and junk stocks rally. Removing each stock's market exposure halves beta's flip and removes quality's. Inside a model, the regime adds nothing to XGBoost (paired t 0.78). A linear model with regime interactions improves (paired t 2.27), but the gain comes from the two validation years with stress months: leaving out 2016 or 2020 takes it to t 1.58 / 1.61. Neither model's long/short makes money. The test was pre-registered on 2026-11-25, before any result.

This is the follow-up to [does-option-skew-predict-stock-returns](https://github.com/Matthijs227/does-option-skew-predict-stock-returns), on the same stock panel.

## Hypothesis

Two critiques of my own first model:

1. Every feature is ranked within each month, so the model cannot see the market regime it is in.
2. News hits stocks differently depending on the regime.

- **Stress** = uncertainty: money flees to safe, defensive stocks (quality, profitability, low beta).
- **Calm** = risk-on: money chases momentum and high beta.

If that is right, a feature's payoff should differ between stress and calm months, in the directions below.

## Pre-registration (2026-11-25, before any result)

| | |
|---|---|
| regime model | 2-state Gaussian hidden Markov model on three monthly inputs: VIX, the Baa minus 10-year Treasury credit spread, and the US market's 12-month return; data from 1990 |
| no look-ahead | refit every January on data up to the previous December; filtered probabilities only (each month labelled with data up to that month); stress = the state with the higher mean VIX; a month is stress if p(stress) > 0.5 |
| features | 9, each ranked within the month: value (book/price), gross profitability, quality, momentum (12-1), earnings surprise, asset growth, operating profitability, size, beta (CAPM, past year of daily returns) |
| stress predictions | quality IC > 0, profitability IC > 0, value IC < 0, beta IC < 0 (low beta wins: flight to safety) |
| calm predictions | momentum IC > 0, value IC < 0, beta IC > 0 (high beta wins: risk-on) |
| regime claim | the difference stress minus calm: quality and profitability stress > calm; momentum and beta calm > stress; value: no flip predicted |
| pass (raw signals) | over all 215 months 2003-2020: the difference has t >= 2 over months, AND the right sign in every stress episode but one (an episode = a run of consecutive stress months) |
| trap check | the beta, quality and value stress cells must survive a beta-adjusted target (a falling market alone makes low beta look good) |
| model test | XGBoost on 9 features + p(stress) vs XGBoost on 9 features; train 2003-2012, early stopping 2013-2015, validation 2016-2020; pass = paired month-by-month t >= 2 |
| risk | every model gets a risk report on its decile long/short: worst month, expected shortfall, maximum drawdown, the result without its best year, and a year-block bootstrap. Reported next to the result, not a pass/fail on its own |
| holdout | 2021-2025 stays sealed |

Two additions, both reported in full below. The same day's analysis plan, written before any result, also ran a linear model with 9 feature x p(stress) interactions next to the XGBoost test. Leaving each validation year out in turn was added **after** the first model result, as a robustness check.

## Data

- **Stocks:** JKP global factor data (USA) via WRDS, 2003-2020. Common shares, primary listing, price >= $5, larger than the smallest 20% each month; 326 corrupt return records (beyond +-100% in a month) removed. 608,633 stock-months, 215 months, about 2,700 stocks a month. The same panel as the skew study, plus beta (JKP `beta_252d`, 97.4% coverage).
- **Target:** the stock's excess return over the next month, ranked within the month.
- **Regime inputs**, each value known on its date, monthly from 1990: VIX month-end close (Cboe), Baa minus 10-year Treasury yield (Federal Reserve H.15, FRED series BAA10Y), US market 12-month total return (Kenneth French data library).
- **2021-2025 is held out and has not been opened.**

## Method

- **Regime:** 2-state Gaussian HMM (hmmlearn), refit every January on all months before that year. Each month gets its **filtered** probability: the model sees data up to and including that month only. The stress state is re-identified at every refit as the one with the higher mean VIX, because an HMM can swap its labels.
- **IC** = Spearman rank correlation between a feature and next-month returns, computed once per month. Stress vs calm difference: Welch t over months.
- **The honest sample is episodes, not months.** The stress state is sticky: 120 stress months come in 6 episodes, one of them 78 months long. So the pass rule also checks the sign in every episode.
- **Beta-adjusted return:** next-month return minus the stock's raw beta times next month's market excess return. A rank target is not beta-adjusted, so in a falling month low beta ranks top without any stock picking.
- **Models:** train 2003-2012, early-stopping slice 2013-2015, validation 2016-2020. XGBoost (max depth 3, learning rate 0.05, early stopping 50) with 9 features vs 9 + p(stress). p(stress) enters raw: it is the same for every stock in a month, so a within-month rank would be constant. A linear model cannot use it as a plain column (it shifts every stock in a month by the same amount), so its regime version gets 9 interactions, feature x p(stress).
- **Risk:** decile long/short of each model's prediction (top 10% minus bottom 10%, equal weight, monthly, before costs): Sharpe, worst month, expected shortfall (mean of the worst 5% of months), maximum drawdown, the mean without the best year, and a bootstrap that draws whole years with replacement (2,000 draws).

## Results

**The regime model.** Fitted on 1990-2002, the stress state has mean VIX 25.3, credit spread 2.41 and 12-month market return +7%; the calm state 15.4, 1.75 and +17%. Stress episodes, 2003-2020:

| episode | months | length |
|---|---|---|
| 1 | 2003-01 to 2004-03 | 15 |
| 2 | 2007-07 to 2013-12 | 78 |
| 3 | 2015-01 | 1 |
| 4 | 2015-08 to 2016-08 | 13 |
| 5 | 2018-12 to 2019-02 | 3 |
| 6 | 2020-02 to 2020-12 | 11 |

Training (2003-2012) holds 81 stress months but only 2 episodes. Spring 2009 (the junk rally) is inside episode 2.

**Table 1: raw signals, stress vs calm** (all 215 months: 120 stress, 95 calm)

| feature | stress IC | calm IC | stress minus calm | t | predicted |
|---|---|---|---|---|---|
| value (be_me) | -0.0008 | -0.0038 | +0.0030 | 0.20 | no flip |
| profitability (gp_at) | 0.0280 | 0.0231 | +0.0048 | 0.44 | stress > calm |
| quality (qmj) | 0.0108 | 0.0253 | -0.0144 | -1.22 | stress > calm |
| momentum (ret_12_1) | -0.0044 | 0.0200 | -0.0244 | -1.35 | calm > stress |
| beta (beta_252d) | 0.0094 | -0.0270 | +0.0364 | 1.64 | calm > stress |
| earnings surprise | -0.0004 | 0.0089 | -0.0093 | -1.49 | |
| asset growth | -0.0056 | 0.0047 | -0.0102 | -1.11 | |
| operating profitability | 0.0142 | 0.0316 | -0.0174 | -1.67 | |
| size | 0.0136 | 0.0316 | -0.0181 | -1.23 | |

No difference reaches |t| >= 2. **The regime test fails.** Quality and beta point the opposite way to the prediction.

**Table 2: the episode check** (each stress episode's mean IC minus the calm mean)

| episode | quality | profitability | value | beta | momentum |
|---|---|---|---|---|---|
| 1 | -0.0330 | 0.0031 | 0.0476 | 0.0213 | -0.0629 |
| 2 | -0.0066 | 0.0014 | -0.0000 | 0.0240 | -0.0108 |
| 3 | -0.0539 | -0.0119 | -0.0075 | 0.2211 | -0.1376 |
| 4 | -0.0106 | -0.0083 | 0.0227 | 0.0175 | -0.0359 |
| 5 | -0.0538 | 0.0247 | -0.0799 | 0.1821 | -0.0582 |
| 6 | -0.0366 | 0.0465 | -0.0396 | 0.1185 | -0.0360 |
| **right sign** | **0 of 6** | **4 of 6** | (no flip predicted) | **0 of 6** | **6 of 6** |

Quality and beta are wrong in every episode: systematic, not one bad episode. Momentum has the predicted sign in all 6 but fails the t half of the rule (t -1.35).

**Table 3: the beta trap** (Table 1 on the beta-adjusted return)

| feature | stress IC | calm IC | difference (raw) | difference (adjusted) | t (adjusted) |
|---|---|---|---|---|---|
| beta | -0.0442 | -0.0637 | +0.0364 | +0.0195 | 1.13 |
| quality | 0.0279 | 0.0312 | -0.0144 | -0.0034 | -0.32 |
| value | -0.0063 | -0.0027 | +0.0030 | -0.0037 | -0.23 |

Once the market part is removed, all three stress cells have the predicted sign, but they have the same sign in calm months: low beta and quality win in both regimes.

**Table 4: does the regime help a model?** (validation 2016-2020, 59 months, mean IC)

| | 9 features | 9 features + regime | paired difference | paired t |
|---|---|---|---|---|
| XGBoost (+ p(stress)) | 0.0113 (t 0.77) | 0.0139 (t 0.94) | 0.0026 | 0.78 |
| linear (+ 9 interactions) | 0.0065 (t 0.45) | 0.0148 (t 1.06) | 0.0083 | 2.27 |

Paired t with one validation year left out:

| year left out | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|
| linear | 1.58 | 2.20 | 2.55 | 2.24 | 1.61 |
| XGBoost | 0.80 | 0.85 | 0.92 | 0.57 | 0.20 |

The linear model's regime weights (feature weight = b + c x p(stress)), calm to stress: momentum 0.022 to 0.003, operating profitability 0.034 to 0.008, beta -0.018 to -0.004, asset growth 0.005 to -0.010.

**Table 5: risk of each decile long/short** (monthly, before costs; the worst month is the month after its formation date)

| | mean / month | Sharpe | t | worst month | ES 5% | max drawdown | mean without best year (2018) | bootstrap share <= 0 |
|---|---|---|---|---|---|---|---|---|
| XGBoost 9 | -0.08% | -0.06 | -0.13 | -11.5% (Nov 2020) | -9.2% | -37.8% | -0.32% | 0.59 |
| XGBoost 9 + p(stress) | +0.11% | 0.08 | 0.18 | -8.6% (Nov 2019) | -7.8% | -32.1% | -0.10% | 0.37 |
| linear 9 | -0.27% | -0.20 | -0.45 | -11.2% (Nov 2020) | -9.1% | -35.8% | -0.52% | 0.74 |
| linear 9 + interactions | +0.09% | 0.07 | 0.16 | -8.4% (Nov 2019) | -7.8% | -22.9% | -0.11% | 0.35 |

## Reading

- **The stress state measured volatility, not crashes.** Volatility stays high after a market bottom, and the HMM switches on fast and off slowly, so many "stress" months are rebounds (2003, spring 2009, most of 2020). In a rebound, high-beta and low-quality stocks rally hardest. That explains why beta and quality went the wrong way in every episode, and why the beta-adjusted return (Table 3) shrinks the effect. The flight-to-safety idea was about crashes. This test did not isolate crashes, so it neither confirms nor kills that idea.
- **Once beta is removed, the factors behave about the same in both regimes.** Low beta beating its beta and quality beating junk are known all-weather patterns (Frazzini and Pedersen, 2014; Asness, Frazzini and Pedersen, 2019). Here they are slightly weaker in stress, not stronger.
- **The linear gain sits where the mechanism says it should, and that is also its weakness.** Interactions only change predictions when p(stress) is high, and the paired t falls below 2 when either validation year with stress months (2016, 2020) is left out. That is 2 episodes. The weights it learned (momentum and profitability switched mostly off in stress, the low-beta tilt mostly off) match Tables 1-2 and the known momentum crashes in rebounds (Daniel and Moskowitz, 2016). XGBoost relied on p(stress) (importance 0.12) but gained nothing: with 2 training episodes, the more flexible model likely fit their quirks.
- **No money either way.** Both long/shorts are near zero before costs. The regime versions had smaller drawdowns, mostly by avoiding November 2020 (the vaccine rotation, the worst month of both models without the regime). That is one month.

## Limitations

- 6 stress episodes in 2003-2020, 2 in training and about 3 in validation. Any regime result on this sample is weak evidence either way. More episodes need a longer history (factor returns back to the 1960s or 1926, with realized volatility in place of VIX) or other countries; a global crisis still counts once.
- One 5-year validation path; equal-weight deciles; no trading costs.
- A crash-vs-rebound split is a different hypothesis. It needs its own pre-registration and a crash definition that is knowable at each month end.

## Reproduce

Needs a WRDS account with access to the JKP global factor data. Data cannot be redistributed, so none is included. The regime inputs are public and downloaded by step 3.

```
pip install -r requirements.txt
# put the JKP USA download at data/jkp_usa.parquet (columns listed in data_prep/1_prep_panel.py)
set WRDS_USERNAME=your_username
python data_prep/1_prep_panel.py
python data_prep/2_add_beta.py
python data_prep/3_regime_inputs.py
python regime_test.py
```

hmmlearn prints "Model is not converging" on a few yearly refits; it does not change the results. A full run takes a few minutes.

## Credits

Research question, hypothesis, pre-registration and analysis code (`regime_test.py`): Matthijs Kok. The data scripts in `data_prep/` were written with AI assistance, as was the tidying of comments and print layout in `regime_test.py`. No data is redistributed (WRDS licence).
