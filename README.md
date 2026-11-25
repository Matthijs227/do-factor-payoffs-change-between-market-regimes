# Do factor payoffs change between market regimes?

**Status: in progress.** The hypothesis and pre-registration below were written on 2026-11-25, before any result was computed. Code and full results will be posted here soon.

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

## Data

- **Stocks:** JKP global factor data (USA) via WRDS, 2003-2020. Common shares, primary listing, price >= $5, larger than the smallest 20% each month: 608,633 stock-months, 215 months, about 2,700 stocks a month. The same panel as the skew study, plus beta.
- **Target:** the stock's excess return over the next month, ranked within the month.
- **Regime inputs**, each value known on its date: VIX month-end close (Cboe), Baa minus 10-year Treasury yield (Federal Reserve H.15), US market 12-month return (Kenneth French data library).

## Method

- IC = Spearman rank correlation between a feature and next-month returns, computed once per month.
- The t-stat counts months, not stock-months: all stocks in a month share that month's news.
- A stress regime lasts many months, so the honest sample size for the regime claim is the number of stress episodes, which is why the pass rule also checks every episode.

## Credits

Research question, hypothesis, pre-registration and analysis code: Matthijs Kok. The data-pull scripts were written with AI assistance. No data is redistributed (WRDS licence).
