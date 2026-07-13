# Netflix Content Strategy Analysis

A decision-science analysis of Netflix's catalog (7,789 titles, 2008–2021), built around four testable business hypotheses rather than open-ended exploration.

## Business Context

Every catalog dollar Netflix spends acquiring or producing a title is a resource-allocation decision. This project treats the dataset as evidence for four decisions the content team is implicitly making:

| # | Decision Question |
|---|---|
| D1 | Is Netflix shifting investment toward TV Shows or Movies — and is that shift statistically real? |
| D2 | Which genres are structurally growing vs. shrinking, with significance, not eyeballing? |
| D3 | How concentrated is content sourcing by country — is that a strategic risk? |
| D4 | Can a title's target audience be predicted from metadata alone, at scale? |

## Hypotheses & Results

| Hypothesis | Test | Result |
|---|---|---|
| **H1** — TV-share trend is real, not noise | Linear regression, TV-Show share of yearly additions vs. year | **Not supported** — p = 0.924 |
| **H2** — Top genres show significant growth trends | Per-genre linear regression, 2015–2020 | **Supported** — 7 of 8 top genres significant (p < 0.05) |
| **H3** — Country sourcing is concentrated | Herfindahl-Hirschman Index (HHI) on primary country | **Supported** — HHI = 1,677 (US alone = 37% of catalog) |
| **H4** — Metadata predicts audience segment above baseline | Random Forest vs. majority-class baseline, held-out test set | **Supported** — 61% accuracy vs. 46% baseline (+15 pts) |

## Methodology Notes

- **Duration is split by type before comparison** — the raw `Duration` field mixes minutes (movies) and seasons (TV shows) in one text column; treating them as one numeric scale would make cross-category comparisons meaningless.
- **Trend claims are tested, not eyeballed.** Every "genre X is growing" or "TV shows are rising" statement is backed by a regression p-value, not a chart that looks like it's going up.
- **Clustering uses a justified k.** K-Means is run for k=2..6 and the silhouette score picks k=3, rather than assuming 3 clusters upfront. The resulting cohorts are named in business terms: *TV Series*, *International Co-Productions*, *Standard Single-Market Movies*.
- **The forecast is validated on a genuine holdout**, not training-set fit. The last 2 years are held out and never seen during fitting; the reported 21% MAPE is the honest, unseen-data error — a linear trend over-forecasts 2020, plausibly reflecting COVID-era production slowdowns.
- **The classifier's accuracy is reported against a baseline**, not in isolation — 61% only means something next to the 46% you'd get by always guessing the majority class ("Adults").

## Key Findings

1. **Content mix**: TV-Show share moved from ~27% (2015) to ~35% (2020), but the year-to-year trend is statistically indistinguishable from noise (p = 0.924) — don't over-commit production budget to a TV-Show pivot on this evidence alone.
2. **Genre growth**: International Movies, Dramas, and Comedies show statistically significant upward trends — the defensible genres for continued investment.
3. **Sourcing risk**: HHI of 1,677 (moderately concentrated) with 37% US dependency and 66.5% from the top 5 countries — a real diversification consideration.
4. **Audience-segment tagging**: A Random Forest classifier can auto-tag new titles by target audience at 61% accuracy (vs. 46% baseline), viable as first-pass triage with human review on lower-precision segments.
5. **Forecasting discipline**: Present catalog-growth forecasts as a range (±21% MAPE), not a single confident number.

## Notebook Structure

| Section | Content |
|---|---|
| 1–2 | Business context & hypotheses |
| 3–5 | Setup, data load, cleaning & feature engineering |
| 6 | H1 — content mix trend test |
| 7 | H2 — genre growth trend test |
| 8 | H3 — country concentration (HHI) |
| 9 | Audience segment × category association (chi-square) |
| 10 | Content clustering (K-Means, silhouette-selected k) |
| 11 | H4 — audience-segment classifier (Random Forest) |
| 12 | Growth forecast with holdout validation (MAPE) |
| 13 | Hypothesis test summary table |
| 14 | Quantified strategic recommendations |
| 15 | Executive summary |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

## Data

`Netflix_Dataset.csv` — 7,789 titles with fields: `Show_Id, Category, Title, Director, Cast, Country, Release_Date, Rating, Duration, Type (genres), Description`.

## Author

Nayana S — Data Analysis Intern, Vodafone Idea Foundation (VOIS) × Edunet Foundation, Sep–Oct 2025.
