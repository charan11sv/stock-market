#  Indian Stock Market & News Feature Factory — Modeling-Ready Parquet with 80+ Engineered Signals

> Build a modeling-ready market dataset by combining **equity price data** (NSE/BSE) with **news-driven sentiment** and engineered features.

This repo contains three modular notebooks plus concise docs that together form a reproducible pipeline:
1) **Equity data extraction** → 2) **News sentiment building** → 3) **Final dataset creation & feature engineering**.

---

## TL;DR (for interviewers)

- **What this demonstrates**
  - Practical **ETL** for market data, **NLP** for daily sentiment, and **feature engineering** for ML.
  - Robustness: **failure logs**, **intermediate saves**, and **idempotent** steps.
  - Interpretability: a simple **linear model** to learn weights for a **composite sentiment feature**.
- **What you get**
  - A **final parquet** with technical indicators, market & macro series, lagged sentiment, and a weighted sentiment metric.
  - A **feature glossary** explaining every column.

---

## Repo Contents

| File | Purpose |
|---|---|
| `equity data extraction.ipynb` | Extract NSE/BSE tickers and fetch historical OHLCV data; logs failures. |
| `sentiment_scores.ipynb` | Fetch financial news (RSS/GDELT), score with VADER, aggregate per company/day. |
| `Dataset creation main.ipynb` | Merge, clean, create lags, learn feature weights, and output the final dataset. |
| `equity data extraction documentation.md` | Documentation for the equity notebook. |
| `sentiment scores creation documentation.md` | Documentation for the sentiment notebook. |
| `main dataset creation documention.md` | Documentation for the final dataset notebook. |
| `final features.md` | One-line description of **every** final dataset column. |

> Tip: Docs are short and skimmable. Start with **`main dataset creation documention.md`** and **`final features.md`**.

---

## Pipeline Overview

```mermaid
flowchart LR
  A[BhavCopy CSVs + NSE_tickers.csv + bse_tickers.csv] --> B[equity data extraction.ipynb]
  B -->|OHLCV per ticker + failure logs| C[sentiment_scores.ipynb]
  C -->|daily sentiment per company| D[Dataset creation main.ipynb]
  D -->|feature weights + final parquet| E[[merged_with_weighted_sentiment_final.parquet]]
  D --> F[(feature_weights_final.csv)]
