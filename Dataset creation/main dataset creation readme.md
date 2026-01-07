# Dataset Creation & Feature Engineering (Notebook: `Dataset creation main.ipynb`)

## Overview
This notebook prepares and merges datasets for downstream modeling by adding **external features** (e.g., sentiment signals, market indicators) and creating a **weighted sentiment metric** learned from data. It also performs data hygiene steps and exports both the learned feature weights and the finalized enriched dataset.

---

## Goals
- Load the pre-merged base dataset that already includes sentiment and market features.
- Engineer **lagged sentiment** features to capture short/medium-term effects.
- Curate/ensure availability of key **sentiment-related predictors** (e.g., `Price_Change`, `Momentum`, `OBV_Change`, `Lagged_Sentiment_*`).
- Clean and validate data (handle missing/inf/NaN).
- Fit a **linear regression** model to estimate **feature weights**.
- Build a **Weighted Sentiment Metric** using the learned coefficients.
- Save the artifacts for later modeling.

---

## Inputs
- **`merged_with_sentiment_interim1.parquet`**  
  Base dataset containing per-company time series with sentiment and market signals (one row per date/company).

> **Expected minimum columns**  
> - Identifier/time: `Company_ID`, `Date`  
> - Sentiment: `sentiment_score` and lag fields to be created (`Lagged_Sentiment_1`, `Lagged_Sentiment_5`, `Lagged_Sentiment_10`)  
> - Market/derived signals (at least a subset of): `Price_Change`, `Momentum`, `OBV_Change`  
> - (Plus any other features already computed upstream)

---

## Outputs
- **`feature_weights_final.csv`** — learned coefficients/weights from the linear model for the selected predictors.  
- **`merged_with_weighted_sentiment_final.parquet`** — enriched dataset with the new `Weighted_Sentiment_Metric` column and cleaned rows.

---

## Processing Steps (What the notebook does)

1. **Load Base Data**
   - Read `merged_with_sentiment_interim1.parquet` into a dataframe.
   - Normalize/standardize column names if needed and ensure types (`Date` as datetime; `Company_ID` as categorical/string).

2. **Create Lagged Sentiment Features**
   - For each `Company_ID`, create:
     - `Lagged_Sentiment_1` (1-day shift of `sentiment_score`)
     - `Lagged_Sentiment_5` (5-day shift)
     - `Lagged_Sentiment_10` (10-day shift)
   - These capture short-term and medium-term sentiment effects on subsequent returns/prices.

3. **Ensure/Compute Sentiment-Related Predictors**
   - Confirm presence (or compute if missing) of:
     - `Price_Change` (e.g., current close − previous close)
     - `Momentum` (e.g., rolling sum/return proxy or price velocity)
     - `OBV_Change` (delta of On-Balance Volume or a similar liquidity/flow signal)
     - `Lagged_Sentiment_1`, `Lagged_Sentiment_5`, `Lagged_Sentiment_10`
   - Optionally clip extreme values to mitigate outliers if present upstream.

4. **Data Hygiene**
   - Drop rows with **missing** or **infinite** values across the selected predictor set and the target used for weight estimation.
   - (Optional) Filter unrealistic values if thresholds are defined.

5. **Estimate Feature Weights (Linear Regression)**
   - Using `scikit-learn`’s `LinearRegression` (OLS) on a chosen target `y` (e.g., a future return/price change or other modeling target), fit:
     - **X** = `[Price_Change, Momentum, OBV_Change, Lagged_Sentiment_1, Lagged_Sentiment_5, Lagged_Sentiment_10]`
     - **y** = (your selected target column; the notebook uses a consistent target across all rows after cleaning)
   - Extract coefficients and export them as **`feature_weights_final.csv`**.
   - Basic sanity checks (e.g., non-zero variance, number of samples) are performed implicitly via fit success.

6. **Build Weighted Sentiment Metric**
   - Compute:
     ```
     Weighted_Sentiment_Metric
       = w1*Price_Change
       + w2*Momentum
       + w3*OBV_Change
       + w4*Lagged_Sentiment_1
       + w5*Lagged_Sentiment_5
       + w6*Lagged_Sentiment_10
     ```
     where `w1..w6` are the learned coefficients.
   - Append `Weighted_Sentiment_Metric` to the dataframe.

7. **Export Final Artifacts**
   - Save the updated dataframe with the new metric as **`merged_with_weighted_sentiment_final.parquet`**.
   - Save the coefficient table as **`feature_weights_final.csv`**.
   - Print brief completion stats (row counts, columns created).

---

## Configuration & Assumptions
- **Grouping key:** `Company_ID`; lags are computed **within** each company’s time series.
- **Date order:** The dataset must be **sorted by `Company_ID`, then `Date` ascending** before lagging.
- **Target selection:** A consistent target column is used for regression (set inside the notebook).  
- **Feature scaling:** Not strictly required for OLS interpretability; if applied, the same scaling is **not** exported (weights reflect whatever scaling was in effect during fit).
- **Data coverage:** Rows with insufficient history for lags (e.g., first few days) may be dropped during cleaning.

---

## How the Weighted Metric Is Typically Used
- As a **single composite feature** feeding downstream models (tree-based, DL, etc.).
- For **signal monitoring** and quick sanity checks against market moves.
- For **feature importance** intuition: the sign/magnitude of weights provides interpretability.

---

## Reproducibility
- Set a fixed random seed if any stochastic steps are added later.
- Record the package versions (e.g., `pandas`, `scikit-learn`) in your environment file or project README.
- Keep the exact input filename stable: `merged_with_sentiment_interim1.parquet`.

---

## Example Snippets (for clarity)

**Creating lags (conceptual):**
```python
df = df.sort_values(["Company_ID", "Date"])
for k in [1, 5, 10]:
    df[f"Lagged_Sentiment_{k}"] = df.groupby("Company_ID")["sentiment_score"].shift(k)
