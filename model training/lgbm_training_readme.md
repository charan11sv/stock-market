# Multi-Horizon Stock Forecasting with LightGBM (10-Day Ahead)

> **TL;DR**: This notebook forecasts the next **10 days of closing prices** for many stocks using a **multi-output LightGBM** model.  
> I initially explored a **Transformers + LightGBM hybrid**, but due to the scale of data and limited compute resources, I focused on LightGBM with optimized preprocessing, experiments on small subsets, scaling to the full dataset, hyperparameter tuning (manual + Optuna), **KFold cross-validation**, MAPE-based evaluation, and detailed plotting.  
> The final trained model is saved as `multi_model_stock1.pkl`.

---

##  Objective

- Build a robust and efficient pipeline to **predict the next 10 daily closing prices** (`close_1` … `close_10`) for each stock.  
- Handle a **large multi-company dataset** across many years.  
- Balance **accuracy vs resource usage** (limited CPU/GPU, memory).  
- Iteratively move from **experiments on subsets** → **full-scale training** → **KFold confirmation**.  
- Evaluate thoroughly with **MAPE, cross-validation, and multiple LightGBM configurations**.  
- Document all attempted approaches, including **discarded ideas**, to highlight breadth of exploration.

---

##  Data & Setup

- Sources used:
  - `/kaggle/input/stock-market-1/final_cleaned.parquet`
  - `/kaggle/input/stock-final-train-test/stock_df_train.parquet`
  - `/kaggle/input/stock-final-train-test/stock_df_test.parquet`
- Features included technical indicators, rolling stats, and market-level signals.  
- **Polars** chosen for:
  - Faster parquet reading
  - Company-wise grouping & processing
  - Efficient concatenation of splits
- Targets generated per company:
  - `close_1` → next day closing price  
  - … up to  
  - `close_10` → 10 days ahead closing price  
- Splits:
  - **Train (80%)**
  - **Test (20%)**
- Test further divided into validation during tuning.

---

##  Environment & Key Libraries

- **Core**: `polars`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`  
- **Hyperparameter Tuning**: `optuna`  
- **Evaluation**: custom MAPE function (NumPy & CuPy variants)  
- **Plots**: `matplotlib`  
- **Persistence**: `joblib`  

### Test Set Results 

The following are the average percentage errors (MAPE) for each prediction horizon:

| Horizon  | MAPE  |
|----------|-------|
| close_1  | 5.87% |
| close_2  | 8.82% |
| close_3  | 10.75% |
| close_4  | 10.84% |
| close_5  | 12.63% |
| close_6  | 13.83% |
| close_7  | 14.35% |
| close_8  | 14.68% |
| close_9  | 15.57% |
| close_10 | 17.37% |

**Key Observations:**
- **Short-horizon forecasts (1–2 days)** are fairly accurate (~6–9% MAPE).  
- **Errors increase steadily with horizon length**, reaching ~17% by day 10 (expected in financial time series).  
- Confirms the model generalizes well on unseen test data and captures short-term price dynamics effectively.



Install (example):
```bash
pip install polars pandas numpy lightgbm scikit-learn optuna matplotlib joblib
