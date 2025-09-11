# Multi-Horizon Stock Forecasting with LightGBM (10-Day Ahead)

> **TL;DR**: This notebook forecasts the next **10 days of closing prices** for many stocks using a **multi-output LightGBM** model.  
> I initially explored a **Transformers + LightGBM hybrid**, but due to the scale of data and limited compute resources, I focused on LightGBM with optimized preprocessing, experiments on small subsets, scaling to the full dataset, hyperparameter tuning (manual + Optuna), **KFold cross-validation**, MAPE-based evaluation, and detailed plotting.  
> The final trained model is saved as `multi_model_stock1.pkl`.

---

## 🎯 Objective

- Build a robust and efficient pipeline to **predict the next 10 daily closing prices** (`close_1` … `close_10`) for each stock.  
- Handle a **large multi-company dataset** across many years.  
- Balance **accuracy vs resource usage** (limited CPU/GPU, memory).  
- Iteratively move from **experiments on subsets** → **full-scale training** → **KFold confirmation**.  
- Evaluate thoroughly with **MAPE, cross-validation, and multiple LightGBM configurations**.  
- Document all attempted approaches, including **discarded ideas**, to highlight breadth of exploration.

---

## 📦 Data & Setup

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

## 🧰 Environment & Key Libraries

- **Core**: `polars`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`  
- **Hyperparameter Tuning**: `optuna`  
- **Evaluation**: custom MAPE function (NumPy & CuPy variants)  
- **Plots**: `matplotlib`  
- **Persistence**: `joblib`  

Install (example):
```bash
pip install polars pandas numpy lightgbm scikit-learn optuna matplotlib joblib
