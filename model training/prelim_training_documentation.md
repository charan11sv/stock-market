# Stock Prediction Experiments — Notebook Retrospective (README)

> **TL;DR**: I set out to test a hybrid approach for stock prediction — **LightGBM** for feature-wise/tabular patterns, **Transformers** for row/temporal patterns, and eventually an **ensemble** of both. Due to **resource constraints** on my local machine, I iterated in this early notebook with scaled-down experiments, tested **sampling strategies**, implemented a semi-automated training pipeline, and then **moved to Kaggle** to continue at scale. Along the way I documented what worked, what didn’t, and the key lessons that shaped my next steps.

---

## Objectives
1. **Compare modeling families**: gradient boosting (**LightGBM**) vs. sequence models (**Transformer**) for stock prediction.
2. **Prototype an ensemble** that combines feature-wise signals (trees) and temporal signals (attention).
3. **Work within limited resources** by running smaller experiments, then **scale up elsewhere (Kaggle)** once validated.
4. **Build reusable training utilities** (preprocessing, feature selection, save/load, basic orchestration).

## Dataset (as used here)
- Primary file: `final_cleaned.parquet`
- Structure: tabular time series per company (contains `Company_ID`), with engineered technical indicators and basic market features.
- In-notebook EDA: listed columns, checked null counts, isolated numeric features, basic cleaning of infinities → NaN → mean imputation.

## Environment & Constraints
- Local machine with limited RAM and CPU/GPU, which impacted:
  - **Model size & batch size**
  - Ability to run **LightGBM** and **Transformer** at scale simultaneously
  - Feasibility of **sequence windows** for Transformers (temporal context is costly)
- I intentionally did not switch to **float32** everywhere in this early notebook (partly due to unawareness at the time), which increased memory pressure.
- After initial iterations and learning, I **moved to a Kaggle notebook** to continue with more resources.

---

## What I Attempted

### 1) Feature Engineering & Selection
- **Cleaning & Imputation**: replace `inf/-inf` → `NaN`, then **mean imputation**.
- **Scaling**: `StandardScaler` for numeric features.
- **Feature Selection**: **Recursive Feature Elimination (RFE)** using `LinearRegression` (select `num_features` top features).

### 2) Models
- **LightGBM (tabular)**: intended to capture **feature-wise, non-linear interactions** efficiently on engineered indicators and lags.
- **Transformer (sequence)**: custom `TransformerEncoder`-based regressor intended to capture **row-to-row temporal dependencies** via self-attention.
  - Early version in this notebook effectively treated **each row as a sequence of length 1** (i.e., intra-row feature interactions), which I later identified as a limitation (see Learnings).

### 3) Training Approaches
- **Company-wise mini-batching**: process **two companies at a time** (subset-based training) to stay within memory limits.
- **Interactive orchestration**: prompts to choose whether to run LightGBM and/or Transformer, save/load models, and continue across batches.
- **Parallelization attempt**: wrapper with `ThreadPoolExecutor` for LightGBM to test parallel runs of param sets (more of a skeleton than fully utilized here).
- **Model persistence**: save **LightGBM** as `.pkl` and **Transformer** as `.pth` for reuse.

### 4) Sampling Experiments (due to constraints)
- When full-data training wasn’t feasible locally, I tried **sampling strategies** to:
  - Train on **subsets** of rows/companies.
  - **Extrapolate** lessons to the full dataset later (primarily for methodology learning and pipeline validation).

### 5) Planned but Deferred
- **True sequence modeling** with Transformer using **windowed inputs** (e.g., 10–30 prior rows) + **positional encoding**.
- **Ensembling** of LightGBM and Transformer predictions (weighted/stacked, with validation-driven blending).
- **Cross-company representation** (e.g., using `Company_ID` embeddings) instead of dropping the column.

---

## What I Actually Did in the Notebook 

This notebook was an early experimental sandbox where I explored different approaches for stock price prediction.  
Here’s everything I attempted, step by step:

---

### 1) Dataset Loading & Exploration
- Loaded the dataset: **`final_cleaned.parquet`**
- Inspected:
  - Column names
  - Null value counts
- Focused on **numeric features** for preprocessing.

---

### 2) Feature Engineering & Selection
- **Cleaning**: replaced `inf/-inf` with `NaN`, filled missing values with column means.
- **Scaling**: applied `StandardScaler` to normalize features.
- **Variance Inflation Factor (VIF)**: checked for multicollinearity.
- **Principal Component Analysis (PCA)**: tested dimensionality reduction.
- **Recursive Feature Elimination (RFE)**: used `LinearRegression` to select top `num_features`.

---

### 3) Models Attempted
- **LightGBM (LGBMRegressor)**
  - Tree-based boosting model, trained on reduced/scaled features.
- **Transformer**
  - Custom PyTorch `nn.TransformerEncoder` model.
  - Trained as a regressor with MSE loss and Adam optimizer.
  - Early design only saw each row as a **sequence of length 1** (captured feature relationships but not row-to-row temporal dependencies).
  - Later attempted **mini-batch training** with `DataLoader`.
- **Ensemble**
  - Combined predictions from LightGBM and Transformer to compare RMSEs.

---

### 4) Training Approaches
- **Company-wise training**
  - Processed **two companies at a time** to fit within memory limits.
- **Interactive orchestration**
  - Notebook prompted whether to:
    - Train LightGBM
    - Train Transformer
    - Save models
    - Continue training or stop
- **Parallelization**
  - Added a `ThreadPoolExecutor` wrapper to test **parallel LightGBM runs**.
- **Model Persistence**
  - Saved models:
    - LightGBM → `.pkl`
    - Transformer → `.pth`
  - Reloaded models for continued training.

---

### 5) Experiments & Results
- Ran multiple Transformer training configurations:
  - **200 epochs @ 0.01 LR**
  - **3000 epochs @ 0.001 LR**
  - **20000 epochs @ 0.001 LR**  
- Documented **RMSEs**:
  - LightGBM
  - Transformer
  - Ensemble (best RMSEs often close to LightGBM baseline).
- **Hyperparameter tuning for LightGBM**
  - Used **KFold cross-validation**.
  - Searched over `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `lambda_l1`, `lambda_l2`, etc.
  - Logged **best trials** and RMSEs.
- Compared performance on **different subsets of companies**:
  - 10, 50, 100, 500 companies.
- Noted cases where **training failed** due to local resource constraints.

---

### 6) Sampling Strategies
Because full dataset training was too heavy locally:
- Trained on **smaller subsets of companies**.
- Used **reduced estimators**.
- Treated this as a **learning exercise** and validated pipelines before scaling to Kaggle.

---

### 7) Persistence
- LightGBM models saved as `.pkl`.
- Transformer models saved as `.pth`.
- Reloading functionality implemented for both.

---

###  Summary
In this notebook I:
- Explored **feature selection methods** (RFE, PCA, VIF).
- Trained **LightGBM** (tabular patterns) and **Transformer** (feature interactions).
- Built a simple **ensemble** of both.
- Ran **hyperparameter tuning** with KFold and parameter sweeps.
- Tested **sampling and scaling strategies** under resource limits.
- Implemented **interactive training loops, model saving/loading, and parallelization skeletons**.
- Collected and logged **RMSE results** across experiments.

> Even though this notebook had limitations (seq_len=1 Transformer, no full dataset runs locally), it laid the groundwork for moving to **Kaggle** for larger-scale experiments, while teaching me important lessons about **resource-aware design, preprocessing, and model orchestration**.


---

## What Didn’t Work / Was Limited (and Why)

- **Full LightGBM at scale (locally)**: ran into **memory/time constraints**.
- **Transformer as true sequence model**: not realized here due to **sequence windowing cost** and early-stage design (seq_len=1).
- **Ensemble (LGBM + Transformer)**: deferred; needed stable individual models first and more resources.
- **Cross-company generalization**: dropped `Company_ID` during training; no embeddings yet → missed inter-company relationships.

---

## Key Learnings

1. **Resource-aware design matters**: batch by company, reduce features (RFE), and keep datatypes lean (float32) — these are not “nice-to-haves”; they’re essential.
2. **Transformer ≠ automatic temporal modeling**: it only captures **temporal dependencies** if you **feed sequences** and **encode positions**. Otherwise, it behaves closer to a feed-forward model on each row.
3. **Sampling is a valid learning tool**: even if samples don’t give final metrics, they validate **pipelines, APIs, and save/load flow** before scaling up.
4. **LightGBM vs. Transformers**: for tabular features, **LightGBM is strong and efficient**; Transformers shine when **true sequences** are modeled and resources allow.
5. **Plan the ensemble last**: first stabilize single-model pipelines and evaluation; then blend with **validation-driven weights** or a meta-learner.
6. **Instrumentation saves time**: interactive prompts and persistence made it easier to iterate safely in low-resource environments.
7. **Move platforms when needed**: switching to **Kaggle** was the right call to keep progress while respecting local limits.

---

## Results & Achievements (from this early notebook)
- Built a **working training pipeline** with preprocessing, RFE-based feature selection, and two modeling tracks (LightGBM + Transformer).
- Implemented **interactive orchestration** to choose models, save artifacts, and iterate over company subsets.
- Created a **parallelization skeleton** for LightGBM experiments.
- Performed **sanity checks** (columns, nulls) and established a pattern for **cleaning/scaling/imputation**.
- Identified **design gaps** (true sequence modeling, ensembling) and clarified a path forward.
- Used the constraints to drive **learning and design decisions** — then **migrated to Kaggle** for scale-up.

---

## Next Steps (Actionable)

1. **True Sequence Inputs for Transformer**
   - Build sliding windows (e.g., 20–60 timesteps), add **positional encodings**, and predict t+1 (or multi-horizon).
   - Use **float32** tensors and **gradient accumulation** if memory is tight.

2. **Company Awareness**
   - Add **learned embeddings** for `Company_ID` (+ optional sector/industry embeddings) for both Transformer and LightGBM (via target encoding / one-hot if feasible).

3. **Robust Validation**
   - **Time-series CV** (rolling/blocked) to avoid leakage; compare LightGBM/Transformer fairly.

4. **Ensemble**
   - Start with **weighted average** based on validation RMSE/MAE, then move to a **stacked meta-learner** (e.g., ridge/GBM on out-of-fold predictions).

5. **Efficiency**
   - Use **float32** end-to-end; consider **smaller hidden sizes** and **fewer heads/layers**; profile seq_len impact.
   - For LightGBM, use **categorical handling** (if any) and **histogram-based training** with tuned `num_leaves`, `feature_fraction`, `bagging_fraction`.

6. **Scale-Up on Kaggle**
   - Port the improved pipeline to Kaggle, persist artifacts, and track experiments with clear configs and logs.

---

## Repro Notes (for this notebook’s flow)
1. Place `final_cleaned.parquet` in working directory.
2. Run the preprocessing & RFE cells.
3. Use the **company-wise loop** to select two companies; choose whether to train **LightGBM**, **Transformer**, and whether to **save** models.
4. (Optional) Try the LightGBM parallel wrapper with small param variants.
5. Inspect logs for RMSE/loss; iterate or move to Kaggle for larger runs.

---

## Appendix — Configuration Highlights (as used here)
- **Feature selection**: RFE (`LinearRegression`), `num_features` configurable.
- **Scaling**: `StandardScaler` on numeric columns; mean-impute NaNs.
- **Transformer**: `TransformerEncoder`-based regressor; trained with MSE + Adam; inputs were `(batch, seq_len=1, num_features)` in this notebook.
- **LightGBM**: parameters provided interactively; model saved as `.pkl` when chosen.
- **Persistence**: Transformer saved as `.pth` and re-loadable.
- **Orchestration**: interactive prompts for per-batch choices; two companies processed per iteration.

---

