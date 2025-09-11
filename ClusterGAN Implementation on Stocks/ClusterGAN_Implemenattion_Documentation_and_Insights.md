# ClusterGAN for Regime Discovery on Indian Stocks (NSE)

> **TL;DR**  
> I adapted **ClusterGAN** to **unlabeled NSE stock data** to discover market regimes directly from high-dimensional features (OHLCV, indicators, sentiment, macro, global indices, FX/commodities). I trained **G/D/E** with **WGAN-GP** and **latent consistency**, swept **latent std & clipping**, explored **cluster counts K ∈ [2, 100]**, and compared against **K-Means**.  
> The biggest takeaways from the final insights:
> - With **K=7**, **4 clusters** were **highly explainable** from available features; **3 clusters** were **not** (these “volatile/mysterious” clusters were ~**10%** of data).  
> - **Moderate noise** (σ≈**0.10–0.15**) and **tighter clipping** (≈**[−0.6, 0.6]**) gave **stable training** and **better separation**.  
> - Example metrics for **K=7** (one sweep run): **Sil ≈ 0.00**, **CH ≈ 34,680.81**, **DB ≈ 2.95**, **Inertia ≈ 2,798,406**, **Compactness ≈ 2.94** (Silhouette sampled for compute).  
> - The “hard-to-explain” clusters likely reflect **exogenous drivers** (macro/news/idiosyncratic shocks) not captured by the current feature set.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Paper Used (ClusterGAN)](#paper-used-clustergan)
  - [Paper at a Glance](#paper-at-a-glance)
- [Data: Sources, Scope & Features](#data-sources-scope--features)
  - [Full Feature Inventory](#full-feature-inventory)
  - [Common Pitfalls & Fixes](#common-pitfalls--fixes)
- [Preprocessing & Splits](#preprocessing--splits)
- [Method: ClusterGAN Recap & My Implementation](#method-clustergan-recap--my-implementation)
  - [Objective & Losses](#objective--losses)
  - [Architectures](#architectures)
  - [Training Configuration](#training-configuration)
- [Experiments I Ran](#experiments-i-ran)
  - [Baselines](#baselines)
  - [ClusterGAN Sweeps](#clustergan-sweeps)
  - [Outputs & Artifacts](#outputs--artifacts)
- [Results & Final Insights (from the notebook)](#results--final-insights-from-the-notebook)
  - [What Worked](#what-worked)
  - [What Didn’t / Limitations](#what-didnt--limitations)
  - [Why These Results Make Sense](#why-these-results-make-sense)
- [What I Did — Step by Step](#what-i-did--step-by-step)
- [Future Plans (as in the notebook)](#future-plans-as-in-the-notebook)


---

## Why This Project

Most ClusterGAN examples are on labeled/“nice” benchmarks (MNIST etc.). **Markets aren’t like that**: unlabeled, non-stationary, fat-tailed, with regime shifts and heterogeneous liquidity. I wanted to test whether **ClusterGAN** can:
- Discover **interpretable market regimes** from **unlabeled** NSE data; and
- Reveal **latent regimes** that our current features **can’t** easily explain (useful for risk/alerts).

---

## Paper Used (ClusterGAN)

**ClusterGAN: Latent Space Clustering in Generative Adversarial Networks**  
*Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, Sreeram Kannan (2019)*

### Paper at a Glance
- Use a **mixed latent**: **discrete** `z_c` (one-hot, denotes cluster) + **continuous** `z_n` (Gaussian).
- Add an **encoder** `E(x)` so `E(G(z)) ≈ z` (MSE on `z_n`, CE on `z_c`).
- Train the GAN (here **WGAN-GP**) **plus** latent consistency so the **discrete code maps to semantic clusters**.
- Keep **continuous noise small** (σ≈0.1) and **bounded** to preserve **cluster separation**.

---

## Data: Sources, Scope & Features

- **Universe**: NSE-listed companies across multiple years and sectors.  
- **Scale**: ~**518,000 rows** used in this experiment (from the consolidated dataset).  
- **Width**: ~**92 features** (engineered by me) spanning price, volume/liquidity, technicals, momentum/volatility, sentiment, market/global context, FX/commodities, and macro proxies.

### Full Feature Inventory

**Core OHLCV & IDs**
- `Date`, `Open_x`, `High_x`, `Low_x`, `Close_x`, `Adj Close_x`, `Volume_x`, `Company_ID`, `COMPANY`, `Row_ID`

**Derived price/volume & ratios**
- `VWAP_x`, `Price_Change`, `Price_Change_Percent`, `Volume_Change`, `Volume_Change_Percent`,  
  `High_Low_Spread`, `Close_Open_Ratio`, `High_Low_Ratio`, `Price_Range`, `Price_Volume_Product`

**Trend/MAs/Bands**
- `SMA_20`, `EMA_20`, `WMA_20`, `Upper_Band`, `Middle_Band`, `Lower_Band`

**Momentum & Oscillators**
- `MACD_x`, `MACD_Signal_x`, `MACD_Hist_x`, `RSI_14`, `Stoch_K`, `Stoch_D`, `CCI_20`, `ROC`, `ROC_5`, `ROC_10`

**Volatility & Stats**
- `ATR_14`, `SAR`, `Rolling_Std_Close`, `Rolling_Std_Volume`,  
  `Rolling_Mean_Close`, `Rolling_Max_Close`, `Rolling_Min_Close`, `Rolling_Price_Range`, `Z_Score_Close`,  
  `Kurtosis`, `Skewness`, `Sharpe_Ratio`, `Pivot_Point`

**Dynamics & Returns**
- `Momentum`, `Price_Acceleration`, `OBV`, `OBV_Change`, `Volume_Ratio`, `Rolling_Return`

**Time Features**
- `Day_of_Week`, `Month`, `Quarter`, `Week_of_Year`

**Lags**
- `Lag_Close_x_1/5/10`, `Lag_Volume_x_1/5/10`

**Sentiment / Text**
- `sentiment_score`, `Lagged_Sentiment_1/5/10`, `Weighted_Sentiment_Metric`

**Market & Global Context**
- `Market_Close`, `Market_Returns`, `Global_Close`, `Global_Returns`,  
  `SP500`, `Nifty50`, `FTSE100`, `DAX`, `Nikkei225`,  
  `Rolling_Beta`, `Rolling_Market_Correlation`, `Rolling_Global_Correlation`

**FX / Commodities / Risk**
- `INR_USD_Close`, `USD_INR`, `Crude_Oil`, `Crude_Oil_Close`, `Gold`, `Gold_Close`, `VIX`

### Common Pitfalls & Fixes
- **±Inf/NaN** from ratios → map to NaN → `fillna(0)` **after scaling**.  
- **Near-constant columns** (illiquidity) → drop/down-weight.  
- **Heavy tails** → rely on MinMax to [−1,1] + WGAN-GP robustness (avoid over-smoothing).  
- **Mixed sampling across exogenous series** → forward-fill & align by date (no look-ahead).  
- Avoided quantile/distribution transforms (hurt GAN stability & tail structure).

---

## Preprocessing & Splits

- **Company-aware split**: shuffle **per company**, then **train/test** (I used both 20/80 and 90/10 variants in different experiments; silhouette sampled to control cost).  
- **Scaling**: **MinMaxScaler to [−1,1]** per feature (pairs well with generator `tanh`).  
- Model inputs are **numeric columns only** (keep `Row_ID` only for joining results back).

---

## Method: ClusterGAN Recap & My Implementation

### Objective & Losses
- **Adversarial**: **WGAN-GP** (gradient penalty λ=10).  
- **Latent**: `z = [z_n (Gaussian), z_c (one-hot K)]`.  
- **Encoder consistency**:  
  - **MSE** on `z_n` (weight 10),  
  - **Cross-entropy** on `z_c` (weight 10).  

### Architectures (MLPs, stable for tabular)
- **Generator G**: `z_dim → 256 → 512 → feature_dim`, LeakyReLU(0.2), BN, final **tanh**.  
- **Discriminator D**: `feature_dim → 512 → 256 → 1`, LeakyReLU(0.2).  
- **Encoder E**: `feature_dim → 512 → 256 → [z_n, z_c]` (softmax head for `z_c`).  

### Training Configuration
- **Epochs**: **8** (short sweeps; plan longer final runs).  
- **Batch size**: up to **10,000** (capped by data).  
- **Optimizers**: **Adam(lr=5e-5, betas=(0.5, 0.9))** for G/D/E.  
- **Latent setup**: defaults like **`zn_dim=85`**, **`zc_dim=7`** in the main sweep; also tested **K=3** and **K=100**.  
- **Sweeps**:
  - **Noise std σ** ∈ {0.05, 0.08, **0.10**, **0.12**, **0.15**, 0.20, 0.25}  
  - **Clipping ranges** on `z_n`: {**[−0.6, 0.6]**, [−0.7, 0.7], [−0.8, 0.8]}  
- **Metric sample size**: **45,000** for Silhouette (downsampled, stratified) to reduce compute.

---

## Experiments I Ran

### Baselines
- **K-Means** on scaled features with same K as `zc_dim`.  
- Metrics: **Silhouette**, **Calinski–Harabasz (CH)**, **Davies–Bouldin (DB)**, **Inertia**.  
- Optional alignment diagnostics (K-Means vs GAN clusters on the same samples): **NMI/ARI** via Hungarian matching (just relative checks—no labels).

### ClusterGAN Sweeps
- Fixed **K=7** for std/clipping sweep to stabilize the search; separate runs for **K=3** and **K=100** as extremes.  
- For each config, I:
  1. Trained ClusterGAN.  
  2. Encoded test samples, assigned clusters via `argmax(E(x).z_c)`.  
  3. Computed **Silhouette/CH/DB/Compactness** (Silhouette on the 45k sample) and logged **Inertia** baseline.  
  4. Exported cluster assignments and metrics.

### Outputs & Artifacts
- **`metrics_df_testing_deviations_set_1.csv`** – per-config metrics (σ, clip, Sil/CH/DB/Inertia/Compactness, etc.).  
- **`clustering_data_1.csv`** – test set with predicted **cluster** id.  
- **`original_with_clusters.parquet`** – merge via `Row_ID` (joins done with **Polars** for speed).

---

## Results & Final Insights (from the notebook)

> This section mirrors the “Final Insights” cell I wrote in the notebook, expanded where needed for clarity.

1) **Cluster counts, search strategy, and expectation**  
- I evaluated **K from 2 to 100** (broad sweep) even though, practically, I expected **≈30** clusters to cover meaningful regimes at “full-market” scale.  
- Rather than picking K with only elbow methods on **K-Means**, I **trained ClusterGAN** at many K values to see which **latent structure** actually stabilized and yielded **good separation** and **decodable clusters**.

2) **Explainability test of clusters**  
- I trained **post-hoc classifiers** (e.g., Random Forest / LightGBM) to predict the **GAN cluster id** from **original features**:  
  - With **K=7**, **4 clusters** were **highly predictable** (strong classification accuracy).  
  - The other **3 clusters** were **hard to predict** from the features.  
- Interpretation: For these 3 clusters, **available features were insufficient**; yet the clusters were **not random** (see next point).

3) **“Mysterious/volatile” cluster behavior**  
- The **hard-to-explain** clusters were **systematic**, not noise:  
  - Characterized by **large moves** / regime shifts that **standard technical/time features** couldn’t explain.  
  - These clusters accounted for **~10%** of the dataset in the representative **K=7** run.  
  - Likely drivers: **macro news**, **idiosyncratic events**, or **other exogenous factors** not in the current feature set.

4) **Latent space knobs that mattered**  
- **Noise std σ** and **clipping** had **clear effects**:  
  - **Moderate σ (≈0.10–0.15)** struck the best balance (too small under-explores; too large destabilizes).  
  - **Tighter clipping** (≈**[−0.6, 0.6]**) improved **early-epoch stability** and **slightly better** separation across metrics.  
- **Example K=7 metrics** (one sweep config):  
  - **Sil ≈ −0.0000**, **CH ≈ 34,680.81**, **DB ≈ 2.9527**, **Inertia ≈ 2,798,406.00**, **Compactness ≈ 2.9432**.  
  - Silhouette was computed on a **sample of 45k** test points to keep compute bounded.

5) **Feature signatures across clusters**  
- Even without pinning “names” to clusters, **feature group patterns** were **consistent** within many clusters:  
  - Momentum/oscillator ranges, band positions, and volatility stats tended to **co-move** within a cluster.  
  - Correlation/beta metrics (market/global) often **aligned** inside a cluster.  
- For the **unpredictable** clusters, these signatures **didn’t suffice** to explain assignments—again suggesting **missing exogenous info**.

6) **K extremes**  
- **K=3** gave **coarse regimes** (sensible, but too broad for tactics).  
- **K=100** trained but tended to **fragment** unless trained **longer** / regularized more; useful as a stress-test or for **very fine-grained** tasks.

### What Worked
- Using **G/D/E + latent consistency** with **WGAN-GP** on **MinMax [−1,1]** features.  
- **Moderate σ** and **tight clipping** on `z_n`.  
- **Company-aware split** and **Silhouette sampling** (45k) to keep evaluation feasible.  
- **Post-hoc explainability** check (tree/LGBM) to separate **explainable vs latent** clusters.

### What Didn’t / Limitations
- **Short epochs (8)** in sweeps limit final separation; longer training should help.  
- **Simple MLPs** are stable but less expressive than deeper nets/temporal modules.  
- **Feature gaps** for exogenous drivers (macro/news/events) limit interpretability for that ~**10%** volatile slice.  
- K-Means is a good sanity check but **lacks generative structure** and **latent semantics**.

### Why These Results Make Sense
- ClusterGAN’s **discrete code** neatly binds regimes; **continuous code** captures intra-cluster variations.  
- Markets have **regime-like** behavior but are also **news-driven**; it’s expected that some clusters are predictable with TA features while others need **macro/event** context.

---

## What I Did — Step by Step

1. **Data prep**  
   - Load consolidated NSE dataset (~**518k rows**, ~**92 features**).  
   - Clean infinities/NaNs; drop low-variance columns.  
   - Company-aware **train/test** split (used 20/80 and 90/10 variants).  
   - **MinMaxScaler** to **[−1,1]** per feature.

2. **Modeling**  
   - Implement **G, D, E** (MLPs), **WGAN-GP**, and **latent consistency** (`MSE(z_n̂,z_n)` + `CE(z_ĉ,z_c)`).  
   - Configure latent: `zn_dim≈85`, sweep **σ** and **clip**; test **K ∈ {3,7,100}**.

3. **Training**  
   - **Epochs=8**, **batch up to 10k**, **Adam(5e-5, betas=0.5,0.9)**.  
   - For each (σ, clip, K): train, encode test, assign `cluster = argmax(E(x).z_c)`.

4. **Evaluation**  
   - Compute **Silhouette** (sample **45k**, stratified), **CH**, **DB**, **Compactness**; log **K-Means Inertia**.  
   - Save to **`metrics_df_testing_deviations_set_1.csv`**.  
   - Export assignments to **`clustering_data_1.csv`** and merge with original via **`Row_ID`** → **`original_with_clusters.parquet`**.

5. **Explainability check**  
   - Train **RF/LGBM** to predict GAN cluster from original features → identify which clusters are **feature-explainable** vs **latent**.

6. **Synthesis of insights**  
   - Compare across σ/clip/K; note stability, separation, and interpretability; document **4 explainable vs 3 latent** clusters and **~10%** volatile slice.

---

## Future Plans 

1. **Strengthen `z_n`**  
   - Adjust loss weights / heads so **continuous latent** captures richer intra-cluster variation (avoid over-dominant `z_c`).

2. **Systematic K sweep**  
   - Full **K ∈ [2, 100]** with **multiple seeds**, **temporal robustness** (train earlier, test later), and pick K that generalizes.

3. **Add more exogenous signals**  
   - **Macro**, **event/news**, **risk** proxies (e.g., earnings events, macro calendars) to explain the **~10%** volatile slice.

4. **Company-aware learning**  
   - **Company embeddings** / **domain adaptation** (mixture-of-experts) to share cross-asset structure without washing out idiosyncrasies.

5. **Downstream usage**  
   - **Regime-aware forecasting** (LightGBM / Transformers) using **cluster id** as a feature.  
   - **Risk alerts** when `E(x)` maps to **low-density** regions of latent space.

6. **Longer training & richer nets**  
   - Increase epochs after stability knobs fixed; try deeper MLPs or light **temporal** modules (e.g., TCN).

---


