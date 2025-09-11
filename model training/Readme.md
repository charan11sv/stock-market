# Stock Forecasting & Experiments — Repository Overview

This repository contains all experiments, documentation, and notebooks related to my work on **multi-horizon stock prediction** using **LightGBM** and exploratory **hybrid approaches (LightGBM + Transformers)**.

---

##  Repository Structure

### 1. Core Training Notebooks & Docs
- **`stock_lgbm_training_10days_pred.ipynb`**
  - Final notebook for **10-day ahead stock prediction** using **multi-output LightGBM**.
  - Trains on full dataset (2000+ companies).
  - Includes preprocessing with Polars, hyperparameter tuning (manual + Optuna), KFold CV confirmation, and **MAPE-based evaluation**.
  - Model is saved as `multi_model_stock1.pkl`.

- **`model training/lgbm_training_readme.md`**
  - Detailed README for the above notebook.
  - Explains objectives, workflow, KFold verification, test set results, and insights.

---

### 2. Preliminary Experiments
- **`prelim_model_training - no outputs.ipynb`**
  - Early trials with feature engineering, LightGBM, and Transformer models.
  - Helped shape final pipeline choices.

- **`model training/prelim_training_documentation.md`**
  - Detailed retrospective of the preliminary notebook.
  - Covers what was tried (LightGBM, Transformer, ensemble sketches), resource constraints, lessons learned, and next steps.

---

### 3. Cloud Training
- **`lgbm via aws.ipynb`**
  - Notebook adapted for running LightGBM training on **AWS**.
  - Useful for large-scale experiments beyond local/Kaggle resources.

---

### 4. Testing & Simulation (New)
- **`testing-stock-model-1.ipynb`**
  - Notebook for **testing results** of the trained LightGBM model.
  - Evaluates saved predictions and performance on unseen data.
  - Used primarily for verification of training outcomes.

- **`stock-trade-simulation-test-version.ipynb`**
  - Ongoing work-in-progress notebook.
  - Exploring **trading simulations** using model predictions.
  - Still experimental and not finalized.

---

##  How to Navigate

1. Start with **`stock_lgbm_training_10days_pred.ipynb`** for the final LightGBM 10-day prediction pipeline.  
   - See **`model training/lgbm_training_readme.md`** for the detailed documentation.  

2. To understand the early exploration and hybrid attempts, check:  
   - **`prelim_model_training - no outputs.ipynb`**  
   - **`model training/prelim_training_documentation.md`**  

3. For cloud-scale experiments, refer to:  
   - **`lgbm via aws.ipynb`** 

4. For evaluation and simulations:  
   - **`testing-stock-model-1.ipynb`** → testing predictions  
   - **`stock-trade-simulation-test-version.ipynb`** → simulation experiments (WIP)  

---

##  Roadmap

- Continue refining the **trading simulation**.  
- Explore hybrid **Transformer + LightGBM** approaches with more compute.  
- Move toward deployment-ready pipelines.

