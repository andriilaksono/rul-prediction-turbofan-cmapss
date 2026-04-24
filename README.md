# 🔧 Remaining Useful Life (RUL) Prediction of Turbofan Engines
### A Comparative Analysis of XGBoost and LSTM on NASA C-MAPSS FD001
 
> This research compares the performance of **XGBoost** and **LSTM** in predicting the *Remaining Useful Life* (RUL) of turbofan engines using the NASA C-MAPSS benchmark dataset, sub-dataset FD001.
 
---
 
## 📋 Table of Contents
 
- [Background](#background)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)
 
---
 
## 📖 Background
 
*Remaining Useful Life* (RUL) is the estimated number of operational cycles remaining before a component experiences failure. Accurate RUL prediction is critical in *Predictive Maintenance* (PdM) applications across the aerospace and manufacturing industries, enabling timely interventions that reduce unplanned downtime and maintenance costs.
 
This study compares two machine learning approaches:
- **XGBoost** — a gradient boosting model applied to tabular features enriched with rolling statistics
- **LSTM** — a deep learning model that leverages sequential temporal patterns in sensor data
 
---
 
## 📦 Dataset
 
**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
 
| Sub-dataset | Operating Conditions | Fault Modes | Train Units | Test Units |
|-------------|----------------------|-------------|-------------|------------|
| **FD001**   | 1                    | 1           | 100         | 100        |
 
Download the dataset from:
- Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- NASA: https://data.nasa.gov
 
**Required files:**
```
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```
 
**Column Description:**
 
| Column | Description |
|--------|-------------|
| `unit_number` | Engine ID (1–100) |
| `time_in_cycles` | Current operational cycle |
| `operational_setting_1,2,3` | Engine operating conditions |
| `sensor_1` – `sensor_21` | Readings from 21 onboard sensors |
 
---
 
## 📁 Notebook Descriptions

This project is organized into five main notebooks, where each notebook represents one stage of the RUL prediction pipeline.

### 📓 01_data_preprocessing.ipynb
This notebook performs all data preparation steps before modeling. It loads the raw NASA C-MAPSS FD001 dataset, assigns column names, calculates Remaining Useful Life (RUL) targets, applies RUL capping, removes low-variance sensors, and normalizes selected features using MinMaxScaler. The processed datasets are then saved for downstream modeling. :contentReference[oaicite:0]{index=0}

**Main outputs:**
- `train_processed.csv`
- `test_processed.csv`
- `test_last_cycle.csv`
- `feature_cols.json`

---

### 📓 02_exploratory_data_analysis.ipynb
This notebook focuses on exploratory data analysis (EDA) to better understand engine degradation patterns and feature behavior. It visualizes sensor trajectories over time, feature correlations with RUL, engine lifespan distributions, and health-state transitions. The generated plots are useful for interpretation and publication figures. :contentReference[oaicite:1]{index=1}

**Main outputs:**
- Sensor trend visualizations
- Correlation analysis plots
- Health-state charts
- Lifespan distribution charts

---

### 📓 03_xgboost_model.ipynb
This notebook builds the XGBoost regression model for RUL prediction. It engineers rolling statistical features from time-series sensor data, trains the model, evaluates performance using multiple regression metrics, and applies explainability techniques such as SHAP for feature importance analysis. :contentReference[oaicite:2]{index=2}

**Main outputs:**
- `xgb_model.pkl`
- `xgb_predictions.csv`
- SHAP visualizations
- Feature importance plots

---

### 📓 04_lstm_model.ipynb
This notebook develops the LSTM deep learning model to capture sequential dependencies in sensor data. It creates sliding-window sequences, trains the network with callbacks such as early stopping, evaluates prediction accuracy, and applies Integrated Gradients for model interpretability. :contentReference[oaicite:3]{index=3}

**Main outputs:**
- `lstm_base_model.h5`
- `lstm_metrics.json`
- Learning curve plots
- Integrated Gradients visualizations

---

### 📓 05_comparison_visualization.ipynb
This notebook compares the final performance of XGBoost and LSTM models. It combines prediction results, summarizes evaluation metrics (RMSE, MAE, R², NASA Score), and creates final comparison figures suitable for reports, presentations, or journal publications. :contentReference[oaicite:4]{index=4}

**Main outputs:**
- `all_predictions.csv`
- Final comparison charts
- Performance summary tables
 
---
 
## 🔄 Pipeline Overview
 
```
Raw Data (C-MAPSS FD001)
        │
        ▼
┌──────────────────────┐
│  01 · Preprocessing  │  → EDA, RUL Labeling (cap=125), Feature Selection,
└──────────────────────┘    MinMax Normalization
        │
        ▼
   processed/
   ├── train_processed.csv
   └── test_last_cycle.csv
        │
   ┌────┴────┐
   ▼         ▼
┌──────────┐ ┌──────────┐
│ 02 · XGB │ │ 03 · LSTM│
└──────────┘ └──────────┘
Rolling        Sequence
Features       (window=30)
(window=10)
   │              │
   └──────┬───────┘
          ▼
┌───────────────────────────┐
│ 04 · Comparison & Output  │  → RMSE, MAE, R², NASA Score, Journal Figures
└───────────────────────────┘
```
 
---
 
## ⚙️ Installation
 
### Prerequisites
- Python >= 3.9
- Jupyter Notebook or JupyterLab
 
### Install Dependencies
 
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow joblib
```
 
Or using a virtual environment (recommended):
 
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
 
# Install packages
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow joblib
```
 
### Recommended Library Versions
 
| Library | Version |
|---------|---------|
| Python | >= 3.9 |
| TensorFlow | >= 2.12 |
| XGBoost | >= 1.7 |
| scikit-learn | >= 1.2 |
| pandas | >= 1.5 |
| numpy | >= 1.23 |
 
---
 
## ▶️ How to Run
 
Run the notebooks **in order**:
 
```bash
# 1. Launch Jupyter
jupyter notebook
 
# 2. Execute notebooks in the following sequence:
#    01_data_preprocessing.ipynb      ← must run first
#    02_xgboost_model.ipynb
#    03_lstm_model.ipynb
#    04_comparison_visualization.ipynb
```
 
> ⚠️ **Important:** Notebooks 02, 03, and 04 depend on outputs generated by Notebook 01.  
> Ensure Notebook 01 has been fully executed before proceeding to the next steps.
 
---
 
## 🏗️ Model Architecture
 
### XGBoost
 
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 500 (with early stopping) |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |
| Additional features | Rolling mean & std (window = 10 cycles) |
| Validation strategy | 5-Fold Cross-Validation + Early Stopping |
 
### LSTM
 
```
Input  →  (30 cycles × n_features)
               ↓
  LSTM(128, return_sequences=True)
  BatchNormalization → Dropout(0.3)
               ↓
  LSTM(64, return_sequences=False)
  BatchNormalization → Dropout(0.3)
               ↓
    Dense(32, activation='relu')
          Dropout(0.2)
               ↓
    Dense(1, activation='linear')   ←  RUL Output
```
 
| Parameter | Value |
|-----------|-------|
| Sequence length | 30 cycles |
| Optimizer | Adam (lr = 0.001) |
| Loss function | MSE |
| Batch size | 256 |
| Max epochs | 200 |
| Early stopping patience | 20 epochs |
| LR scheduler | ReduceLROnPlateau (factor = 0.5) |
 
---
 
## 📊 Evaluation Metrics
 
| Metric | Formula | Description |
|--------|---------|-------------|
| **RMSE** | √(Σ(ŷ−y)²/n) | Root Mean Squared Error — penalizes large deviations |
| **MAE** | Σ\|ŷ−y\|/n | Mean Absolute Error — robust to outliers |
| **R²** | 1 − SS_res/SS_tot | Coefficient of determination — proportion of variance explained |
| **NASA Score** | Σ(e^(d/13)−1) if d<0 ; Σ(e^(d/10)−1) if d≥0 | Asymmetric penalty score |
 
> **NASA Score** imposes a **higher penalty** when the predicted RUL exceeds the actual RUL (late prediction), as failing to detect imminent engine failure is considerably more hazardous than predicting it too early. **Lower is better.**
 
### RUL Labeling — Piecewise Linear Degradation
 
RUL is capped at **125 cycles** following the standard convention in C-MAPSS literature:
 
```
RUL(t) = min(max_cycle − t, 125)
```
 
This prevents the model from being overly influenced by the early healthy phase of engine operation, where degradation signals are not yet informative.
 
---
 
## 📈 Results
 
*(To be updated after experiments are complete)*
 
| Model | RMSE ↓ | MAE ↓ | R² ↑ | NASA Score ↓ |
|-------|--------|-------|------|--------------|
| XGBoost | 17.69 | 12.89 | 0.8187 | 843 |
| LSTM    | 15.52 | 12.25 | 0.8605 | 349 |
 
All comparison figures are saved in the `output/` folder and are ready for journal submission.
 
## 👤 Author
 
| | |
|-|-|
| **Name** | Andri Laksono |
| **Study Program** | Informatics |
| **Institution** | Universitas Amikom Yogyakarta |
| **Year** | 2026 |
 
---
 
## 📄 License
 
This project is intended for academic research purposes. Please cite this work appropriately if you use the code or methodology in your own research.
 
---
 
> *This research is conducted as part of a journal-track graduation requirement (in lieu of a conventional thesis).*