# 🔧 Prediksi Remaining Useful Life (RUL) Turbin Jet — C-MAPSS Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-green)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Proyek ini mengimplementasikan dan membandingkan dua pendekatan machine learning untuk memprediksi **Remaining Useful Life (RUL)** mesin turbin jet menggunakan dataset **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dari NASA — mencakup 4 subset (FD001–FD004).

---

## 📑 Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Explainable AI (XAI)](#explainable-ai-xai)

---

## Latar Belakang

Prediksi RUL merupakan komponen kritis dalam **Predictive Maintenance (PdM)** — memungkinkan tim perawatan untuk mengganti komponen tepat waktu sebelum terjadi kegagalan, sehingga mengurangi downtime dan biaya operasional.

Proyek ini membandingkan dua paradigma model:

| Model | Tipe | XAI Method |
|---|---|---|
| **XGBoost** | Gradient Boosted Trees (fitur 2D) | SHAP TreeExplainer |
| **Bi-LSTM + Attention** | Deep Learning (sequence temporal) | Integrated Gradients |

Evaluasi dilakukan pada **4 subset C-MAPSS** yang berbeda tingkat kompleksitasnya, dan hasilnya dibandingkan dengan benchmark literatur SOTA.

---

## Dataset

Dataset yang digunakan adalah **NASA C-MAPSS Turbofan Engine Degradation Simulation**, tersedia di [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

### Karakteristik Subset

| Subset | Operating Conditions | Fault Modes | Train Engines | Test Engines |
|--------|---------------------|-------------|---------------|--------------|
| FD001  | 1                   | 1           | 100           | 100          |
| FD002  | 6                   | 1           | 260           | 259          |
| FD003  | 1                   | 2           | 100           | 100          |
| FD004  | 6                   | 2           | 249           | 248          |

Setiap record memiliki **21 kolom sensor** (T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32) dan 3 kolom operational settings.

> **Catatan:** Unduh dataset dan letakkan file `.txt` di direktori `data/`:
> `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, … (hingga FD004)

---

## Arsitektur Model

### 1. XGBoost Regressor

Menggunakan representasi fitur 2D (rata-rata statistik per engine cycle). Hyperparameter berbasis kombinasi best-practice dari literatur SOTA:

```
n_estimators    : 500 (800 untuk FD002/FD004)
learning_rate   : 0.05
max_depth       : 6 (8 untuk FD002/FD004)
min_child_weight: 3
subsample       : 0.8
colsample_bytree: 0.8
```

### 2. Bi-LSTM + Self-Attention

```
Input  (window=30, n_features)
   ↓
Bidirectional LSTM (64 units, return_sequences=True) → Dropout(0.2)
   ↓
Bidirectional LSTM (32 units, return_sequences=True) → Dropout(0.2)
   ↓
Self-Attention Layer (Bahdanau-style, additive)
   ↓
Dense (32, ReLU) → Dropout(0.2)
   ↓
Dense (1)  →  RUL Prediction
```

---

## Struktur Proyek

```
.
├── data/                               # Dataset mentah C-MAPSS (tidak di-commit)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   └── ...  (FD002–FD004)
│
├── notebook/                           # Notebook utama penelitian
│   ├── 01_data_preprocessing.ipynb     # Preprocessing & feature engineering
│   ├── 02_exploratory_data_analysis.ipynb  # EDA komparatif 4 subset
│   ├── 03_xgboost_model.ipynb          # Training XGBoost + SHAP
│   ├── 04_Bi-lstm_model.ipynb          # Training Bi-LSTM + Integrated Gradients
│   └── 05_comparison_visualization.ipynb   # Komparasi & XAI side-by-side
│
├── processed/                          # Output preprocessing per-subset (tidak di-commit)
│   ├── FD001/
│   │   ├── train_2d.csv                # Fitur tabular + label RUL
│   │   ├── test_2d.csv
│   │   ├── train_seq.npy               # Sequence 3D untuk Bi-LSTM
│   │   ├── test_seq.npy
│   │   ├── xgb_metrics.json
│   │   ├── lstm_metrics.json
│   │   ├── xgb_shap_global.csv
│   │   └── lstm_ig_global.csv
│   ├── FD002/ ... FD004/
│   └── master_comparison.csv           # Tabel komparasi akhir
│
├── models/                             # Model tersimpan (tidak di-commit)
│   ├── xgb_FD001.pkl
│   ├── bilstm_FD001.h5
│   └── ...
│
├── output/                             # Semua visualisasi PNG (tidak di-commit)
│   ├── eda_lifespan_4subsets.png
│   ├── eda_correlation_4subsets.png
│   ├── xgb_pred_FD001.png
│   ├── comparison_metrics_grid.png
│   └── ...
│
├── rul-streamlit-app/                  # Aplikasi web demo prediksi RUL (Streamlit)
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Catatan:** Direktori `data/`, `processed/`, `models/`, dan `output/` tidak di-commit ke repository (lihat `.gitignore`). Semua direktori tersebut di-generate ulang dengan menjalankan notebook secara berurutan.

---

## Instalasi

### Prasyarat

- Python 3.10+
- pip

### Langkah Instalasi

```bash
# 1. Clone repository
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependensi
pip install -r requirements.txt
```

### Dependensi Utama

```
numpy==1.26.4          # Komputasi numerik
pandas==2.2.2          # Manipulasi data
scikit-learn==1.5.2    # Preprocessing & metrik
xgboost==2.1.1         # Model XGBoost
tensorflow==2.17.0     # Model Bi-LSTM
shap==0.46.0           # Explainability XGBoost (SHAP)
matplotlib==3.8.4      # Visualisasi
seaborn==0.13.2        # Visualisasi statistik
```

---

## Cara Penggunaan

Jalankan notebook secara berurutan dari dalam folder `notebook/`:

### 1. Preprocessing Data

```bash
jupyter notebook notebook/01_data_preprocessing.ipynb
```

Menghasilkan:
- Pembersihan sensor konstan/non-informatif per subset
- RUL labeling dengan metode *piecewise-linear* (threshold = 125 siklus)
- **Condition-aware normalization** menggunakan KMeans (k=6) untuk FD002/FD004
- File `train_2d.csv`, `test_2d.csv`, `train_seq.npy`, `test_seq.npy` di `processed/`

### 2. Exploratory Data Analysis

```bash
jupyter notebook notebook/02_exploratory_data_analysis.ipynb
```

Menghasilkan:
- Distribusi engine lifespan per subset
- Korelasi sensor terhadap RUL (heatmap)
- Trajektori sensor kunci (sensor T24)
- Scatter plot operating conditions (FD002/FD004)

### 3. Training XGBoost + SHAP

```bash
jupyter notebook notebook/03_xgboost_model.ipynb
```

Menghasilkan:
- Model `.pkl` per subset di `models/`
- Metrik evaluasi (RMSE, MAE, R², NASA Score)
- Visualisasi SHAP summary & waterfall plot
- File `xgb_metrics.json` dan `xgb_shap_global.csv`

### 4. Training Bi-LSTM + Integrated Gradients

```bash
jupyter notebook notebook/04_Bi-lstm_model.ipynb
```

Menghasilkan:
- Model `.h5` per subset di `models/`
- Metrik evaluasi dan learning curves
- Integrated Gradients attribution per sensor
- File `lstm_metrics.json` dan `lstm_ig_global.csv`

### 5. Komparasi & Visualisasi Akhir

```bash
jupyter notebook notebook/05_comparison_visualization.ipynb
```

Menghasilkan:
- Tabel master komparasi kedua model × 4 subset
- Bar chart 4×4 semua metrik
- XAI side-by-side (SHAP vs Integrated Gradients)
- Benchmarking vs literatur SOTA

---

## Hasil Evaluasi

Model dievaluasi menggunakan 4 metrik:

| Metrik | Keterangan |
|--------|-----------|
| **RMSE** | Root Mean Squared Error — semakin rendah semakin baik |
| **MAE** | Mean Absolute Error — semakin rendah semakin baik |
| **R²** | Koefisien determinasi — semakin tinggi semakin baik |
| **NASA Score** | semakin rendah semakin baik; memberikan penalti lebih besar pada prediksi *terlalu optimis* |

> Hasil lengkap tersimpan di `processed/master_comparison.csv` setelah menjalankan Notebook 05.

---

## Explainable AI (XAI)

Proyek ini menggunakan dua metode XAI untuk memahami keputusan model:

### SHAP (untuk XGBoost)
Menggunakan `shap.TreeExplainer` — memberikan kontribusi fitur berbasis nilai Shapley yang akurat dan efisien untuk tree-based models.

### Integrated Gradients (untuk Bi-LSTM)
Mengimplementasikan metode Sundararajan et al. (2017) secara native dengan TensorFlow:

$$\text{IG}_i(x) = (x_i - x_i') \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} \, d\alpha$$

Visualisasi XAI side-by-side memungkinkan analisis apakah kedua model "sepakat" terhadap sensor yang paling berpengaruh pada prediksi RUL.

---

## Aplikasi Demo (Streamlit)

Folder `rul-streamlit-app/` berisi aplikasi web interaktif untuk melakukan prediksi RUL secara langsung menggunakan model yang telah dilatih.

```bash
cd rul-streamlit-app
pip install -r requirements.txt   # jika ada requirements terpisah
streamlit run app.py
```

> Pastikan model (`.pkl` / `.h5`) sudah dilatih dan tersimpan di `models/` sebelum menjalankan aplikasi.

---

