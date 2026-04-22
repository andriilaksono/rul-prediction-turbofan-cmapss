import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk mempercantik tampilan kartu metrik
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD MODEL (DI-CACHE AGAR CEPAT)
# ==========================================
@st.cache_resource
def load_ml_models():
    """Fungsi ini akan memuat model ke memori hanya sekali saat web dibuka"""
    lstm_model = None
    xgb_model = None
    
    # Load LSTM
    lstm_path = 'models/lstm_best_model.h5'
    if os.path.exists(lstm_path):
        try:
            # Gunakan dummy input untuk inisiasi weight jika perlu, atau langsung load .h5
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat LSTM: {e}")
            
    # Load XGBoost
    xgb_path = 'models/xgb_model.pkl'
    if os.path.exists(xgb_path):
        try:
            xgb_model = joblib.load(xgb_path)
        except Exception as e:
            st.sidebar.error(f"Gagal memuat XGBoost: {e}")
            
    return lstm_model, xgb_model

lstm_model, xgb_model = load_ml_models()

# ==========================================
# 3. SIDEBAR (KONTROL & INPUT)
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=150)
st.sidebar.title("⚙️ Konfigurasi Sistem")

# Pilihan Model
st.sidebar.subheader("1. Pilih Model Prediksi")
model_choice = st.sidebar.radio(
    "Arsitektur:",
    ["Deep Learning (LSTM) - Rekomendasi", "Machine Learning (XGBoost)"]
)

st.sidebar.markdown("---")

# Input Data
st.sidebar.subheader("2. Input Data Sensor")
input_method = st.sidebar.radio("Metode Input:", ["Demo Interaktif (Manual)", "Upload File CSV"])

# Variabel untuk menampung data input
input_data = None

if input_method == "Demo Interaktif (Manual)":
    st.sidebar.info("Geser slider untuk mensimulasikan nilai sensor utama (T50 & Ps30).")
    # Sensor yang paling berpengaruh dari hasil XAI kita
    t50_val = st.sidebar.slider("T50 (Suhu LPT Outlet)", min_value=1380.0, max_value=1440.0, value=1410.0, step=0.5)
    ps30_val = st.sidebar.slider("Ps30 (Tekanan Statis HPC)", min_value=46.5, max_value=48.5, value=47.5, step=0.1)
    
    # Tombol Prediksi Manual
    predict_btn = st.sidebar.button("Prediksi RUL Sekarang", type="primary")

elif input_method == "Upload File CSV":
    st.sidebar.info("Upload file CSV berisi log 30 siklus terakhir dari sensor mesin.")
    uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=['csv'])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.sidebar.success("File berhasil dimuat!")
    
    predict_btn = st.sidebar.button("Prediksi RUL dari File", type="primary")

# ==========================================
# 4. KONTEN UTAMA (DASHBOARD)
# ==========================================
st.title("✈️ Predictive Maintenance: Turbofan Engine")
st.markdown("Dashboard cerdas untuk memprediksi **Remaining Useful Life (RUL)** mesin pesawat berbasis dataset NASA C-MAPSS FD001.")

st.markdown("---")

# Logika Prediksi
if predict_btn:
    with st.spinner('Menghitung estimasi sisa umur mesin...'):
        rul_prediction = 0
        
        # MOCKUP LOGIC: Jika model asli belum ditaruh di folder 'models', gunakan rumus simulasi
        # (Silakan hapus blok simulasi ini nanti jika model asli sudah berjalan sempurna)
        if (model_choice == "Deep Learning (LSTM) - Rekomendasi" and lstm_model is None) or \
           (model_choice == "Machine Learning (XGBoost)" and xgb_model is None):
            
            st.warning(f"⚠️ File model asli belum terdeteksi di folder `models/`. Menggunakan mode simulasi berdasarkan input manual.")
            # Simulasi sederhana: RUL turun drastis jika suhu/tekanan naik
            sim_rul = 150 - ((t50_val - 1380) * 2) - ((ps30_val - 46.5) * 20)
            rul_prediction = max(0, int(sim_rul))
            
        else:
            # ==========================================
            # AREA EKSEKUSI MODEL ASLI (JIKA FILE TERSEDIA)
            # ==========================================
            try:
                if input_method == "Demo Interaktif (Manual)":
                    st.error("Untuk menggunakan model asli, harap gunakan metode 'Upload File CSV' agar fitur (21 sensor) lengkap.")
                    st.stop()
                
                # Preprocessing data dari CSV (Sesuaikan dengan pipeline aslimu)
                # Contoh: scaling menggunakan joblib scaler
                # scaler = joblib.load('models/scaler.pkl')
                # scaled_data = scaler.transform(input_data)
                
                if "LSTM" in model_choice:
                    # Pastikan bentuknya (1, 30, n_features)
                    # input_3d = scaled_data[-30:].reshape(1, 30, 21) 
                    # rul_prediction = float(lstm_model.predict(input_3d)[0][0])
                    pass # Ganti pass dengan kode di atas
                else:
                    # XGBoost
                    # input_2d = scaled_data[-1:].reshape(1, -1)
                    # rul_prediction = float(xgb_model.predict(input_2d)[0])
                    pass
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data ke model: {e}")
                st.stop()
        
        # ==========================================
        # 5. VISUALISASI HASIL PREDIKSI
        # ==========================================
        # Menentukan Status Kesehatan
        if rul_prediction > 80:
            status = "SEHAT (Aman untuk Terbang)"
            color = "green"
            progress_val = min(100, int((rul_prediction/150)*100))
        elif rul_prediction > 30:
            status = "PERINGATAN (Butuh Perawatan Segera)"
            color = "orange"
            progress_val = min(100, int((rul_prediction/150)*100))
        else:
            status = "KRITIS (Ganti Mesin Segera!)"
            color = "red"
            progress_val = min(100, int((rul_prediction/150)*100))

        # Tampilkan dalam kolom
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Model Digunakan", value="LSTM" if "LSTM" in model_choice else "XGBoost")
        with col2:
            st.metric(label="Prediksi Sisa Umur (RUL)", value=f"{rul_prediction} Siklus")
        with col3:
            st.markdown(f"**Status Kesehatan Mesin:**")
            st.markdown(f"<h3 style='color: {color}; margin-top: 0;'>{status}</h3>", unsafe_allow_html=True)
            
        st.markdown("### Indikator Keausan Mesin")
        st.progress(progress_val, text=f"Estimasi Siklus Tersisa (Maksimal referensi ~150)")

        # Tambahan Info XAI (Explainable AI)
        with st.expander("📊 Lihat Detail & Analisis Sensor (Explainable AI)"):
            st.write("""
            Berdasarkan ekstraksi fitur (Integrated Gradients / SHAP), penurunan angka RUL secara signifikan 
            berkorelasi kuat dengan kenaikan suhu pada **LPT Outlet (T50)** dan tekanan statis pada **HPC Outlet (Ps30)**.
            """)
            if input_method == "Upload File CSV" and input_data is not None:
                st.line_chart(input_data) # Menampilkan tren raw data dari CSV yang diupload
else:
    st.info("👈 Silakan atur konfigurasi di sidebar dan klik **Prediksi Sekarang** untuk melihat hasil.")