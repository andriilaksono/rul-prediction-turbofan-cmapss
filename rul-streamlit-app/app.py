import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN
nasa_logo_url = "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg"

st.set_page_config(
    page_title="Pemantauan Individu Mesin", 
    page_icon=nasa_logo_url, 
    layout="wide"
)

# 2. GENERATOR DATA MOCKUP 
@st.cache_data
def get_engine_data(engine_id):
    np.random.seed(engine_id)
    max_life = np.random.randint(150, 250)
    cycles = np.arange(1, max_life + 1)
    true_rul = np.clip(max_life - cycles, 0, 125)
    pred_rul = np.clip(true_rul + np.random.normal(0, 4, size=max_life), 0, None)
    
    # Sensor Data
    t50 = 1390 + (cycles / max_life) * 40 + np.random.normal(0, 1, size=max_life)
    ps30 = 47.0 + (cycles / max_life) * 1.5 + np.random.normal(0, 0.05, size=max_life)
    phi = 520 - (cycles / max_life) * 3 + np.random.normal(0, 0.1, size=max_life)
    
    return pd.DataFrame({
        'Cycle': cycles, 'True RUL': true_rul, 'Predicted RUL': pred_rul,
        'T50 (Suhu)': t50, 'Ps30 (Tekanan)': ps30, 'phi (Rasio BBM)': phi
    }), max_life

# 3. AREA KONTROL UTAMA
st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <img src="{nasa_logo_url}" width="70" style="margin-right: 20px;">
        <h1 style="margin: 0;">Pemantauan Individu Mesin Turbofan</h1>
    </div>
    """, 
    unsafe_allow_html=True
)
st.markdown("Fokus pada analisis trajektori degradasi dan dinamika sensor per unit mesin berdasarkan dataset NASA C-MAPSS.")

# Baris Input 
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    engine_id = st.selectbox("Pilih ID Mesin:", [i for i in range(1, 101)], index=0)

df_ts, max_life = get_engine_data(engine_id)

with col_input2:
    current_cycle = st.slider("Pilih Siklus Saat Ini:", 1, max_life, max_life-15)

df_current = df_ts[df_ts['Cycle'] <= current_cycle]
latest_rul = int(df_current['Predicted RUL'].iloc[-1])

# 4. KPI METRICS
st.markdown("---")
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Siklus Berjalan", current_cycle)
with m2:
    st.metric("Estimasi Sisa Umur (RUL)", f"{latest_rul} Siklus")
with m3:
    if latest_rul > 80:
        st.success("STATUS: AMAN")
    elif latest_rul > 30:
        st.warning("STATUS: PERINGATAN")
    else:
        st.error("STATUS: KRITIS")

# 5. VISUALISASI
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Trajektori RUL**")
    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=df_current['Cycle'], y=df_current['True RUL'], name="Actual", line=dict(dash='dash', color='gray')))
    fig_rul.add_trace(go.Scatter(x=df_current['Cycle'], y=df_current['Predicted RUL'], name="LSTM Pred", line=dict(color='#2ca02c', width=3)))
    fig_rul.add_hrect(y0=0, y1=30, line_width=0, fillcolor="red", opacity=0.15, annotation_text="Zona Bahaya", annotation_position="bottom left")
    
    fig_rul.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
        xaxis_title="Siklus Operasional (Time Cycle)",  
        yaxis_title="Sisa Umur Mesin (RUL)")
    st.plotly_chart(fig_rul, use_container_width=True)

with col_g2:
    st.markdown("**Dinamika Sensor (Normalisasi)**")
    selected_sensors = st.multiselect("Pilih Sensor:", ['T50 (Suhu)', 'Ps30 (Tekanan)', 'phi (Rasio BBM)'], default=['T50 (Suhu)', 'Ps30 (Tekanan)', 'phi (Rasio BBM)'])
    
    if selected_sensors:
        fig_sens = go.Figure()
        for s in selected_sensors:
            s_data = df_current[s]
            s_norm = (s_data - s_data.min()) / (s_data.max() - s_data.min()) if s_data.max() != s_data.min() else s_data
            fig_sens.add_trace(go.Scatter(x=df_current['Cycle'], y=s_norm, name=s))
        
        fig_sens.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
            xaxis_title="Siklus Operasional (Time Cycle)",
            yaxis_title="Skala Normalisasi (0 - 1)")
        st.plotly_chart(fig_sens, use_container_width=True)