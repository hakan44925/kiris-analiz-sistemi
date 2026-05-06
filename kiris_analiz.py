import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hakan Çırak - İnteraktif Gerber Analiz", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A; text-align: center; font-family: sans-serif;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Etkileşimli Statik Simülatör")
st.write("---")

# --- SIDEBAR (SÜRGÜLER - GÖRSELDEKİ GİBİ) ---
with st.sidebar:
    st.header("📐 Geometri")
    L1 = st.slider("L1 Uzunluğu (m)", 1.0, 10.0, 4.0)
    L2 = st.slider("L2 Uzunluğu (m)", 0.5, 5.0, 1.0)
    L3 = st.slider("L3 Uzunluğu (m)", 1.0, 10.0, 2.0)
    
    st.header("🔴 Yükler")
    q = st.slider("Yayılı Yük (q - kN/m)", 0.0, 10.0, 2.0)
    F = st.slider("Tekil Yük (F - kN)", 0.0, 20.0, 5.0)
    alpha = st.slider("Yük Açısı (α - Derece)", 0, 180, 60)
    xF = st.slider("F Yükü Konumu (m)", 0.0, L1, 1.0)

# --- ANALİZ MOTORU ---
def analiz():
    total_L = L1 + L2 + L3
    mafsal_x = L1 + L2 # Görseldeki mafsal konumu
    
    # Yük Bileşenleri
    rad = np.deg2rad(alpha)
    Fx = F * np.cos(rad)
    Fy = F * np.sin(rad)
    
    # Reaksiyon Hesapları (Statik Ayırma Metodu)
    # 1. Sağ Parça (Mafsal - Sağ Mesnet)
    # By * L3 - (q * L3 * L3/2) = 0
    By = (q * L3) / 2
    Cy = (q * L3) - By # Mafsal tepkisi
    
    # 2. Sol Parça (Sol Mesnet - Mafsal)
    # ΣM_sol_mesnet = 0 => R2y * L1 - Fy * xF - Cy * (L1 + L2) = 0
    R2y = (Fy * xF + Cy * (L1 + L2)) / L1
    R1y = Fy + Cy - R2y
    
    # Diyagram Verileri
    x = np.linspace(0, total_L, 1000)
    N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Normal Kuvvet (N) - Eksenel
        if xi < xF: N[i] = 0
        elif xi <= L1: N[i] = -Fx
        else: N[i] = 0
        
        # Kesme (V) ve Moment (M)
        if xi <= xF:
            V[i] = R1y
            M[i] = R1y * xi
        elif xi <= L1:
            V[i] = R1y - Fy
            M[i] = R1y * xi - Fy * (xi - xF)
        elif xi <= (L1 + L2):
            V[i] = R1y - Fy + R2y
            M[i] = R1y * xi - Fy * (xi - xF) + R2y * (xi - L1)
        else: # Yayılı yük bölgesi
            dist_mafsal = xi - (L1 + L2)
            V[i] = Cy - q * dist_mafsal
            M[i] = Cy * dist_mafsal - (q * dist_mafsal**2 / 2)
            
    return x, N, V, M

x, N, V, M = analiz()

# --- GÖRSELLEŞTİRME (PLOTLY - ETKİLEŞİMLİ) ---
fig = go.Figure()

# Normal Kuvvet (Yeşil)
fig.add_trace(go.Scatter(x=x, y=N, fill='tozeroy', name='Normal (N) [kN]', line=dict(color='green', width=2)))

# Kesme Kuvveti (Kırmızı)
fig.add_trace(go.Scatter(x=x, y=V, fill='tozeroy', name='Kesme (V) [kN]', line=dict(color='red', width=2)))

# Moment (Mavi)
fig.add_trace(go.Scatter(x=x, y=M, fill='tozeroy', name='Moment (M) [kNm]', line=dict(color='blue', width=2)))

fig.update_layout(
    height=600,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=50, b=20)
)
fig.update_yaxes(autorange="reversed", selector=dict(name='Moment (M) [kNm]')) # Moment ters çizim

st.plotly_chart(fig, use_container_width=True)

# Sonuç Paneli
c1, c2, c3 = st.columns(3)
c1.metric("Max Moment", f"{np.max(np.abs(M)):.1f} kNm")
c2.metric("Max Kesme", f"{np.max(np.abs(V)):.1f} kN")
c3.metric("Eksenel Etki", f"{np.max(np.abs(N)):.1f} kN")
