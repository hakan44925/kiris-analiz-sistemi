import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - İnteraktif Statik", layout="wide")

st.title("🏗️ Profesyonel Statik Simülatör (Slider Kontrollü)")
st.write("Sürgüleri kaydırarak diyagramlardaki değişimi anlık izleyebilirsiniz.")

# --- SIDEBAR (SÜRGÜLER) ---
with st.sidebar:
    st.header("📐 Geometri Ayarları")
    L1 = st.slider("L1 Uzunluğu (m)", 1.0, 10.0, 4.0)
    L2 = st.slider("L2 Uzunluğu (m)", 0.5, 5.0, 1.0)
    L3 = st.slider("L3 Uzunluğu (m)", 1.0, 10.0, 2.0)
    
    st.header("🔴 Yük Ayarları")
    q = st.slider("Yayılı Yük (kN/m)", 0.0, 10.0, 2.0)
    F = st.slider("Tekil Yük (kN)", 0.0, 20.0, 5.0)
    alpha = st.slider("Yük Açısı (Derece)", 0, 180, 60)
    xF = st.slider("Yük Konumu (m)", 0.0, float(L1), 1.0)

# --- ANALİZ MOTORU ---
def analiz():
    total_L = L1 + L2 + L3
    # Yük Bileşenleri
    rad = np.deg2rad(alpha)
    Fx = F * np.cos(rad)
    Fy = F * np.sin(rad)
    
    # Reaksiyonlar (Basitleştirilmiş Gerber Çözümü)
    By = (q * L3) / 2
    Cy = (q * L3) - By
    R2y = (Fy * xF + Cy * (L1 + L2)) / L1
    R1y = Fy + Cy - R2y
    
    # Veri Noktaları
    x = np.linspace(0, total_L, 1000)
    N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Normal Kuvvet
        if xi >= xF and xi <= L1: N[i] = -Fx
        # Kesme ve Moment
        if xi <= xF:
            V[i], M[i] = R1y, R1y * xi
        elif xi <= L1:
            V[i], M[i] = R1y - Fy, R1y * xi - Fy * (xi - xF)
        elif xi <= (L1 + L2):
            V[i], M[i] = R1y - Fy + R2y, R1y * xi - Fy * (xi - xF) + R2y * (xi - L1)
        else: # Yayılı yük
            d = xi - (L1 + L2)
            V[i], M[i] = Cy - q * d, Cy * d - (q * d**2 / 2)
            
    return x, N, V, M

x, N, V, M = analiz()

# --- GÖRSELLEŞTİRME (MATPLOTLIB - STANDART) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.4)

# 1. Normal Kuvvet (Yeşil)
axes[0].plot(x, N, color='green', lw=2)
axes[0].fill_between(x, N, color='green', alpha=0.2)
axes[0].set_title("Normal Kuvvet (N) - kN", fontweight='bold', loc='left')

# 2. Kesme Kuvveti (Kırmızı)
axes[1].plot(x, V, color='red', lw=2)
axes[1].fill_between(x, V, color='red', alpha=0.2)
axes[1].set_title("Kesme Kuvveti (V) - kN", fontweight='bold', loc='left')

# 3. Moment Diyagramı (Mavi)
axes[2].plot(x, M, color='blue', lw=2)
axes[2].fill_between(x, M, color='blue', alpha=0.2)
axes[2].set_title("Moment Diyagramı (M) - kNm", fontweight='bold', loc='left')
axes[2].invert_yaxis() # Çekme tarafı

for ax in axes:
    ax.axhline(0, color='black', lw=1)
    ax.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig)

# --- REAKSİYON KARTLARI ---
st.write("---")
c1, c2, c3 = st.columns(3)
c1.metric("Sol Mesnet (R1y)", f"{np.abs(V[0]):.2f} kN")
c2.metric("Orta Mesnet (R2y)", f"{np.abs(V[np.abs(x-L1).argmin()+1] - V[np.abs(x-L1).argmin()-1]):.2f} kN")
c3.metric("Sağ Mesnet (By)", f"{By:.2f} kN")
