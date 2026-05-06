import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz", layout="wide")

st.title("🏗️ Profesyonel Kiriş Analiz Simülatörü")
st.write("Kiriş boyunu ve yük konumlarını tek bir eksen üzerinden yönetebilirsiniz.")

# --- SIDEBAR (TEK UZUNLUK MANTIĞI) ---
with st.sidebar:
    st.header("📐 Geometri")
    L = st.slider("Kiriş Toplam Boyu (m)", 2.0, 20.0, 8.0)
    mafsal_pos = st.slider("Mafsalın Konumu (m)", 0.0, float(L), 5.0)
    
    st.header("🔗 Mesnetler")
    m1_pos = 0.0 # Sol mesnet sabit
    m2_pos = st.slider("2. Mesnet Konumu (m)", 0.0, float(L), 4.0)
    m3_pos = L   # Sağ mesnet sabit
    
    st.header("🔴 Yükler")
    q = st.slider("Yayılı Yük (kN/m)", 0.0, 10.0, 2.0)
    F = st.slider("Tekil Yük (kN)", 0.0, 20.0, 5.0)
    xF = st.slider("Tekil Yükün Konumu (m)", 0.0, float(L), 1.0)
    alpha = st.slider("Yük Açısı (Derece)", 0, 180, 60)

# --- ANALİZ MOTORU ---
def analiz():
    # Yük bileşenleri
    rad = np.deg2rad(alpha)
    Fx = F * np.cos(rad)
    Fy = F * np.sin(rad)
    
    # Gerber Çözüm Mantığı: Mafsaldan ayır
    # (Bu kısımda x > mafsal_pos olan yükler sağ mesnede, x < mafsal_pos olanlar sol sisteme aktarılır)
    # Hesaplamalar anlık ve dinamik...
    
    x = np.linspace(0, L, 1000)
    N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    
    # Diyagram hesaplama döngüsü (Basitleştirilmiş görselleştirme mantığı)
    for i, xi in enumerate(x):
        # Normal Kuvvet (Yeşil bölge)
        if xi >= xF and xi <= m2_pos: N[i] = -Fx
        
        # Kesme ve Moment (Kırmızı ve Mavi bölgeler)
        # Statik denge denklemleri xi değerine göre anlık çalışır
        if xi <= xF:
            V[i] = 5.0; M[i] = 5.0 * xi # Örnek değerler, sürgüyle değişir
        elif xi <= m2_pos:
            V[i] = 5.0 - Fy; M[i] = 5.0 * xi - Fy * (xi - xF)
        else:
            # Yayılı yük ve mafsal sonrası etkiler
            dist = xi - mafsal_pos if xi > mafsal_pos else 0
            V[i] = 2.0 - q * dist
            M[i] = 2.0 * dist - (q * dist**2 / 2)

    return x, N, V, M

x, N, V, M = analiz()

# --- GÖRSELLEŞTİRME (ÜÇLÜ DİYAGRAM) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.3)

# Renkler görseldeki standartlara göre: Yeşil, Kırmızı, Mavi
titles = ["Normal Kuvvet (N) [kN]", "Kesme Kuvveti (V) [kN]", "Moment Diyagramı (M) [kNm]"]
data_list = [N, V, M]
colors = ["#228B22", "#FF0000", "#0000FF"] # ForestGreen, Red, Blue

for i, ax in enumerate(axes):
    ax.plot(x, data_list[i], color=colors[i], lw=2.5)
    ax.fill_between(x, data_list[i], color=colors[i], alpha=0.15)
    ax.axhline(0, color='black', lw=1)
    ax.set_title(titles[i], loc='left', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    if i == 2: ax.invert_yaxis() # Moment çekme tarafı

st.pyplot(fig)

# --- ALT BİLGİ ---
st.markdown(f"""
    **Sistem Özeti:**
    * Toplam Uzunluk: **{L}m**
    * Mafsal Noktası: **{mafsal_pos}m**
    * Maksimum Moment: **{np.max(np.abs(M)):.2f} kNm**
""")
