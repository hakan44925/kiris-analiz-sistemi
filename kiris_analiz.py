import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hakan Çırak - Parçalı Gerber Analiz", layout="wide")

st.title("🏗️ Parçalı Sistem Gerber Analiz Portalı")
st.markdown("---")

# --- GİRİŞ PANELİ (YENİ NESİL) ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Sol Parça (Taşıyan)")
    L1 = st.number_input("Sol Kiriş Boyu (m)", value=4.0)
    mesnet_sol = st.selectbox("Sol Mesnet Türü", ["Ankastre", "Sabit/Hareketli"], index=0)
    q1 = st.number_input("Sol Kiriş Yayılı Yük (kN/m)", value=2.0)

with col2:
    st.header("2. Sağ Parça (Taşınan)")
    L2 = st.number_input("Sağ Kiriş Boyu (m)", value=4.0)
    mesnet_sag = st.selectbox("Sağ Mesnet Türü", ["Sabit/Hareketli", "Boş (Konsol)"], index=0)
    p2 = st.number_input("Sağ Kiriş Tekil Yük (kN)", value=4.0)
    pk2 = st.number_input("Yükün Mafsaldan Uzaklığı (m)", value=2.0)

# --- HESAPLAMA MOTORU (STATİK AYIRMA) ---
try:
    # A. SAĞ PARÇA ÇÖZÜMÜ
    # Sağ uçta mesnet varsa
    if mesnet_sag == "Sabit/Hareketli":
        # ΣM_mafsal = 0 => By * L2 - p2 * pk2 = 0
        By = (p2 * pk2) / L2
        Cy = p2 - By # Mafsal reaksiyonu
    else:
        # Konsol ise
        By = 0
        Cy = p2

    # B. SOL PARÇA ÇÖZÜMÜ
    # Sağdan gelen Cy mafsal tepkisi, sol kirişin ucuna yük olarak biner.
    if mesnet_sol == "Ankastre":
        # ΣFy = 0 => Ay = q1*L1 + Cy
        Ay = (q1 * L1) + Cy
        # ΣM_A = 0 => Ma = (q1*L1 * L1/2) + (Cy * L1)
        Ma = (q1 * L1 * (L1/2)) + (Cy * L1)
    else:
        # İki mesnetli basit kiriş gibi düşün (Örn: Sabit + Hareketli)
        Ay = (q1 * L1 * 0.5) + (Cy * 1.0) # Basit bir oranlama (geliştirilebilir)
        Ma = 0

    # --- DİYAGRAMLARIN OLUŞTURULMASI ---
    x1 = np.linspace(0, L1, 500)
    x2 = np.linspace(L1, L1 + L2, 500)
    
    # Sol Parça (0 - L1)
    V1 = Ay - (q1 * x1)
    M1 = (Ay * x1) - (q1 * x1**2 / 2) - Ma
    
    # Sağ Parça (L1 - L1+L2)
    x2_rel = x2 - L1
    V2 = Cy - (np.where(x2_rel >= pk2, p2, 0))
    M2 = (Cy * x2_rel) - (np.where(x2_rel >= pk2, p2 * (x2_rel - pk2), 0))

    # Birleştirme
    x_full = np.concatenate([x1, x2])
    V_full = np.concatenate([V1, V2])
    M_full = np.concatenate([M1, M2])

    # --- GÖRSELLEŞTİRME ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Kesme Kuvveti
    axes[0].plot(x_full, V_full, color='blue', lw=2)
    axes[0].fill_between(x_full, V_full, color='blue', alpha=0.1)
    axes[0].set_title("V - Kesme Kuvveti Diyagramı (kN)", fontweight='bold')
    
    # Moment
    axes[1].plot(x_full, M_full, color='red', lw=2)
    axes[1].fill_between(x_full, M_full, color='red', alpha=0.1)
    axes[1].set_title("M - Moment Diyagramı (kNm)", fontweight='bold')
    axes[1].invert_yaxis()

    for ax in axes:
        ax.axhline(0, color='black', lw=1)
        ax.grid(True, alpha=0.2)

    st.pyplot(fig)

    # REAKSİYONLAR
    c1, c2, c3 = st.columns(3)
    c1.metric("Ay Reaksiyonu", f"{Ay:.1f} kN")
    c2.metric("Ankastre Moment (Ma)", f"{Ma:.1f} kNm")
    c3.metric("By Reaksiyonu", f"{By:.1f} kN")

except Exception as e:
    st.error(f"Hesaplama sırasında bir hata oluştu: {e}")
