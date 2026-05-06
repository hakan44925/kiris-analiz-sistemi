import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Gerber Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.write("---")

# --- SIDEBAR (PARAMETRELER) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0)
    mafsal = st.number_input("Mafsal Konumu (m)", value=4.0)
    
    st.subheader("🔗 Mesnetler")
    st.info("Sistem: Sol uç Ankastre, Sağ uç Hareketli")
    
    st.subheader("🔴 Yükler")
    ws = st.number_input("Yayılı Yük (kN/m)", value=2.0)
    ps = st.number_input("Tekil Yük (kN)", value=4.0)
    pk = st.number_input("Tekil Yük Konumu (m)", value=6.0)

def analiz_motoru():
    try:
        # --- EL HESABI MANTIĞI (ADIM ADIM ÇÖZÜM) ---
        
        # 1. SAĞ TARAF (Mafsal ile Sağ Mesnet Arası)
        # ΣM_mafsal = 0 => By * (L - mafsal) - ps * (pk - mafsal) = 0
        By = (ps * (pk - mafsal)) / (L - mafsal)
        # ΣFy = 0 => Cy + By - ps = 0 => Cy = ps - By (Mafsal tepkisi)
        Cy = ps - By

        # 2. SOL TARAF (Ankastre ile Mafsal Arası)
        # ΣFy = 0 => Ay - (ws * mafsal) - Cy = 0
        Ay = (ws * mafsal) + Cy
        # ΣM_Ankastre = 0 => Ma - (ws * mafsal * mafsal/2) - (Cy * mafsal) = 0
        Ma = (ws * mafsal * (mafsal / 2)) + (Cy * mafsal)

        # --- DİYAGRAM HESAPLARI ---
        x = np.linspace(0, L, 1000)
        V = np.zeros_like(x)
        M = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi <= mafsal:
                # AC Kirişi (Sol Parça)
                V[i] = Ay - (ws * xi)
                # Moment Denklemi: M = Ay*x - (ws*x^2)/2 - Ma
                M[i] = (Ay * xi) - (ws * (xi**2) / 2) - Ma
            else:
                # CB Kirişi (Sağ Parça)
                # Sol taraftan gelen tüm etkiler Cy olarak mafsala aktarıldı
                V[i] = Cy - (ps if xi >= pk else 0)
                # M = Cy*(x-mafsal) - ps*(x-pk)
                M[i] = (Cy * (xi - mafsal)) - (ps * (xi - pk) if xi >= pk else 0)

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(2, 1, figsize=(11, 9))
        plt.subplots_adjust(hspace=0.4)

        # Kesme Kuvveti (V)
        axes[0].plot(x, V, color='#2563EB', lw=2.5)
        axes[0].fill_between(x, V, color='#2563EB', alpha=0.15)
        axes[0].axhline(0, color='black', lw=1.2)
        axes[0].set_title(f"V - Kesme Kuvveti (Ay: {Ay:.1f} kN, By: {By:.1f} kN)", loc='left', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Moment (M)
        axes[1].plot(x, M, color='#DC2626', lw=2.5)
        axes[1].fill_between(x, M, color='#DC2626', alpha=0.15)
        axes[1].axhline(0, color='black', lw=1.2)
        axes[1].set_title(f"M - Moment (Ma: {Ma:.1f} kNm)", loc='left', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis() # Moment çekme tarafına

        # Kritik Nokta Değerleri
        for ax, data in zip(axes, [V, M]):
            for p in [0, mafsal, pk, L]:
                idx = np.abs(x - p).argmin()
                val = data[idx]
                ax.text(p, val, f' {val:.1f}', fontweight='bold')

        st.pyplot(fig)

        # Reaksiyon Paneli
        st.subheader("📊 Hesaplanan Reaksiyon Kuvvetleri")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ay (Düşey)", f"{Ay:.1f} kN")
        c2.metric("Ma (Moment)", f"{Ma:.1f} kNm")
        c3.metric("By (Düşey)", f"{By:.1f} kN")

    except Exception as e:
        st.error(f"Sistem çözülemedi: {e}")

if __name__ == "__main__":
    analiz_motoru()
