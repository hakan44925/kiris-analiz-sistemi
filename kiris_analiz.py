import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Gerber Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #1E3A8A; text-align: center;}
    .metric-box {background-color: #f0f2f6; padding: 10px; border-radius: 10px; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.write("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0, step=1.0)
    mafsal = st.number_input("Gerber Mafsal Konumu (m)", value=4.0, step=0.5)
    
    st.subheader("🔗 Mesnetler")
    st.info("A Noktası (0m): Ankastre\nB Noktası (8m): Hareketli")
    
    st.subheader("🔴 Yükler")
    q = st.number_input("Yayılı Yük (kN/m) - [0-4m arası]", value=2.0)
    P = st.number_input("Tekil Yük (kN)", value=4.0)
    Pk = st.number_input("Tekil Yük Konumu (m)", value=6.0)

def analiz_motoru():
    try:
        # --- STATİK HESAPLAMA (EL HESABI MANTIĞI) ---
        # 1. Mafsalın Sağındaki Sistem (CB Kirişi: 4m - 8m)
        # ΣM_mafsal = 0 => By * (8 - 4) - P * (6 - 4) = 0
        By = (P * (Pk - mafsal)) / (L - mafsal)
        Cy = P - By # Mafsal reaksiyonu (Düşey denge)

        # 2. Mafsalın Solundaki Sistem (AC Kirişi: 0m - 4m)
        # ΣFy = 0 => Ay - (q * 4) - Cy = 0
        Ay = (q * mafsal) + Cy
        # ΣM_A = 0 => Ma - (q * 4 * 2) - (Cy * 4) = 0
        Ma = (q * mafsal * (mafsal/2)) + (Cy * mafsal)

        # --- DİYAGRAM VERİLERİ ---
        x = np.linspace(0, L, 1000)
        V = np.zeros_like(x)
        M = np.zeros_like(x)

        for i, xi in enumerate(x):
            if xi <= mafsal:
                # 0 - 4m Arası (Sol parça)
                V[i] = Ay - (q * xi)
                # Moment Denklemi: M(x) = Ay*x - q*x²/2 - Ma
                M[i] = (Ay * xi) - (q * (xi**2) / 2) - Ma
            else:
                # 4 - 8m Arası (Sağ parça)
                # Buradan itibaren reaksiyonlar ve yükler değişir
                V[i] = Ay - (q * mafsal) - (P if xi >= Pk else 0)
                # Moment: M(x) = Ay*x - q*mafsal*(x - mafsal/2) - P*(x - Pk) - Ma
                m_part1 = (Ay * xi) - Ma
                m_part2 = (q * mafsal) * (xi - (mafsal / 2))
                m_part3 = P * (xi - Pk) if xi >= Pk else 0
                M[i] = m_part1 - m_part2 - m_part3

        # --- GÖRSELLEŞTİRME ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
        plt.subplots_adjust(hspace=0.4)

        # Kesme Kuvveti (V)
        ax1.plot(x, V, color='#2563EB', lw=2.5)
        ax1.fill_between(x, V, color='#2563EB', alpha=0.15)
        ax1.axhline(0, color='black', lw=1.2)
        ax1.set_title("V - Kesme Kuvveti Diyagramı (kN)", loc='left', fontweight='bold', color='#1E3A8A')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Moment (M)
        ax2.plot(x, M, color='#DC2626', lw=2.5)
        ax2.fill_between(x, M, color='#DC2626', alpha=0.15)
        ax2.axhline(0, color='black', lw=1.2)
        ax2.set_title("M - Eğilme Momenti Diyagramı (kNm)", loc='left', fontweight='bold', color='#1E3A8A')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.invert_yaxis() # Moment çekme tarafına (El hesabındaki gibi alt tarafa pozitif)

        # Kritik Noktaları İşaretleme
        kritik_noktalar = [0, mafsal, Pk, L]
        for ax, data in zip([ax1, ax2], [V, M]):
            for kn in kritik_noktalar:
                idx = np.abs(x - kn).argmin()
                val = data[idx]
                ax.plot(kn, val, 'o', color='black', ms=5)
                ax.text(kn, val, f' {val:.1f}', fontweight='bold', va='bottom' if val > 0 else 'top')

        st.pyplot(fig)

        # --- SONUÇ PANELİ ---
        st.markdown("### 📊 Hesaplanan Reaksiyon Kuvvetleri")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Ay (Düşey)", f"{Ay:.1f} kN")
        with c2: st.metric("Ma (Ankastre)", f"{Ma:.1f} kNm")
        with c3: st.metric("By (Düşey)", f"{By:.1f} kN")
        with c4: st.metric("Cy (Mafsal Etkisi)", f"{Cy:.1f} kN")

    except Exception as e:
        st.error(f"Hesaplama hatası: {e}")

if __name__ == "__main__":
    analiz_motoru()
