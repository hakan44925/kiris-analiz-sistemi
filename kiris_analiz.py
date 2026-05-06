import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak | Gerber Analiz", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #1E3A8A; margin-bottom: 0px;}
    .reportview-container { background: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Gerber Kirişi (Mafsallı) Analiz Sistemi")
st.write("Statikçe Belirli Sistemlerin Mafsal Yöntemiyle Analizi")
st.markdown("---")

# --- SIDEBAR GİRİŞLERİ ---
st.sidebar.header("📐 Kiriş ve Mafsal")
L = st.sidebar.number_input("Kiriş Boyu (m)", value=15.0, min_value=1.0)
mafsal_raw = st.sidebar.text_input("Mafsal Konumları (m)", "5, 10")

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 7, 15")
# Mesnet tipleri Gerber'de genellikle sabit/hareketlidir. 
# Bu sistemde 3 mesnet + 1 mafsal = İzostatik mantığı kurulur.

st.sidebar.subheader("🔴 Yüklemeler")
p_s_raw = st.sidebar.text_input("Tekil Yükler (kN)", "20")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "3")
w_s_raw = st.sidebar.text_input("Yayılı Yük (kN/m)", "5")
w_range_raw = st.sidebar.text_input("Yayılı Yük Aralığı (m-m)", "0, 15")

def analiz_motoru():
    try:
        def parse(raw):
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

        # Inputlar
        m_pos = parse(m_pos_raw)
        mafsallar = parse(mafsal_raw)
        ps = parse(p_s_raw)
        pk = parse(p_k_raw)
        ws = parse(w_s_raw)
        wr = parse(w_range_raw)

        # Çözünürlük
        pts = 1000
        x = np.linspace(0, L, pts)
        V = np.zeros(pts)
        M = np.zeros(pts)

        # --- STATİK ÇÖZÜM (MATRİS YÖNTEMİ) ---
        # Gerber sistemlerinde bilinmeyen sayısı = Mesnet Reaksiyonları
        # Denklem sayısı = Toplam Denge (2) + Mafsal Sayısı (Moment=0)
        n_reak = len(m_pos)
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Toplam Düşey Denge (ΣFy = 0)
        A[0, :] = 1
        total_load = np.sum(ps)
        if len(ws) > 0:
            total_load += ws[0] * (wr[1] - wr[0])
        B[0] = total_load

        # 2. Toplam Moment Dengesi (En sol mesnete göre ΣMa = 0)
        for i in range(n_reak):
            A[1, i] = m_pos[i] - m_pos[0]
        
        m_load = np.sum(ps * (pk - m_pos[0]))
        if len(ws) > 0:
            w_mid = (wr[0] + wr[1]) / 2
            m_load += (ws[0] * (wr[1] - wr[0])) * (w_mid - m_pos[0])
        B[1] = m_load

        # 3. Mafsal Denklemleri (Mafsalın solundaki moment ΣM_mafsal = 0)
        for i, m_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            # Reaksiyonların mafsala göre momenti (Sadece mafsalın solundakiler)
            for j in range(n_reak):
                if m_pos[j] < m_x:
                    A[row, j] = m_x - m_pos[j]
            
            # Yüklerin mafsala göre momenti (Sadece mafsalın solundakiler)
            m_load_maf = np.sum(ps[pk < m_x] * (m_x - pk[pk < m_x]))
            if len(ws) > 0 and wr[0] < m_x:
                w_end = min(wr[1], m_x)
                w_len = w_end - wr[0]
                m_load_maf += (ws[0] * w_len) * (m_x - (wr[0] + w_end)/2)
            B[row] = m_load_maf

        # Reaksiyonları Çöz
        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI ---
        for i, xi in enumerate(x):
            cv, cm = 0, 0
            # Reaksiyonlar
            for r_val, r_pos in zip(reaksiyonlar, m_pos):
                if xi >= r_pos:
                    cv += r_val
                    cm += r_val * (xi - r_pos)
            # Tekil Yükler
            for p_val, p_pos in zip(ps, pk):
                if xi >= p_pos:
                    cv -= p_val
                    cm -= p_val * (xi - p_pos)
            # Yayılı Yük
            if len(ws) > 0:
                if xi > wr[0]:
                    yük_boyu = min(xi, wr[1]) - wr[0]
                    cv -= ws[0] * yük_boyu
                    cm -= (ws[0] * yük_boyu) * (xi - (wr[0] + yük_boyu/2))
            
            V[i] = cv
            M[i] = cm

        # --- GÖRSELLEŞTİRME ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.4)

        # Kiriş Şeması
        ax1.hlines(0, 0, L, color='black', lw=3)
        ax1.scatter(m_pos, [-0.1]*len(m_pos), marker='^', s=200, color='red', label='Mesnet')
        ax1.scatter(mafsallar, [0]*len(mafsallar), marker='o', s=100, color='white', edgecolors='black', zorder=5, label='Mafsal')
        ax1.set_title("Kiriş Modeli (Mafsallar Beyaz Daire)")
        ax1.set_ylim(-0.5, 0.5)
        ax1.legend()
        ax1.axis('off')

        # Kesme Kuvveti
        ax2.plot(x, V, color='blue', lw=2)
        ax2.fill_between(x, V, color='blue', alpha=0.1)
        ax2.axhline(0, color='black', lw=1)
        ax2.set_title("V - Kesme Kuvveti Diyagramı (kN)")
        ax2.grid(True, alpha=0.3)

        # Moment
        ax3.plot(x, M, color='red', lw=2)
        ax3.fill_between(x, M, color='red', alpha=0.1)
        ax3.axhline(0, color='black', lw=1)
        ax3.set_title("M - Eğilme Momenti Diyagramı (kNm)")
        ax3.invert_yaxis() # Mühendislik standardı
        ax3.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Veri Tablosu
        st.subheader("📊 Analiz Sonuçları")
        cols = st.columns(len(m_pos))
        for i, r in enumerate(reaksiyonlar):
            cols[i].metric(f"{m_pos[i]}. m Reaksiyonu", f"{r:.2f} kN")

    except Exception as e:
        st.error(f"Sistem Kararsız veya Giriş Hatası! Lütfen mesnet/mafsal sayısını kontrol edin. (Hata: {e})")

if __name__ == "__main__":
    analiz_motoru()
