import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    h1 {margin-bottom: 0rem;}
    hr {margin-top: 1rem; margin-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR (GİRİŞ PANELİ) ---
st.sidebar.header("📐 Sistem Parametreleri")
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=12.0, min_value=0.1)

st.sidebar.subheader("⚪ Mafsallar (Gerber)")
mafsal_raw = st.sidebar.text_input("Mafsal Konumları (m)", "")

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 10")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "1, 2")

st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "")
p_a_raw = st.sidebar.text_input("Yük Açıları (Derece)", "")

st.sidebar.subheader("🔄 Tekil Momentler")
m_s_raw = st.sidebar.text_input("Moment Şiddetleri (kNm)", "")
m_k_raw = st.sidebar.text_input("Moment Konumları (m)", "")

st.sidebar.subheader("🟠 Yayılı Yükler")
w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "5")
w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "12")

def analiz_motoru():
    try:
        def parse_input(raw):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        # Input Parsing
        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        pa = parse_input(p_a_raw)
        ms_val = parse_input(m_s_raw)
        mk_pos = parse_input(m_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Çözünürlük ve Arrayler
        x = np.linspace(0, L, 2000)
        N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        
        rad = np.deg2rad(pa) if pa.size > 0 else np.array([])
        py = ps * np.sin(rad) if ps.size > 0 else ps
        px = ps * np.cos(rad) if ps.size > 0 else np.zeros_like(ps)

        # --- REAKSİYON HESABI (Matris Formu - Gerber Desteği) ---
        # Bilinmeyenler: Mesnet Reaksiyonları (R1y, R2y, ... Rn, M_ankastre)
        n_reak = len(m_pos)
        if 3 in m_type: n_reak += 1 # Ankastre momenti için +1 bilinmeyen
        
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # Denge Denklemleri
        # 1. Toplam Fy = 0
        A[0, :len(m_pos)] = 1
        total_p_y = np.sum(py)
        total_w_y = np.sum(ws * (we - wb))
        B[0] = total_p_y + total_w_y

        # 2. Toplam M_x=0 = 0
        for i in range(len(m_pos)):
            A[1, i] = m_pos[i]
        if 3 in m_type: 
            ank_idx = np.where(m_type == 3)[0][0]
            A[1, len(m_pos)] = 1 # Ankastre Moment Bilinmeyeni
        
        moment_load = np.sum(py * pk) + np.sum(ms_val)
        for i in range(len(ws)):
            moment_load += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)
        B[1] = moment_load

        # 3. Mafsal Denklemleri (Mafsalın solundaki moment = 0)
        for i, maf_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            for j in range(len(m_pos)):
                if m_pos[j] < maf_x:
                    A[row, j] = maf_x - m_pos[j]
            if 3 in m_type and m_pos[ank_idx] < maf_x:
                A[row, len(m_pos)] = 1

            m_load_maf = np.sum(py[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            m_load_maf += np.sum(ms_val[mk_pos < maf_x])
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    w_end = min(we[k], maf_x)
                    w_len = w_end - wb[k]
                    m_load_maf += (ws[k] * w_len) * (maf_x - (wb[k] + w_end)/2)
            B[row] = m_load_maf

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            # Mesnet Katkıları
            for j in range(len(m_pos)):
                if xi >= m_pos[j]:
                    v_val += reaksiyonlar[j]
                    m_val += reaksiyonlar[j] * (xi - m_pos[j])
            if 3 in m_type and xi >= m_pos[ank_idx]:
                m_val -= reaksiyonlar[len(m_pos)]
            
            # Tekil Yükler
            for j in range(len(ps)):
                if xi >= pk[j]:
                    v_val -= py[j]
                    m_val -= py[j] * (xi - pk[j])
            
            # Yayılı Yükler
            for k in range(len(ws)):
                if xi > wb[k]:
                    active_w_len = min(xi, we[k]) - wb[k]
                    centroid = wb[k] + active_w_len/2
                    v_val -= ws[k] * active_w_len
                    m_val -= (ws[k] * active_w_len) * (xi - centroid)
            
            # Momentler
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk: m_val += mv
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.5)

        # 1. ŞEMA BÖLÜMÜ
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=20, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=15, color='gray')
            if t == 3: axes[0].vlines(p, -0.6, 0.6, color='black', lw=10)
        
        # Mafsal Görseli
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)

        # Yükleme Etiketleri (Aynı ilk kod gibi)
        for i in range(len(ps)):
            axes[0].annotate(f'{ps[i]}kN', xy=(pk[i], 0), xytext=(pk[i], 1.2),
                             arrowprops=dict(facecolor='red', width=1.5, headwidth=7), 
                             ha='center', color='red', fontweight='bold')
        
        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3)
            axes[0].add_patch(rect)

        axes[0].set_ylim(-1, 2)
        axes[0].axis('off')

        # 2. DİYAGRAMLAR
        titles = ["N (Normal Kuvvet)", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        colors = ['green', 'blue', 'red']
        data_list = [N, V, M]

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data_list)):
            ax.plot(x, d, color=c, lw=2)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.set_title(t, fontsize=10, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.axhline(0, color='black', lw=1)
            if "M" in t: ax.invert_yaxis()

        st.pyplot(fig)
        st.success("Analiz Başarıyla Tamamlandı!")
        
        # Reaksiyon Tablosu
        st.subheader("📋 Hesaplanan Reaksiyonlar")
        for i in range(len(m_pos)):
            st.write(f"Mesnet {i+1} ({m_pos[i]}m): **{reaksiyonlar[i]:.2f} kN**")
        if 3 in m_type:
            st.write(f"Ankastre Momenti: **{reaksiyonlar[len(m_pos)]:.2f} kNm**")

    except Exception as e:
        st.info("Sistem henüz çözülemedi. Lütfen mesnet/mafsal sayılarını ve konumlarını kontrol edin.")

if __name__ == "__main__":
    analiz_motoru()
