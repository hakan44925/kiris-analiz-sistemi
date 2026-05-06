import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Gerber Analiz", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.info("Not: El hesabınızla uyumlu olması için moment diyagramı çekme tarafına (mühendislik standardı) çizilmektedir.")

# --- SIDEBAR (GİRİŞ PANELİ) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0, min_value=0.1)
    
    st.subheader("⚪ Gerber Mafsalları")
    mafsal_raw = st.text_input("Mafsal Konumları (m)", "4", help="Örn: 4")
    
    st.subheader("🔗 Mesnetler")
    m_pos_raw = st.text_input("Mesnet Konumları (m)", "0, 8")
    m_type_raw = st.text_input("Türler (1:Sabit, 2:Hark, 3:Ankastre)", "3, 2")

    st.subheader("🔴 Tekil Yükler")
    p_s_raw = st.text_input("Yük Şiddetleri (kN)", "4")
    p_k_raw = st.text_input("Yük Konumları (m)", "6")

    st.subheader("🟠 Yayılı Yükler")
    w_s_raw = st.text_input("Yayılı Yük Şiddeti (kN/m)", "2")
    w_b_raw = st.text_input("Başlangıç (m)", "0")
    w_e_raw = st.text_input("Bitiş (m)", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            if not raw.strip(): return np.array([])
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

        # Input Parsing
        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # --- REAKSİYON HESABI (MATRİS SİSTEMİ) ---
        # Bilinmeyenlerin haritalanması
        reak_map = [] # List of tuples: (mesnet_idx, type_label)
        for i, t in enumerate(m_type):
            reak_map.append((i, 'Ry'))
            if t == 3: reak_map.append((i, 'Ma'))
        
        n_reak = len(reak_map)
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Toplam Düşey Denge (ΣFy = 0)
        for j, (m_idx, r_type) in enumerate(reak_map):
            if r_type == 'Ry': A[0, j] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. Toplam Moment Dengesi (ΣM_x0 = 0)
        for j, (m_idx, r_type) in enumerate(reak_map):
            if r_type == 'Ry': A[1, j] = m_pos[m_idx]
            if r_type == 'Ma': A[1, j] = 1
        B[1] = np.sum(ps * pk) + np.sum(ws * (we - wb) * (wb + we)/2)

        # 3. Mafsal Denklemleri (ΣM_mafsal_sol = 0)
        for i, maf_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            for j, (m_idx, r_type) in enumerate(reak_map):
                if m_pos[m_idx] < maf_x:
                    if r_type == 'Ry': A[row, j] = (maf_x - m_pos[m_idx])
                    if r_type == 'Ma': A[row, j] = 1
            
            m_load_sol = np.sum(ps[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    w_end = min(we[k], maf_x)
                    m_load_sol += (ws[k] * (w_end - wb[k])) * (maf_x - (wb[k] + w_end)/2)
            B[row] = m_load_sol

        res = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI ---
        x = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0.0, 0.0
            # Reaksiyonların katkısı
            for j, (m_idx, r_type) in enumerate(reak_map):
                if xi >= m_pos[m_idx]:
                    if r_type == 'Ry':
                        v_val += res[j]
                        m_val += res[j] * (xi - m_pos[m_idx])
                    if r_type == 'Ma':
                        # Ankastre momenti doğrudan M değerini öteler
                        m_val -= res[j] # Statik işareti: Ma genellikle saat yönü tersi kabul edilir
            
            # Tekil yükler
            for j in range(len(ps)):
                if xi >= pk[j]:
                    v_val -= ps[j]
                    m_val -= ps[j] * (xi - pk[j])
            
            # Yayılı yükler
            for k in range(len(ws)):
                if xi > wb[k]:
                    act_len = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * act_len
                    m_val -= (ws[k] * act_len) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        plt.subplots_adjust(hspace=0.4)

        # Şema
        axes[0].hlines(0, 0, L, color='black', lw=4)
        for p, t in zip(m_pos, m_type):
            if t == 3: axes[0].vlines(p, -0.5, 0.5, color='black', lw=10)
            else: axes[0].plot(p, -0.2, '^', ms=15, color='gray')
        if len(mafsallar) > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)
        axes[0].set_title("Sistem Kesiti", fontweight='bold')
        axes[0].axis('off')

        # Kesme Kuvveti
        axes[1].plot(x, V, color='blue')
        axes[1].fill_between(x, V, color='blue', alpha=0.1)
        axes[1].set_title("V (Kesme Kuvveti) - kN", loc='left', fontweight='bold')
        axes[1].axhline(0, color='black', lw=1)

        # Moment
        axes[2].plot(x, M, color='red')
        axes[2].fill_between(x, M, color='red', alpha=0.1)
        axes[2].set_title("M (Eğilme Momenti) - kNm", loc='left', fontweight='bold')
        axes[2].axhline(0, color='black', lw=1)
        axes[2].invert_yaxis() # Çekme tarafı (El hesabıyla aynı yön)

        for ax in axes[1:]:
            ax.grid(True, alpha=0.3)
            # Kritik noktaları etiketle
            k_points = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk)))
            for kp in k_points:
                idx = np.abs(x - kp).argmin()
                val = V[idx] if ax == axes[1] else M[idx]
                ax.text(kp, val, f'{val:.1f}', fontsize=8, fontweight='bold', ha='center')

        st.pyplot(fig)

        # Reaksiyonları Listele
        st.subheader("📊 Hesaplanan Reaksiyonlar")
        cols = st.columns(len(reak_map))
        for i, (m_idx, r_type) in enumerate(reak_map):
            unit = "kN" if r_type == 'Ry' else "kNm"
            cols[i].metric(f"Mesnet {m_idx} ({r_type})", f"{res[i]:.2f} {unit}")

    except Exception as e:
        st.error(f"Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
