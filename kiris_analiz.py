import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    h1 {margin-bottom: 0rem; color: #1E3A8A;}
    hr {margin-top: 1rem; margin-bottom: 1rem;}
    .stNumberInput, .stTextInput {margin-bottom: -10px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.caption("Statik Analiz ve Kesit Tesir Diyagramları Hesaplayıcı")
st.markdown("---")

# --- SIDEBAR (GİRİŞ PANELİ) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=12.0, min_value=0.1, step=1.0)
    
    st.subheader("⚪ Gerber Mafsalları")
    mafsal_raw = st.text_input("Mafsal Konumları (m)", "8", help="Virgülle ayırın (Örn: 4, 8)")
    
    st.subheader("🔗 Mesnetler")
    m_pos_raw = st.text_input("Mesnet Konumları (m)", "0, 6, 12")
    m_type_raw = st.text_input("Türler (1:Sabit, 2:Hareketli, 3:Ankastre)", "1, 2, 2")

    st.subheader("🔴 Dış Yükler")
    p_s_raw = st.text_input("Tekil Yükler (kN)", "10")
    p_k_raw = st.text_input("Konumları (m)", "4")
    p_a_raw = st.text_input("Açıları (Derece)", "90")

    st.subheader("🟠 Yayılı Yükler")
    w_s_raw = st.text_input("Şiddeti (kN/m)", "2")
    w_b_raw = st.text_input("Başlangıç (m)", "0")
    w_e_raw = st.text_input("Bitiş (m)", "12")

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
        pa = parse_input(p_a_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Açı kontrolleri ve bileşenler
        if len(pa) == 0 and len(ps) > 0: pa = np.full_like(ps, 90.0)
        rad = np.deg2rad(pa)
        py = ps * np.sin(rad)
        px = ps * np.cos(rad)

        # Kritik Noktalar
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk, wb, we)))
        kritik_x = np.sort(kritik_x)

        # --- REAKSİYON HESABI (Statik Denklem Sistemi) ---
        # n_reak: Toplam bilinmeyen sayısı
        reak_labels = []
        for i, t in enumerate(m_type):
            if t == 3: reak_labels.extend([f"R{i}y", f"M{i}"])
            else: reak_labels.append(f"R{i}y")
        
        n_reak = len(reak_labels)
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Toplam Düşey Denge (ΣFy = 0)
        idx_col = 0
        for i, t in enumerate(m_type):
            A[0, idx_col] = 1 # Düşey reaksiyon
            idx_col += (2 if t == 3 else 1)
        B[0] = np.sum(py) + np.sum(ws * (we - wb))

        # 2. Toplam Moment Dengesi (ΣM_0 = 0)
        idx_col = 0
        for i, t in enumerate(m_type):
            A[1, idx_col] = m_pos[i] # R*d
            if t == 3: A[1, idx_col + 1] = 1 # Ankastre momenti
            idx_col += (2 if t == 3 else 1)
        
        m_load = np.sum(py * pk) + np.sum(ws * (we - wb) * (wb + we)/2)
        B[1] = m_load

        # 3. Gerber Mafsal Denklemleri (ΣM_mafsal_sol = 0)
        for i, m_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            idx_col = 0
            for j, t in enumerate(m_type):
                if m_pos[j] < m_x:
                    A[row, idx_col] = (m_x - m_pos[j])
                    if t == 3: A[row, idx_col + 1] = 1
                idx_col += (2 if t == 3 else 1)
            
            # Mafsalın solundaki yükler
            m_l = np.sum(py[pk < m_x] * (m_x - pk[pk < m_x]))
            for k in range(len(ws)):
                if wb[k] < m_x:
                    w_end = min(we[k], m_x)
                    m_l += (ws[k] * (w_end - wb[k])) * (m_x - (wb[k] + w_end)/2)
            B[row] = m_l

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        x = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0.0, 0.0
            # Reaksiyon etkileri
            idx_reak = 0
            for j, t in enumerate(m_type):
                if xi >= m_pos[j]:
                    v_val += reaksiyonlar[idx_reak]
                    m_val += reaksiyonlar[idx_reak] * (xi - m_pos[j])
                    if t == 3: m_val += reaksiyonlar[idx_reak+1]
                idx_reak += (2 if t == 3 else 1)
            
            # Yük etkileri
            for j in range(len(py)):
                if xi >= pk[j]:
                    v_val -= py[j]
                    m_val -= py[j] * (xi - pk[j])
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    active_len = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * active_len
                    m_val -= (ws[k] * active_len) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.4)

        # 1. SİSTEM ŞEMASI
        ax0 = axes[0]
        ax0.hlines(0, 0, L, color='black', lw=4)
        for p, t in zip(m_pos, m_type):
            if t == 1: ax0.plot(p, -0.15, '^', ms=15, color='#2563EB')
            if t == 2: ax0.plot(p, -0.15, 'o', ms=12, color='#2563EB', markerfacecolor='white')
            if t == 3: ax0.vlines(p, -0.5, 0.5, color='black', lw=8)
        
        if len(mafsallar) > 0:
            ax0.scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=80, zorder=5)
        
        for i in range(len(py)):
            ax0.annotate('', xy=(pk[i], 0), xytext=(pk[i], 0.8), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax0.text(pk[i], 0.9, f'{py[i]}kN', ha='center', color='red', fontweight='bold')

        ax0.set_title("Yapısal Sistem Şeması", fontweight='bold')
        ax0.set_xlim(-0.5, L+0.5)
        ax0.set_ylim(-1, 1.5)
        ax0.axis('off')

        # 2. KESME KUVVETİ (V)
        axes[1].plot(x, V, color='blue', lw=2)
        axes[1].fill_between(x, V, color='blue', alpha=0.1)
        axes[1].set_title("V - Kesme Kuvveti Diyagramı (kN)", loc='left', fontweight='bold')

        # 3. MOMENT (M)
        axes[2].plot(x, M, color='red', lw=2)
        axes[2].fill_between(x, M, color='red', alpha=0.1)
        axes[2].set_title("M - Eğilme Momenti Diyagramı (kNm)", loc='left', fontweight='bold')
        axes[2].invert_yaxis() # Moment çekme tarafına çizilir

        for ax in axes[1:]:
            ax.axhline(0, color='black', lw=1.2)
            ax.grid(True, linestyle='--', alpha=0.5)
            # Kritik değerleri yazdır
            for kx in kritik_x:
                idx = np.abs(x - kx).argmin()
                val = V[idx] if ax == axes[1] else M[idx]
                ax.text(kx, val, f' {val:.1f}', fontsize=9, fontweight='bold')

        st.pyplot(fig)
        
        # Sonuç Tablosu
        cols = st.columns(len(reak_labels))
        for i, label in enumerate(reak_labels):
            cols[i].metric(label, f"{reaksiyonlar[i]:.2f} kN/kNm")

    except Exception as e:
        st.error(f"Sistem Çözülemedi: {e}")
        st.info("İpucu: Mesnet ve mafsal sayısının izostatik dengeyi sağladığından emin olun.")

if __name__ == "__main__":
    analiz_motoru()
