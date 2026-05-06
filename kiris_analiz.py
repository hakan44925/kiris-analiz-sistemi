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
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR (TÜM AYARLAR GERİ GELDİ) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=8.0, min_value=0.1)

    st.subheader("⚪ Mafsallar (Gerber)")
    mafsal_raw = st.sidebar.text_input("Mafsal Konumları (m)", "4")

    st.subheader("🔗 Mesnetler")
    m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 8")
    m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "3, 2")

    st.subheader("🔴 Tekil Yükler")
    p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "4")
    p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "6")
    p_a_raw = st.sidebar.text_input("Yük Açıları (Derece)", "90")

    st.subheader("🟠 Yayılı Yükler")
    w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
    w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
    w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        # Girdileri Çek
        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        pa = parse_input(p_a_raw)
        if pa.size == 0 and ps.size > 0: pa = np.full_like(ps, 90.0)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Bileşenler
        py = ps * np.sin(np.deg2rad(pa)) if ps.size > 0 else np.array([])

        # --- REAKSİYON ÇÖZÜCÜ (GENEL MATRİS) ---
        reak_defs = []
        for i, t in enumerate(m_type):
            reak_defs.append({'idx': i, 'pos': m_pos[i], 'type': 'Ry'})
            if t == 3: reak_defs.append({'idx': i, 'pos': m_pos[i], 'type': 'Ma'})
        
        n_reak = len(reak_defs)
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # ΣFy = 0
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[0, j] = 1
        B[0] = np.sum(py) + np.sum(ws * (we - wb))

        # ΣM_0 = 0
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[1, j] = r['pos']
            if r['type'] == 'Ma': A[1, j] = 1
        B[1] = np.sum(py * pk) + np.sum(ws * (we - wb) * (wb + we)/2)

        # Mafsal Denklemleri
        for i, m_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            for j, r in enumerate(reak_defs):
                if r['pos'] <= m_x:
                    if r['type'] == 'Ry': A[row, j] = (m_x - r['pos'])
                    if r['type'] == 'Ma': A[row, j] = 1
            
            m_load = np.sum(py[pk < m_x] * (m_x - pk[pk < m_x]))
            for k in range(len(ws)):
                if wb[k] < m_x:
                    e_x = min(we[k], m_x)
                    m_load += (ws[k] * (e_x - wb[k])) * (m_x - (wb[k] + e_x)/2)
            B[row] = m_load

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        x = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            for j, r in enumerate(reak_defs):
                if xi >= r['pos']:
                    if r['type'] == 'Ry':
                        v_val += reaksiyonlar[j]
                        m_val += reaksiyonlar[j] * (xi - r['pos'])
                    if r['type'] == 'Ma':
                        m_val -= reaksiyonlar[j] # Ankastre moment etkisi
            
            for j in range(len(py)):
                if xi >= pk[j]:
                    v_val -= py[j]
                    m_val -= py[j] * (xi - pk[j])
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    L_eff = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * L_eff
                    m_val -= (ws[k] * L_eff) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 12))
        plt.subplots_adjust(hspace=0.5)

        # Şema
        axes[0].hlines(0, 0, L, color='black', lw=5)
        for r in reak_defs:
            if r['type'] == 'Ry':
                t = m_type[r['idx']]
                if t == 3: axes[0].vlines(r['pos'], -0.5, 0.5, lw=8)
                else: axes[0].plot(r['pos'], -0.2, '^', ms=15)
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)
        axes[0].set_title("Yapısal Şema")
        axes[0].axis('off')

        # V ve M Diyagramları
        for ax, data, title, color in zip(axes[1:], [V, M], ["V (Kesme) - kN", "M (Moment) - kNm"], ["blue", "red"]):
            ax.plot(x, data, color=color, lw=2)
            ax.fill_between(x, data, color=color, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            if "Moment" in title: ax.invert_yaxis()
            
            # Kritik Etiketler
            kp = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk)))
            for p in kp:
                val = data[np.abs(x - p).argmin()]
                ax.text(p, val, f'{val:.1f}', fontsize=8, fontweight='bold')

        st.pyplot(fig)
        st.success("Analiz Tamamlandı!")

        # Reaksiyon Tablosu
        cols = st.columns(len(reak_defs))
        for i, r in enumerate(reak_defs):
            label = f"Mesnet {r['pos']}m ({r['type']})"
            cols[i].metric(label, f"{reaksiyonlar[i]:.2f}")

    except Exception as e:
        st.error(f"Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
