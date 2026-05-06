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

# --- SIDEBAR (MANUEL GİRİŞLER GERİ GELDİ) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0)

    st.subheader("⚪ Mafsallar (Gerber)")
    mafsal_raw = st.text_input("Mafsal Konumları (m)", "4")

    st.subheader("🔗 Mesnetler")
    m_pos_raw = st.text_input("Mesnet Konumları", "0, 8")
    m_type_raw = st.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "3, 2")

    st.subheader("🔴 Tekil Yükler")
    p_s_raw = st.text_input("Yük Şiddetleri (kN)", "4")
    p_k_raw = st.text_input("Yük Konumları (m)", "6")

    st.subheader("🟠 Yayılı Yükler")
    w_s_raw = st.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
    w_b_raw = st.text_input("Başlangıç Metreleri", "0")
    w_e_raw = st.text_input("Bitiş Metreleri", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

        # Girdileri Çek
        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Reaksiyon Tanımları
        reak_defs = []
        for i, t in enumerate(m_type):
            reak_defs.append({'pos': m_pos[i], 'type': 'Ry'})
            if t == 3: reak_defs.append({'pos': m_pos[i], 'type': 'Ma'})
        
        n_reak = len(reak_defs)
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # ΣFy = 0
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[0, j] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # ΣM_0 = 0
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[1, j] = r['pos']
            if r['type'] == 'Ma': A[1, j] = 1
        B[1] = np.sum(ps * pk) + np.sum(ws * (we - wb) * (wb + we)/2)

        # Mafsal Denklemleri
        for i, mx in enumerate(mafsallar):
            row = i + 2
            if row >= n_reak: break
            for j, r in enumerate(reak_defs):
                if r['pos'] <= mx:
                    if r['type'] == 'Ry': A[row, j] = (mx - r['pos'])
                    if r['type'] == 'Ma': A[row, j] = 1
            
            B[row] = np.sum(ps[pk <= mx] * (mx - pk[pk <= mx]))
            for k in range(len(ws)):
                if wb[k] < mx:
                    ex = min(we[k], mx)
                    B[row] += (ws[k] * (ex - wb[k])) * (mx - (wb[k] + ex)/2)

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        x = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            cv, cm = 0, 0
            for j, r in enumerate(reak_defs):
                if xi >= r['pos']:
                    if r['type'] == 'Ry':
                        cv += reaksiyonlar[j]
                        cm += reaksiyonlar[j] * (xi - r['pos'])
                    if r['type'] == 'Ma':
                        # El hesabı standardı: Ankastre momenti başlangıçta çıkarılır
                        cm -= reaksiyonlar[j]
            
            cv -= np.sum(ps[pk <= xi])
            cm -= np.sum(ps[pk <= xi] * (xi - pk[pk <= xi]))
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    leff = min(xi, we[k]) - wb[k]
                    cv -= ws[k] * leff
                    cm -= (ws[k] * leff) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = cv, cm

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 11))
        plt.subplots_adjust(hspace=0.5)

        axes[0].hlines(0, 0, L, color='black', lw=4)
        for i, r in enumerate(reak_defs):
            if r['type'] == 'Ry':
                axes[0].plot(r['pos'], -0.2, '^', ms=12, color='gray')
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), c='white', edgecolors='black', s=80, zorder=5)
        axes[0].axis('off')

        for ax, data, title, color in zip(axes[1:], [V, M], ["V (Kesme) - kN", "M (Moment) - kNm"], ["blue", "red"]):
            ax.plot(x, data, color=color, lw=2)
            ax.fill_between(x, data, color=color, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            if "Moment" in title: ax.invert_yaxis()
            
            kp = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk)))
            for p in kp:
                v = data[np.abs(x - p).argmin()]
                ax.text(p, v, f'{v:.1f}', fontsize=8, fontweight='bold')

        st.pyplot(fig)

        # Reaksiyon Tablosu
        st.subheader("📊 Reaksiyon Sonuçları")
        cols = st.columns(len(reak_defs))
        for i, r in enumerate(reak_defs):
            cols[i].metric(f"{r['pos']}m - {r['type']}", f"{reaksiyonlar[i]:.2f}")

    except Exception as e:
        st.error(f"Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
