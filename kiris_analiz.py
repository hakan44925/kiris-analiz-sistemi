import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR (PARAMETRELER) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0, min_value=0.1)
    mafsal_raw = st.text_input("Mafsal Konumları (m)", "4")
    m_pos_raw = st.text_input("Mesnet Konumları", "0, 8")
    m_type_raw = st.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "3, 2")
    p_s_raw = st.text_input("Yük Şiddetleri (kN)", "4")
    p_k_raw = st.text_input("Yük Konumları (m)", "6")
    w_s_raw = st.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
    w_b_raw = st.text_input("Başlangıç Metreleri", "0")
    w_e_raw = st.text_input("Bitiş Metreleri", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            if not raw.strip(): return np.array([])
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

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
            reak_defs.append({'pos': m_pos[i], 'type': 'Ry', 'id': i})
            if t == 3:
                reak_defs.append({'pos': m_pos[i], 'type': 'Ma', 'id': i})
        
        n_reak = len(reak_defs)
        if n_reak < 2:
            st.warning("Yetersiz mesnet tanımı.")
            return

        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Denge Denklemi (ΣFy = 0)
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[0, j] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. Moment Dengesi (ΣM_0 = 0)
        for j, r in enumerate(reak_defs):
            if r['type'] == 'Ry': A[1, j] = r['pos']
            if r['type'] == 'Ma': A[1, j] = 1
        B[1] = np.sum(ps * pk) + np.sum(ws * (we - wb) * (wb + we)/2)

        # 3. Gerber Mafsal Denklemleri
        for i, mx in enumerate(mafsallar):
            row = i + 2
            if row >= n_reak: break
            for j, r in enumerate(reak_defs):
                if r['pos'] <= mx:
                    if r['type'] == 'Ry': A[row, j] = (mx - r['pos'])
                    if r['type'] == 'Ma': A[row, j] = 1
            
            # Mafsalın solundaki yüklerin momenti
            m_l = np.sum(ps[pk <= mx] * (mx - pk[pk <= mx]))
            for k in range(len(ws)):
                if wb[k] < mx:
                    ex = min(we[k], mx)
                    m_l += (ws[k] * (ex - wb[k])) * (mx - (wb[k] + ex)/2)
            B[row] = m_l

        # Reaksiyonları Çöz
        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI ---
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
                        cm -= reaksiyonlar[j] # El hesabı ile işaret uyumu
            
            for j in range(len(ps)):
                if xi >= pk[j]:
                    cv -= ps[j]
                    cm -= ps[j] * (xi - pk[j])
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    len_w = min(xi, we[k]) - wb[k]
                    cv -= ws[k] * len_w
                    cm -= (ws[k] * len_w) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = cv, cm

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 11))
        plt.subplots_adjust(hspace=0.5)

        axes[0].hlines(0, 0, L, color='black', lw=4)
        for r in reak_defs:
            if r['type'] == 'Ry':
                if m_type[r['id']] == 3: axes[0].vlines(r['pos'], -0.4, 0.4, lw=6)
                else: axes[0].plot(r['pos'], -0.2, '^', ms=12, color='gray')
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), c='white', edgecolors='black', s=80, zorder=5)
        axes[0].set_title("Sistem Şeması", fontweight='bold')
        axes[0].axis('off')

        for ax, data, title, color in zip(axes[1:], [V, M], ["V (Kesme) - kN", "M (Moment) - kNm"], ["#2563EB", "#DC2626"]):
            ax.plot(x, data, color=color, lw=2)
            ax.fill_between(x, data, color=color, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            if "Moment" in title: ax.invert_yaxis()
            
            # Değer etiketleri
            kp = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk)))
            for p in kp:
                val = data[np.abs(x - p).argmin()]
                ax.text(p, val, f'{val:.1f}', fontsize=8, fontweight='bold')

        st.pyplot(fig)

        # Tablo
        st.subheader("📊 Tepki Kuvvetleri")
        cols = st.columns(len(reak_defs))
        for i, r in enumerate(reak_defs):
            cols[i].metric(f"{r['pos']}m - {r['type']}", f"{reaksiyonlar[i]:.1f}")

    except Exception as e:
        st.error(f"Matematiksel hata veya yetersiz veri: {e}")

if __name__ == "__main__":
    analiz_motoru()
