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

st.title("🏗️ Profesyonel Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("📐 Sistem Parametreleri")
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=11.0, min_value=0.1)

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 8")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark)", "1, 2")

st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "20")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "11")

st.sidebar.subheader("🔄 Tekil Momentler")
m_s_raw = st.sidebar.text_input("Moment Şiddetleri (kNm)", "-150")
m_k_raw = st.sidebar.text_input("Moment Konumları (m)", "11")

st.sidebar.subheader("🟠 Yayılı Yükler")
w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "40")
w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "8")

def analiz_motoru():
    try:
        def parse_input(raw):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        m_pos = parse_input(m_pos_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        ms_val = parse_input(m_s_raw)
        mk_pos = parse_input(m_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # --- REAKSİYON HESABI ---
        w_totals = ws * (np.minimum(we, L) - np.maximum(wb, 0)) if ws.size > 0 else np.array([0])
        w_centroids = (wb + we) / 2 if wb.size > 0 else np.array([0])
        
        m1, m2 = m_pos[0], m_pos[1]
        # Statik denge: m1'e göre moment
        sum_M = np.sum(ps * (pk - m1)) + np.sum(w_totals * (w_centroids - m1)) - np.sum(ms_val)
        R2y = sum_M / (m2 - m1)
        R1y = (np.sum(ps) + np.sum(w_totals)) - R2y

        # --- HASSAS DİYAGRAM HESABI ---
        # Sıçramaları net göstermek için kritik noktaların 0.000001 öncesini ve sonrasını hesaplıyoruz
        x_points = [0, L]
        for p in m_pos: x_points.extend([p-1e-9, p, p+1e-9])
        for p in pk: x_points.extend([p-1e-9, p, p+1e-9])
        for p in mk_pos: x_points.extend([p-1e-9, p, p+1e-9])
        for p in wb: x_points.extend([p-1e-9, p, p+1e-9])
        for p in we: x_points.extend([p-1e-9, p, p+1e-9])
        
        x = np.unique(np.sort(np.concatenate((np.linspace(0, L, 1000), x_points))))
        x = x[(x >= 0) & (x <= L)] # Sınırları koru

        V = np.zeros_like(x)
        M = np.zeros_like(x)

        for i, xi in enumerate(x):
            # Kesme
            cv = 0
            if xi >= m1: cv += R1y
            if xi >= m2: cv += R2y
            for j in range(len(ps)):
                if xi >= pk[j]: cv -= ps[j]
            for k in range(len(ws)):
                active = max(0, min(xi, we[k]) - wb[k])
                cv -= ws[k] * active
            V[i] = cv

            # Moment
            cm = 0
            if xi >= m1: cm += R1y * (xi - m1)
            if xi >= m2: cm += R2y * (xi - m2)
            for j in range(len(ps)):
                if xi >= pk[j]: cm -= ps[j] * (xi - pk[j])
            for k in range(len(ws)):
                if xi > wb[k]:
                    a_w = min(xi, we[k]) - wb[k]
                    cm -= (ws[k] * a_w) * (xi - (wb[k] + a_w/2))
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk: cm += mv
            M[i] = cm

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.4)

        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p in m_pos: axes[0].plot(p, -0.2, '^', ms=15, color='gray')
        axes[0].set_ylim(-1, 2); axes[0].axis('off')

        # V Diyagramı
        axes[1].plot(x, V, color='blue', lw=2)
        axes[1].fill_between(x, V, color='blue', alpha=0.1)
        axes[1].set_title("V (Kesme Kuvveti) - kN", loc='left', fontweight='bold')
        axes[1].axhline(0, color='black', lw=1)

        # M Diyagramı
        axes[2].plot(x, M, color='red', lw=2)
        axes[2].fill_between(x, M, color='red', alpha=0.1)
        axes[2].set_title("M (Eğilme Momenti) - kNm", loc='left', fontweight='bold')
        axes[2].axhline(0, color='black', lw=1)
        axes[2].invert_yaxis()

        # Etiketler (Sadece tam metrelerde ve uçlarda)
        label_points = np.unique(np.concatenate(([0, L], m_pos, pk, mk_pos)))
        for lp in label_points:
            idx = np.searchsorted(x, lp)
            if idx < len(x):
                mv, mm = V[idx], M[idx]
                axes[1].text(lp, mv, f'{round(mv,1)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                axes[2].text(lp, mm, f'{round(mm,1)}', ha='center', va='top' if mm > 0 else 'bottom', fontsize=9, fontweight='bold')

        st.pyplot(fig)
        st.success(f"Analiz Tamamlandı! R1: {R1y:.1f}kN, R2: {R2y:.1f}kN")

    except Exception as e:
        st.info("Hesaplanıyor...")

if __name__ == "__main__":
    analiz_motoru()
