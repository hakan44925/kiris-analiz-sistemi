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

# --- SIDEBAR (GİRİŞ PANELİ) ---
st.sidebar.header("📐 Sistem Parametreleri")
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=11.0, min_value=0.1)

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 8")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "1, 2")

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
        m_type = parse_input(m_type_raw).astype(int)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        ms_val = parse_input(m_s_raw)
        mk_pos = parse_input(m_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Çözünürlük
        num_points = 5000
        x = np.linspace(0, L, num_points)
        V = np.zeros(num_points)
        M = np.zeros(num_points)

        # --- REAKSİYON HESABI ---
        w_totals = ws * (np.minimum(we, L) - np.maximum(wb, 0)) if ws.size > 0 else np.array([0])
        w_centroids = (wb + we) / 2 if wb.size > 0 else np.array([0])
        
        # m1'e göre moment dengesi: Sum(M_m1) = 0
        m1, m2 = m_pos[0], m_pos[1]
        load_moments = np.sum(ps * (pk - m1)) + np.sum(w_totals * (w_centroids - m1)) - np.sum(ms_val)
        R2y = load_moments / (m2 - m1)
        R1y = (np.sum(ps) + np.sum(w_totals)) - R2y

        # --- ANALİZ ---
        for i in range(num_points):
            xi = x[i]
            # Kesme (V)
            cv = 0
            if xi >= m1: cv += R1y
            if xi >= m2: cv += R2y
            for j in range(len(ps)):
                if xi >= pk[j]: cv -= ps[j]
            for k in range(len(ws)):
                active_w = max(0, min(xi, we[k]) - wb[k])
                cv -= ws[k] * active_w
            V[i] = cv

            # Moment (M)
            cm = 0
            if xi >= m1: cm += R1y * (xi - m1)
            if xi >= m2: cm += R2y * (xi - m2)
            for j in range(len(ps)):
                if xi >= pk[j]: cm -= ps[j] * (xi - pk[j])
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_len = min(xi, we[k]) - wb[k]
                    cm -= (ws[k] * w_len) * (xi - (wb[k] + w_len/2))
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk: cm += mv
            M[i] = cm

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.4)

        # 1. ŞEMA
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            axes[0].plot(p, -0.2, '^' if t==1 else 'o', ms=18, color='gray')
        for i in range(len(ps)):
            axes[0].annotate(f'{ps[i]}kN', xy=(pk[i], 0), xytext=(pk[i], 1.2), arrowprops=dict(facecolor='red', width=2), ha='center', color='red', fontweight='bold')
        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3)
            axes[0].add_patch(rect)
            axes[0].text((wb[k]+we[k])/2, 0.7, f'{ws[k]}kN/m', ha='center', fontweight='bold', color='darkorange')
        for mv, mk in zip(ms_val, mk_pos):
            axes[0].plot(mk, 0.3, 'o', mfc='none', mec='purple', ms=15, mew=2)
            axes[0].text(mk, 0.5, f'{mv}kNm', ha='center', color='purple', fontweight='bold')
        axes[0].set_ylim(-1, 2); axes[0].axis('off')

        # 2. V DİYAGRAMI
        axes[1].step(x, V, where='post', color='blue', lw=2.5) # Dikey geçişler için .step kullanıldı
        axes[1].fill_between(x, V, step='post', color='blue', alpha=0.1)
        axes[1].set_title("V (Kesme Kuvveti) - kN", loc='left', fontweight='bold')
        axes[1].axhline(0, color='black', lw=1); axes[1].grid(True, alpha=0.2)

        # 3. M DİYAGRAMI (Dikey Kapanış Düzeltmesi)
        # Sıçramanın net görünmesi için uç noktaya dikey bir çizgi ekliyoruz
        x_plot = np.append(x, [L])
        m_plot = np.append(M, [0]) # En sonu sıfıra zorla
        
        axes[2].plot(x_plot, m_plot, color='red', lw=2.5)
        axes[2].fill_between(x_plot, m_plot, color='red', alpha=0.1)
        axes[2].set_title("M (Eğilme Momenti) - kNm", loc='left', fontweight='bold')
        axes[2].axhline(0, color='black', lw=1); axes[2].grid(True, alpha=0.2)
        axes[2].invert_yaxis()

        # Kritik Nokta Etiketleri
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, pk, mk_pos)))
        for kx in kritik_x:
            idx = np.argmin(np.abs(x - kx))
            v_val, m_val = V[idx], M[idx]
            if abs(v_val) > 0.1: axes[1].text(kx, v_val, f'{round(v_val,1)}', ha='center', fontweight='bold', fontsize=8)
            if abs(m_val) > 0.1: axes[2].text(kx, m_val, f'{round(m_val,1)}', ha='center', fontweight='bold', fontsize=8)

        st.pyplot(fig)
        st.success(f"Analiz Tamamlandı! R1: {R1y:.1f}kN, R2: {R2y:.1f}kN")

    except Exception as e:
        st.info("Parametreleri kontrol ediniz...")

if __name__ == "__main__":
    analiz_motoru()
