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
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=12.0, min_value=0.1)

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 10")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "1, 2")

st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "")
p_a_raw = st.sidebar.text_input("Yük Açıları (Derece)", "90")

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

        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        pa = parse_input(p_a_raw)
        ms_val = parse_input(m_s_raw)
        mk_pos = parse_input(m_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        x = np.linspace(0, L, 5000)
        N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        
        rad = np.deg2rad(pa) if pa.size > 0 else np.array([])
        py = ps * np.sin(rad) if ps.size > 0 else ps 
        px = ps * np.cos(rad) if ps.size > 0 else np.zeros_like(ps)

        # --- REAKSİYON HESABI ---
        w_totals = ws * (np.minimum(we, L) - np.maximum(wb, 0)) if ws.size > 0 else np.array([0])
        w_centroids = (wb + we) / 2 if wb.size > 0 else np.array([0])
        
        if 3 in m_type: 
            f_idx = np.where(m_type == 3)[0][0]
            fixed_x = m_pos[f_idx]
            R1y = np.sum(py) + np.sum(w_totals)
            R1x = -np.sum(px)
            # Moment dengesi (Ankastre mesnet momenti)
            M_reak = np.sum(py * (pk - fixed_x)) + np.sum(w_totals * (w_centroids - fixed_x)) - np.sum(ms_val)
            R2y = 0
        else: 
            m1, m2 = m_pos[0], m_pos[1]
            # m1'e göre moment alarak R2y bulma
            sum_M_m1 = np.sum(py * (pk - m1)) + np.sum(w_totals * (w_centroids - m1)) - np.sum(ms_val)
            R2y = sum_M_m1 / (m2 - m1)
            R1y = (np.sum(py) + np.sum(w_totals)) - R2y
            R1x = -np.sum(px)
            M_reak = 0

        # --- DİYAGRAM HESAPLARI ---
        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            # Mesnetler
            if xi >= m_pos[0]: 
                v_val += R1y
                m_val += R1y * (xi - m_pos[0])
            if len(m_pos) > 1 and xi >= m_pos[1]: 
                v_val += R2y
                m_val += R2y * (xi - m_pos[1])
            if 3 in m_type and xi >= m_pos[0]: 
                m_val -= M_reak
            
            # Yükler
            for j in range(len(ps)):
                if xi >= pk[j]:
                    v_val -= py[j]
                    m_val -= py[j] * (xi - pk[j])
            for k in range(len(ws)):
                if xi > wb[k]:
                    a_w = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * a_w
                    m_val -= (ws[k] * a_w) * (xi - (wb[k] + a_w/2))
            
            # TEKİL MOMENTLER (Burada m_val'e eklenir)
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk:
                    m_val += mv
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.5)

        # 1. ŞEMA
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=20, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=15, color='gray')
            if t == 3: axes[0].vlines(p, -0.6, 0.6, color='black', lw=10)
        
        for i in range(len(ps)):
            axes[0].annotate(f'{ps[i]}kN', xy=(pk[i], 0), xytext=(pk[i], 1.2),
                             arrowprops=dict(facecolor='red', width=1.5, headwidth=7), ha='center', color='red', fontweight='bold')
        
        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3)
            axes[0].add_patch(rect)
            axes[0].text((wb[k]+we[k])/2, 0.7, f'{ws[k]}kN/m', ha='center', color='darkorange', fontweight='bold')

        for mv, mk in zip(ms_val, mk_pos):
            axes[0].plot(mk, 0.3, 'o', mfc='none', mec='purple', ms=15, mew=2)
            axes[0].text(mk, 0.5, f'{mv}kNm', ha='center', color='purple', fontweight='bold')

        axes[0].set_ylim(-1, 2)
        axes[0].axis('off')

        # 2. DİYAGRAMLAR
        titles = ["N (Normal Kuvvet) - kN", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        colors = ['green', 'blue', 'red']
        data_list = [N, V, M]
        # Kritik noktaları bul (Etiketler için)
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, pk, wb, we, mk_pos)))

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data_list)):
            ax.plot(x, d, color=c, lw=2)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.set_title(t, fontsize=10, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.axhline(0, color='black', lw=1)
            if "M" in t: 
                ax.invert_yaxis() # Mühendislik standardı: Moment çekme tarafına

            # Etiketleme
            for kx in kritik_x:
                idx = np.argmin(np.abs(x - kx))
                val = d[idx]
                if abs(val) > 0.01:
                    ax.text(kx, val, f'{round(val, 1)}', ha='center', va='bottom' if val>0 else 'top', fontsize=9, fontweight='bold')

        st.pyplot(fig)
        st.success(f"Analiz Tamamlandı! R1y: {R1y:.1f}kN, R2y: {R2y:.1f}kN | Ankastre Momenti: {M_reak:.1f}kNm")

    except Exception as e:
        st.info("Sistem girişleri bekleniyor veya bir hata oluştu. Lütfen parametreleri kontrol edin.")

if __name__ == "__main__":
    analiz_motoru()
