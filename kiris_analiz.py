import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Web Sayfası Yapılandırması
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.title("🏗️ Profesyonel Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR (GİRİŞ PANELİ) ---
st.sidebar.header("📐 Sistem Parametreleri")

L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=12.0, min_value=0.1)

# Mesnetler
st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 10")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "1, 2")

# Tekil Yükler (Boş bırakılabilir)
st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "")
p_a_raw = st.sidebar.text_input("Yük Açıları (Derece)", "")

# Tekil Momentler (YENİ EKLENDİ)
st.sidebar.subheader("🔄 Tekil Momentler")
m_s_raw = st.sidebar.text_input("Moment Şiddetleri (kNm) (+: Saat Yönü Ters)", "")
m_k_raw = st.sidebar.text_input("Moment Konumları (m)", "")

# Yayılı Yükler (Boş bırakılabilir)
st.sidebar.subheader("🟠 Yayılı Yükler")
w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "5")
w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "12")

def analiz_motoru():
    try:
        # --- VERİ AYRIŞTIRMA VE HATA KONTROLÜ ---
        def parse_input(raw, default_val=0):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        pa = parse_input(p_a_raw)
        ms_val = parse_input(m_s_raw) # Moment şiddetleri
        mk_pos = parse_input(m_k_raw) # Moment konumları
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        x = np.linspace(0, L, 1000)
        N, V = np.zeros_like(x), np.zeros_like(x)
        
        # Kuvvet Bileşenleri
        rad = np.deg2rad(pa) if pa.size > 0 else np.array([])
        py = ps * np.sin(rad) if ps.size > 0 else np.zeros_like(ps)
        px = ps * np.cos(rad) if ps.size > 0 else np.zeros_like(ps)
        if px.size > 0: px[np.abs(px) < 1e-10] = 0

        # Yayılı Yük Bileşkesi
        w_totals = ws * (np.minimum(we, L) - np.maximum(wb, 0)) if ws.size > 0 else np.array([0])
        w_centroids = (wb + we) / 2 if wb.size > 0 else np.array([0])
        total_w_force = np.sum(w_totals)

        # --- REAKSİYON HESABI ---
        if 3 in m_type and len(m_pos) == 1: # Konsol
            fixed_x = m_pos[0]
            R1y = np.sum(py) + total_w_force
            R1x = -np.sum(px)
            # Moment dengesi (Ankastre noktası) + Dış Momentler
            M_ext = np.sum(py * (pk - fixed_x)) + np.sum(w_totals * (w_centroids - fixed_x)) - np.sum(ms_val)
            R2y = 0
        else: # İki Mesnetli
            m1, m2 = m_pos[0], m_pos[1]
            # Moment dengesi (m1 noktasına göre): Saat yönü tersi (+)
            moment_sum = np.sum(py * (pk - m1)) + np.sum(w_totals * (w_centroids - m1)) - np.sum(ms_val)
            R2y = moment_sum / (m2 - m1)
            R1y = (np.sum(py) + total_w_force) - R2y
            R1x = -np.sum(px)
            M_ext = 0

        # --- DİYAGRAMLAR ---
        for i, xi in enumerate(x):
            # N ve V
            if xi >= m_pos[0]: N[i] += R1x
            if xi >= m_pos[0]: V[i] += R1y
            if len(m_pos) > 1 and xi >= m_pos[1]: V[i] += R2y
            if ps.size > 0:
                for j in range(len(ps)):
                    if xi >= pk[j]: V[i] -= py[j]; N[i] += px[j]
            if ws.size > 0:
                for k in range(len(ws)):
                    if xi > wb[k]:
                        active_w = min(xi, we[k]) - wb[k]
                        if active_w > 0: V[i] -= ws[k] * active_w

        # Moment (V'nin integrali + Tekil Moment Sıçramaları)
        dx = L / 999
        M = np.cumsum(V) * dx
        if 3 in m_type and m_pos[0] == 0: M -= M_ext
        
        # Tekil Moment Sıçramaları
        if ms_val.size > 0:
            for m_v, m_k in zip(ms_val, mk_pos):
                M[x >= m_k] += m_v

        # --- ÇİZİM ---
        fig, axes = plt.subplots(4, 1, figsize=(10, 14))
        
        # Şema
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=20, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=15, color='gray')
            if t == 3: axes[0].vlines(p, -0.5, 0.5, color='black', lw=10)
        
        # Tekil Moment Gösterimi (Dairesel Ok)
        for mv, mk in zip(ms_val, mk_pos):
            axes[0].plot(mk, 0.3, 'o', mfc='none', mec='purple', ms=15)
            axes[0].text(mk, 0.5, f'{mv}kNm', color='purple', ha='center', fontweight='bold')

        axes[0].set_ylim(-1.5, 2); axes[0].axis('off'); axes[0].set_title("Sistem Geometrisi")

        # Diyagramlar
        titles = ["N (kN)", "V (kN)", "M (kNm)"]
        colors = ['g', 'b', 'r']
        data_plots = [N, V, M]
        for ax, t, c, d in zip(axes[1:], titles, colors, data_plots):
            ax.plot(x, d, color=c, lw=2)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.set_ylabel(t); ax.grid(True, alpha=0.3); ax.axhline(0, color='black')
            if t == "M (kNm)": ax.invert_yaxis()

        st.pyplot(fig)
        st.info(f"Sonuçlar: R1y={R1y:.2f}kN, R2y={R2y:.2f}kN, R1x={R1x:.2f}kN")

    except Exception as e:
        st.warning("Lütfen sistem verilerini kontrol edin. (Boş bırakılan yükler sıfır kabul edilir)")

analiz_motoru()
