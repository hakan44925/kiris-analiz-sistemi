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
        # Veri Ayrıştırma Fonksiyonu
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

        # HASSASİYET ARTIRIMI: 1000 -> 10000 parça
        x = np.linspace(0, L, 10000)
        N, V = np.zeros_like(x), np.zeros_like(x)
        
        rad = np.deg2rad(pa) if pa.size > 0 else np.array([])
        py = ps * np.sin(rad) if ps.size > 0 else np.array([])
        px = ps * np.cos(rad) if ps.size > 0 else np.array([])
        if px.size > 0: px[np.abs(px) < 1e-10] = 0

        w_totals = ws * (np.minimum(we, L) - np.maximum(wb, 0)) if ws.size > 0 else np.array([0])
        w_centroids = (wb + we) / 2 if wb.size > 0 else np.array([0])
        total_w_force = np.sum(w_totals)

        # Statik Hesaplar (Reaksiyonlar)
        if 3 in m_type and len(m_pos) == 1:
            fixed_x = m_pos[0]
            R1y = np.sum(py) + total_w_force
            R1x = -np.sum(px)
            M_ext = np.sum(py * (pk - fixed_x)) + np.sum(w_totals * (w_centroids - fixed_x)) - np.sum(ms_val)
            R2y = 0
        else:
            m1, m2 = m_pos[0], m_pos[1]
            moment_sum = np.sum(py * (pk - m1)) + np.sum(w_totals * (w_centroids - m1)) - np.sum(ms_val)
            R2y = moment_sum / (m2 - m1)
            R1y = (np.sum(py) + total_w_force) - R2y
            R1x = -np.sum(px)
            M_ext = 0

        # Diyagram Hesapları
        for i, xi in enumerate(x):
            if xi >= m_pos[0]: N[i] += R1x
            if xi >= m_pos[0]: V[i] += R1y
            if len(m_pos) > 1 and xi >= m_pos[1]: V[i] += R2y
            for j in range(len(ps)):
                if xi >= pk[j]: V[i] -= py[j]; N[i] += px[j]
            for k in range(len(ws)):
                if xi > wb[k]:
                    active_w = min(xi, we[k]) - wb[k]
                    if active_w > 0: V[i] -= ws[k] * active_w

        # Hassas İntegral (Moment Hesabı)
        dx = L / (len(x) - 1)
        M = np.cumsum(V) * dx
        if 3 in m_type and m_pos[0] == 0: M -= M_ext
        for m_v, m_k in zip(ms_val, mk_pos):
            M[x >= m_k] += m_v

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 20))
        
        # 1. SİSTEM ŞEMASI
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=20, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=15, color='gray')
            if t == 3: axes[0].vlines(p, -0.6, 0.6, color='black', lw=10)
        
        # Yük Okları
        for i in range(len(ps)):
            dx_ok = 0.8 * np.cos(np.deg2rad(pa[i] + 180))
            dy_ok = 0.8 * np.sin(np.deg2rad(pa[i] + 180))
            axes[0].annotate(f'{ps[i]}kN', xy=(pk[i], 0), xytext=(pk[i]-dx_ok, -dy_ok),
                             arrowprops=dict(facecolor='red', width=2), ha='center', color='red', weight='bold')
        
        # Yayılı Yük Alanları
        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.5, color='orange', alpha=0.3)
            axes[0].add_patch(rect)
            axes[0].text((wb[k]+we[k])/2, 0.6, f'{ws[k]}kN/m', color='orange', ha='center', weight='bold')

        # Tekil Momentler
        for mv, mk in zip(ms_val, mk_pos):
            axes[0].plot(mk, 0.3, 'o', mfc='none', mec='purple', ms=20, mew=2)
            axes[0].text(mk, 0.7, f'{mv}kNm', color='purple', ha='center', weight='bold')

        axes[0].set_ylim(-1.5, 3); axes[0].axis('off')
        axes[0].set_title("Sistem ve Yükleme Modeli", fontweight='bold', fontsize=14)

        # Kritik Noktaları Belirle
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, pk, wb, we, mk_pos)))

        # Diyagramlar
        titles = ["Eksenel Kuvvet - N (kN)", "Kesme Kuvveti - V (kN)", "Eğilme Momenti - M (kNm)"]
        colors = ['green', 'blue', 'red']
        data_list = [N, V, M]

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data_list)):
            ax.plot(x, d, color=c, lw=2.5)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.set_ylabel(t, fontweight='bold'); ax.grid(True, alpha=0.3); ax.axhline(0, color='black', lw=1.5)
            if "M (kNm)" in t: ax.invert_yaxis()

            # Değer Yazdırma ve Yuvarlama
            for kx in kritik_x:
                if kx > L: continue
                idx = np.argmin(np.abs(x - kx))
                val = d[idx]
                
                # Sayısal hataları temizle (11.99 -> 12.0)
                val_display = round(val, 2)
                
                if abs(val_display) > 0.01 or kx in [0, L]:
                    offset = np.max(np.abs(d)) * 0.12 if np.max(np.abs(d)) > 0 else 0.5
                    if "M (kNm)" in t: offset *= -1
                    ax.text(kx, val + offset, f'{val_display:.1f}', ha='center', fontsize=10, 
                            fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                    ax.plot(kx, val, 'o', color=c, ms=5)

        st.pyplot(fig)
        st.success(f"Analiz Tamamlandı! R1y: {R1y:.2f}kN | R2y: {R2y:.2f}kN | R1x: {R1x:.2f}kN")

    except Exception as e:
        st.warning("Lütfen giriş verilerini kontrol edin. Tüm alanların virgülle ayrılmış olduğundan emin olun.")

analiz_motoru()
