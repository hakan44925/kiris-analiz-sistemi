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

# --- SIDEBAR (GİRİŞ PANELİ) ---
st.sidebar.header("📐 Sistem Parametreleri")
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=12.0, min_value=0.1)
mafsal_raw = st.sidebar.text_input("Mafsal Konumları (m)", "")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 10")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "1, 2")

st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "20")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "5")
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
        def parse_input(raw):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        # Input Parsing
        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        pa = parse_input(p_a_raw)
        ms_val = parse_input(m_s_raw)
        mk_pos = parse_input(m_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Çözünürlük
        x = np.linspace(0, L, 2000)
        N, V, M = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
        
        # Açı Ayarı (Boşsa 90 derece kabul et)
        rad = np.deg2rad(pa) if pa.size > 0 else np.full_like(ps, np.pi/2)
        py = ps * np.sin(rad) if ps.size > 0 else ps
        px = ps * np.cos(rad) if ps.size > 0 else np.zeros_like(ps)

        # --- REAKSİYON HESABI ---
        n_reak = len(m_pos)
        ank_mom_idx = -1
        if 3 in m_type:
            ank_mom_idx = n_reak
            n_reak += 1
        
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Toplam Düşey Denge
        A[0, :len(m_pos)] = 1
        B[0] = np.sum(py) + np.sum(ws * (we - wb))

        # 2. Toplam Moment Dengesi (x=0'a göre)
        for i in range(len(m_pos)):
            A[1, i] = m_pos[i]
        if ank_mom_idx != -1:
            A[1, ank_mom_idx] = 1
        
        m_load = np.sum(py * pk) + np.sum(ms_val)
        for i in range(len(ws)):
            m_load += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)
        B[1] = m_load

        # 3. Mafsal Denklemleri (ΣM_mafsal_sol = 0)
        for i, maf_x in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            for j in range(len(m_pos)):
                if m_pos[j] < maf_x:
                    A[row, j] = maf_x - m_pos[j]
            if ank_mom_idx != -1:
                ank_pos = m_pos[np.where(m_type == 3)[0][0]]
                if ank_pos < maf_x: A[row, ank_mom_idx] = 1

            m_load_maf = np.sum(py[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            m_load_maf += np.sum(ms_val[mk_pos < maf_x])
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    w_end = min(we[k], maf_x)
                    m_load_maf += (ws[k] * (w_end - wb[k])) * (maf_x - (wb[k] + w_end)/2)
            B[row] = m_load_maf

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        for i, xi in enumerate(x):
            v_val, m_val, n_val = 0, 0, 0
            for j in range(len(m_pos)):
                if xi >= m_pos[j]:
                    v_val += reaksiyonlar[j]
                    m_val += reaksiyonlar[j] * (xi - m_pos[j])
            if ank_mom_idx != -1:
                ank_pos = m_pos[np.where(m_type == 3)[0][0]]
                if xi >= ank_pos: m_val -= reaksiyonlar[ank_mom_idx]
            for j in range(len(ps)):
                if xi >= pk[j]:
                    v_val -= py[j]
                    m_val -= py[j] * (xi - pk[j])
                    n_val -= px[j]
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_len = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * w_len
                    m_val -= (ws[k] * w_len) * (xi - (wb[k] + w_len/2))
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk: m_val += mv
            V[i], M[i], N[i] = v_val, m_val, n_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        plt.subplots_adjust(hspace=0.6)

        # 1. ŞEMA
        axes[0].hlines(0, 0, L, color='black', lw=5)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=20, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=15, color='gray')
            if t == 3: axes[0].vlines(p, -0.6, 0.6, color='black', lw=10)
        
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=120, zorder=5)

        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3)
            axes[0].add_patch(rect)
            axes[0].text((wb[k]+we[k])/2, 0.7, f"{ws[k]} kN/m", ha='center', fontweight='bold', color='darkorange')
            axes[0].text((wb[k]+we[k])/2, -0.6, f"{wb[k]}m - {we[k]}m", ha='center', fontsize=9, color='orange')

        axes[0].set_ylim(-1, 2)
        axes[0].axis('off')

        # 2. DİYAGRAMLAR
        titles = ["N (Eksenel Kuvvet) - kN", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        units = ["kN", "kN", "kNm"]
        data_list = [N, V, M]
        colors = ['green', '#1E40AF', '#B91C1C']
        
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, pk, wb, we, mk_pos, mafsallar)))

        for i, (ax, t, c, d, u) in enumerate(zip(axes[1:], titles, colors, data_list, units)):
            ax.plot(x, d, color=c, lw=2.5)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.axhline(0, color='black', lw=1.5)
            ax.set_title(t, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            if "M" in t: ax.invert_yaxis()

            # Değer Yazdırma
            for kx in kritik_x:
                idx = np.argmin(np.abs(x - kx))
                val = d[idx]
                if abs(val) > 0.01:
                    ax.text(kx, val, f"{val:.1f} {u}", fontsize=8, fontweight='bold', ha='center')

            # Max Noktası
            max_idx = np.argmax(np.abs(d))
            ax.text(x[max_idx], d[max_idx], f"MAX: {d[max_idx]:.1f}", color='black', fontweight='black', bbox=dict(facecolor='white', alpha=0.5))

        st.pyplot(fig)
        st.success("Analiz Başarıyla Tamamlandı!")

    except Exception as e:
        st.warning(f"Sistem çözülemedi. Lütfen mesnet/mafsal sayısını (İzostatiklik) kontrol edin. Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
