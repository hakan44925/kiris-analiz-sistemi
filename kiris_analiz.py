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

        # Girdileri Parse Et
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

        x_plot = np.linspace(0, L, 2000)
        N, V, M = np.zeros_like(x_plot), np.zeros_like(x_plot), np.zeros_like(x_plot)
        
        # Açıları Düzenle
        rad = np.deg2rad(pa) if pa.size > 0 else np.full_like(ps, np.pi/2)
        py = ps * np.sin(rad) if ps.size > 0 else ps
        px = ps * np.cos(rad) if ps.size > 0 else np.zeros_like(ps)

        # --- REAKSİYON ÇÖZÜCÜ (MATRİS) ---
        n_mesnet = len(m_pos)
        n_reak = n_mesnet + (1 if 3 in m_type else 0)
        
        A = np.zeros((n_reak, n_reak))
        B = np.zeros(n_reak)

        # 1. Denklem: Düşey Denge (ΣFy = 0)
        A[0, :n_mesnet] = 1
        B[0] = np.sum(py) + np.sum(ws * (we - wb))

        # 2. Denklem: Moment Dengesi (x=0'a göre)
        for i in range(n_mesnet):
            A[1, i] = m_pos[i]
        if 3 in m_type:
            A[1, n_mesnet] = 1 # Ankastre momenti
        
        ext_moment = np.sum(py * pk) + np.sum(ms_val)
        for i in range(len(ws)):
            ext_moment += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)
        B[1] = ext_moment

        # 3. Denklemler: Mafsal Koşulları (ΣM_mafsal_sol = 0)
        for i, mx in enumerate(mafsallar):
            if i + 2 >= n_reak: break
            row = i + 2
            for j in range(n_mesnet):
                if m_pos[j] < mx:
                    A[row, j] = mx - m_pos[j]
            if 3 in m_type:
                ank_pos = m_pos[np.where(m_type == 3)[0][0]]
                if ank_pos < mx: A[row, n_mesnet] = 1
            
            m_load_maf = np.sum(py[pk < mx] * (mx - pk[pk < mx])) + np.sum(ms_val[mk_pos < mx])
            for k in range(len(ws)):
                if wb[k] < mx:
                    w_limit = min(we[k], mx)
                    m_load_maf += (ws[k] * (w_limit - wb[k])) * (mx - (wb[k] + w_limit)/2)
            B[row] = m_load_maf

        R = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI ---
        for i, xi in enumerate(x_plot):
            cv, cm, cn = 0, 0, 0
            # Mesnetler
            for j in range(n_mesnet):
                if xi >= m_pos[j]:
                    cv += R[j]
                    cm += R[j] * (xi - m_pos[j])
            if 3 in m_type and xi >= m_pos[np.where(m_type == 3)[0][0]]:
                cm -= R[n_mesnet]
            # Tekil Yükler
            for j in range(len(ps)):
                if xi >= pk[j]:
                    cv -= py[j]
                    cm -= py[j] * (xi - pk[j])
                    cn -= px[j]
            # Yayılı Yükler
            for k in range(len(ws)):
                if xi > wb[k]:
                    length = min(xi, we[k]) - wb[k]
                    cv -= ws[k] * length
                    cm -= (ws[k] * length) * (xi - (wb[k] + length/2))
            # Momentler
            for mv, mk in zip(ms_val, mk_pos):
                if xi >= mk: cm += mv
            V[i], M[i], N[i] = cv, cm, cn

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 15), gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.6)

        # 1. Şema
        ax0 = axes[0]
        ax0.hlines(0, 0, L, color='black', lw=4)
        for p, t in zip(m_pos, m_type):
            if t == 1: ax0.plot(p, -0.2, '^', ms=18, color='gray')
            if t == 2: ax0.plot(p, -0.2, 'o', ms=14, color='gray')
            if t == 3: ax0.vlines(p, -0.5, 0.5, color='black', lw=8)
        
        if mafsallar.size > 0:
            ax0.scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)

        for k in range(len(ws)):
            rect = plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.5, color='orange', alpha=0.3)
            ax0.add_patch(rect)
            ax0.text((wb[k]+we[k])/2, 0.6, f"{ws[k]} kN/m", ha='center', color='darkorange', fontweight='bold')
            ax0.text((wb[k]+we[k])/2, -0.6, f"{wb[k]}m - {we[k]}m", ha='center', fontsize=8, color='gray')

        ax0.set_ylim(-1, 1.5)
        ax0.set_title("Kiriş Yükleme ve Sistem Şeması", fontweight='bold')
        ax0.axis('off')

        # 2. Diyagramlar
        titles = ["N (Eksenel) [kN]", "V (Kesme) [kN]", "M (Moment) [kNm]"]
        datas = [N, V, M]
        colors = ['#059669', '#2563EB', '#DC2626']
        
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, pk, wb, we, mk_pos, mafsallar)))

        for i, (ax, t, d, c) in enumerate(zip(axes[1:], titles, datas, colors)):
            ax.plot(x_plot, d, color=c, lw=2)
            ax.fill_between(x_plot, d, color=c, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(t, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            if "M" in t: ax.invert_yaxis()

            # Değer Yazdırma
            for kx in kritik_x:
                idx = np.argmin(np.abs(x_plot - kx))
                val = d[idx]
                if abs(val) > 0.01:
                    ax.text(kx, val, f"{val:.1f}", fontsize=8, fontweight='bold', ha='center', va='bottom' if val > 0 else 'top')
            
            # Max Gösterimi
            mx_val = d[np.argmax(np.abs(d))]
            ax.set_xlabel(f"MAKS: {mx_val:.2f}", loc='right', color=c, fontweight='bold')

        st.pyplot(fig)
        st.success("Analiz Tamamlandı!")

    except Exception as e:
        st.error(f"Sistem hatası! Lütfen mesnet/mafsal sayısının dengeli olduğundan emin olun. Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
