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

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR (GİRİŞ PANELİ) ---
st.sidebar.header("📐 Sistem Parametreleri")
L = st.sidebar.number_input("Kiriş Toplam Boyu (m)", value=8.0, min_value=0.1)

st.sidebar.subheader("⚪ Mafsallar (Gerber)")
mafsal_raw = st.sidebar.text_input("Mafsal Konumları (m)", "4")

st.sidebar.subheader("🔗 Mesnetler")
m_pos_raw = st.sidebar.text_input("Mesnet Konumları", "0, 8")
m_type_raw = st.sidebar.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "3, 2")

st.sidebar.subheader("🔴 Tekil Yükler")
p_s_raw = st.sidebar.text_input("Yük Şiddetleri (kN)", "4")
p_k_raw = st.sidebar.text_input("Yük Konumları (m)", "6")
p_a_raw = st.sidebar.text_input("Yük Açıları (Derece)", "90")

st.sidebar.subheader("🔄 Tekil Momentler")
m_s_raw = st.sidebar.text_input("Moment Şiddetleri (kNm)", "")
m_k_raw = st.sidebar.text_input("Moment Konumları (m)", "")

st.sidebar.subheader("🟠 Yayılı Yükler")
w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            processed = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in processed]) if processed else np.array([])

        # Girdiler
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

        # Bilinmeyen ve Matris Kurulumu
        n_mesnet = len(m_pos)
        ank_var = 3 in m_type
        n_vars = n_mesnet + (1 if ank_var else 0)
        
        A = np.zeros((n_vars, n_vars))
        B = np.zeros(n_vars)

        # 1. Fy = 0
        A[0, :n_mesnet] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. Moment Toplamı (x=0)
        for i in range(n_mesnet):
            A[1, i] = m_pos[i]
        if ank_var: A[1, n_vars-1] = 1
        B[1] = np.sum(ps * pk) + np.sum(ms_val)
        for i in range(len(ws)):
            B[1] += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)

        # 3. Mafsal Şartı (M_sol = 0)
        for i, maf_x in enumerate(mafsallar):
            row = 2 + i
            if row >= n_vars: break
            for j in range(n_mesnet):
                if m_pos[j] < maf_x: A[row, j] = (maf_x - m_pos[j])
            if ank_var and m_pos[0] < maf_x: A[row, n_vars-1] = 1
            
            m_load = np.sum(ps[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    end = min(we[k], maf_x)
                    m_load += (ws[k] * (end - wb[k])) * (maf_x - (wb[k] + end)/2)
            B[row] = m_load

        reak = np.linalg.solve(A, B)

        # Diyagram Hesapları
        x = np.linspace(0, L, 1001)
        V, M = np.zeros_like(x), np.zeros_like(x)
        for i, xi in enumerate(x):
            v, m = 0, 0
            for j in range(n_mesnet):
                if xi >= m_pos[j] - 1e-9:
                    v += reak[j]; m += reak[j]*(xi - m_pos[j])
            if ank_var and xi >= m_pos[0] - 1e-9: m -= reak[n_vars-1]
            for j in range(len(ps)):
                if xi >= pk[j] - 1e-9:
                    v -= ps[j]; m -= ps[j]*(xi - pk[j])
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_L = min(xi, we[k]) - wb[k]
                    v -= ws[k]*w_L; m -= (ws[k]*w_L)*(xi - (wb[k] + min(xi, we[k]))/2)
            V[i], M[i] = v, m

        # Grafik Hazırlığı
        fig, axes = plt.subplots(4, 1, figsize=(11, 15))
        plt.subplots_adjust(hspace=0.6)

        # Şema
        axes[0].hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 3: axes[0].vlines(p, -0.6, 0.6, color='black', lw=10)
            else: axes[0].plot(p, -0.2, '^' if t==1 else 'o', ms=15, color='gray')
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)
        
        # Yük ve Değer Gösterimi
        for i in range(len(ps)):
            axes[0].annotate(f'{ps[i]}kN', (pk[i], 0.1), xytext=(pk[i], 1.2), arrowprops=dict(arrowstyle='->', color='red'), color='red', ha='center', fontweight='bold')
        for k in range(len(ws)):
            axes[0].add_patch(plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3))
            axes[0].text((wb[k]+we[k])/2, 0.7, f'{ws[k]}kN/m', ha='center', fontweight='bold', color='darkorange')
        axes[0].axis('off')

        # Diyagramlar ve Tüm Kırılım Noktaları
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk, wb, we)))
        titles = ["N (Normal Kuvvet)", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        data = [np.zeros_like(x), V, M]; colors = ['green', 'blue', 'red']

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data)):
            ax.plot(x, d, color=c, lw=2.5); ax.fill_between(x, d, color=c, alpha=0.1)
            ax.axhline(0, color='black', lw=1); ax.set_title(t, fontweight='bold', loc='left'); ax.grid(True, alpha=0.2)
            
            if i > 0: # Etiketleme
                for kx in kritik_x:
                    idx = np.abs(x - kx).argmin()
                    if i == 1: # Kesme Sıçramaları
                        v1, v2 = d[max(0, idx-2)], d[min(len(d)-1, idx+2)]
                        ax.text(kx, v1, f'{v1:.1f}', color=c, fontsize=8, ha='right', fontweight='bold')
                        if abs(v2-v1) > 0.1: ax.text(kx, v2, f'{v2:.1f}', color=c, fontsize=8, ha='left', fontweight='bold')
                    else: # Moment Kırılımları
                        ax.text(kx, d[idx], f'{d[idx]:.1f}', color=c, fontsize=8, ha='center', va='top' if d[idx]<0 else 'bottom', fontweight='bold')
            if i == 2: ax.invert_yaxis()

        st.pyplot(fig)
        st.subheader("📋 Hesaplanan Reaksiyonlar")
        cols = st.columns(n_vars)
        for i in range(n_mesnet): cols[i].metric(f"Mesnet {i+1}", f"{reak[i]:.2f} kN")
        if ank_var: cols[-1].metric("Ankastre Momenti", f"{reak[n_vars-1]:.2f} kNm")

    except Exception as e: st.error(f"Hata: {e}")

if __name__ == "__main__": analiz_motoru()
