import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    h1 {margin-bottom: 0rem;}
    hr {margin-top: 1rem; padding-top: 0rem;}
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

st.sidebar.subheader("🟠 Yayılı Yükler")
w_s_raw = st.sidebar.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
w_b_raw = st.sidebar.text_input("Başlangıç Metreleri", "0")
w_e_raw = st.sidebar.text_input("Bitiş Metreleri", "4")

def analiz_motoru():
    try:
        def parse_input(raw):
            proc = [i.strip() for i in raw.split(',') if i.strip()]
            return np.array([float(i) for i in proc]) if proc else np.array([])

        m_pos = parse_input(m_pos_raw)
        m_type = parse_input(m_type_raw).astype(int)
        mafsallar = parse_input(mafsal_raw)
        ps = parse_input(p_s_raw)
        pk = parse_input(p_k_raw)
        ws = parse_input(w_s_raw)
        wb = parse_input(w_b_raw)
        we = parse_input(w_e_raw)

        # Bilinmeyenler: Mesnet Reaksiyonları + Ankastre Momentleri
        ank_count = np.sum(m_type == 3)
        n_vars = len(m_pos) + ank_count
        
        A = np.zeros((n_vars, n_vars))
        B = np.zeros(n_vars)

        # 1. Toplam Fy = 0
        A[0, :len(m_pos)] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. Toplam M (x=0) = 0
        for i in range(len(m_pos)):
            A[1, i] = m_pos[i]
        if ank_count > 0:
            A[1, len(m_pos)] = 1 # İlk ankastre momenti
        
        B[1] = np.sum(ps * pk)
        for i in range(len(ws)):
            B[1] += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)

        # 3. Mafsal Şartları (M=0)
        for i, maf_x in enumerate(mafsallar):
            row = 2 + i
            if row >= n_vars: break
            # Mafsalın solundaki kuvvetlerin mafsala göre momenti
            for j in range(len(m_pos)):
                if m_pos[j] < maf_x:
                    A[row, j] = (maf_x - m_pos[j])
            if ank_count > 0 and m_pos[0] < maf_x:
                A[row, len(m_pos)] = 1
            
            m_load = np.sum(ps[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    w_end = min(we[k], maf_x)
                    w_L = w_end - wb[k]
                    m_load += (ws[k] * w_L) * (maf_x - (wb[k] + w_end)/2)
            B[row] = m_load

        reaksiyonlar = np.linalg.solve(A, B)

        # --- HESAPLAMA ---
        x = np.linspace(0, L, 1001)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            for j in range(len(m_pos)):
                if xi >= m_pos[j] - 1e-9:
                    v_val += reaksiyonlar[j]
                    m_val += reaksiyonlar[j] * (xi - m_pos[j])
            if ank_count > 0 and xi >= m_pos[0] - 1e-9:
                m_val -= reaksiyonlar[len(m_pos)]
            
            for j in range(len(ps)):
                if xi >= pk[j] - 1e-9:
                    v_val -= ps[j]
                    m_val -= ps[j] * (xi - pk[j])
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_L = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * w_L
                    m_val -= (ws[k] * w_L) * (xi - (wb[k] + min(xi, we[k]))/2)
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 14))
        plt.subplots_adjust(hspace=0.6)

        # Şema
        ax0 = axes[0]
        ax0.hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 3: ax0.vlines(p, -0.6, 0.6, color='black', lw=10)
            else: ax0.plot(p, -0.25, '^' if t==1 else 'o', ms=15, color='gray')
        if mafsallar.size > 0:
            ax0.scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)
        
        # Yük Etiketleri
        for i in range(len(ps)):
            ax0.annotate(f'{ps[i]}kN', (pk[i], 0.1), xytext=(pk[i], 1.2), arrowprops=dict(arrowstyle='->', color='red'), color='red', ha='center', fontweight='bold')
        for k in range(len(ws)):
            ax0.add_patch(plt.Rectangle((wb[k], 0), we[k]-wb[k], 0.6, color='orange', alpha=0.3))
            ax0.text((wb[k]+we[k])/2, 0.7, f'{ws[k]}kN/m', ha='center', fontweight='bold')
        ax0.axis('off')

        # Diyagramlar
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk, wb, we)))
        titles = ["N (Normal Kuvvet)", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        data = [np.zeros_like(x), V, M]
        colors = ['green', 'blue', 'red']

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data)):
            ax.plot(x, d, color=c, lw=2)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(t, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            
            if i > 0: # V ve M etiketleme
                for kx in kritik_x:
                    idx = np.abs(x - kx).argmin()
                    # Kesme kuvvetinde sıçrama kontrolü
                    if i == 1:
                        v1, v2 = d[max(0, idx-1)], d[min(len(d)-1, idx+1)]
                        ax.text(kx, v1, f'{v1:.1f}', color=c, fontsize=8, ha='right', va='bottom' if v1>=0 else 'top')
                        if abs(v2-v1) > 0.1:
                            ax.text(kx, v2, f'{v2:.1f}', color=c, fontsize=8, ha='left', va='bottom' if v2>=0 else 'top')
                    else:
                        val = d[idx]
                        ax.text(kx, val, f'{val:.1f}', color=c, fontsize=8, ha='center', va='top' if (i==2 and val<=0) else 'bottom')

            if i == 2: ax.invert_yaxis()

        st.pyplot(fig)
        st.success("Analiz başarıyla tamamlandı.")

    except Exception as e:
        st.error(f"Hata: {e}. Lütfen sistemin statikçe belirli olduğunu (Mesnet + Mafsal dengesi) kontrol edin.")

if __name__ == "__main__":
    analiz_motoru()
