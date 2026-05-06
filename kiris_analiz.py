import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A; margin-bottom: 0px;}
    hr {margin-top: 1rem; margin-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.write("---")

# --- SIDEBAR (MANUEL GİRİŞLER) ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0)
    mafsallar_raw = st.text_input("Mafsal Konumları (m)", "4")
    m_pos_raw = st.text_input("Mesnet Konumları", "0, 8")
    m_type_raw = st.text_input("Mesnet Türleri (1:Sabit, 2:Hark, 3:Ank)", "3, 2")

    st.subheader("🔴 Yükler")
    ps_raw = st.text_input("Tekil Yükler (kN)", "4")
    pk_raw = st.text_input("Yük Konumları (m)", "6")
    ws_raw = st.text_input("Yayılı Yük Şiddetleri (kN/m)", "2")
    wb_raw = st.text_input("Başlangıç (m)", "0")
    we_raw = st.text_input("Bitiş (m)", "4")

def analiz_motoru():
    try:
        def parse(raw):
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

        m_pos = parse(m_pos_raw)
        m_type = parse(m_type_raw).astype(int)
        maf = parse(mafsallar_raw)
        ps, pk = parse(ps_raw), parse(pk_raw)
        ws, wb, we = parse(ws_raw), parse(wb_raw), parse(we_raw)

        # Reaksiyon Tanımları
        reaks = []
        for i, t in enumerate(m_type):
            reaks.append({'pos': m_pos[i], 'type': 'Ry', 'label': f'R{i}y'})
            if t == 3: reaks.append({'pos': m_pos[i], 'type': 'Ma', 'label': f'M{i}'})
        
        n = len(reaks)
        A, B = np.zeros((n, n)), np.zeros(n)

        # Denklemler
        # 1. ΣFy = 0
        for j, r in enumerate(reaks):
            if r['type'] == 'Ry': A[0, j] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. ΣM_x0 = 0
        for j, r in enumerate(reaks):
            if r['type'] == 'Ry': A[1, j] = r['pos']
            if r['type'] == 'Ma': A[1, j] = 1
        B[1] = np.sum(ps * pk) + np.sum(ws * (we - wb) * (wb + we)/2)

        # 3. Mafsal Denklemleri (ΣM_sol = 0)
        for i, mx in enumerate(maf):
            row = i + 2
            if row >= n: break
            for j, r in enumerate(reaks):
                if r['pos'] <= mx:
                    if r['type'] == 'Ry': A[row, j] = (mx - r['pos'])
                    if r['type'] == 'Ma': A[row, j] = 1
            
            B[row] = np.sum(ps[pk <= mx] * (mx - pk[pk <= mx]))
            for k in range(len(ws)):
                if wb[k] < mx:
                    ex = min(we[k], mx)
                    B[row] += (ws[k] * (ex - wb[k])) * (mx - (wb[k] + ex)/2)

        res = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI (KESİM METODU) ---
        x_plot = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x_plot), np.zeros_like(x_plot)

        for i, x in enumerate(x_plot):
            v_val, m_val = 0, 0
            # Reaksiyonların etkisi (Sol taraftaki reaksiyonlar)
            for j, r in enumerate(reaks):
                if x >= r['pos']:
                    if r['type'] == 'Ry':
                        v_val += res[j]
                        m_val += res[j] * (x - r['pos'])
                    if r['type'] == 'Ma':
                        # EL HESABI DÜZELTMESİ: 
                        # Saat yönü tersi varsayılan Ma, diyagramı eksiye çeker.
                        m_val -= res[j] 
            
            # Yüklerin etkisi
            for k in range(len(ps)):
                if x >= pk[k]:
                    v_val -= ps[k]
                    m_val -= ps[k] * (x - pk[k])
            
            for k in range(len(ws)):
                if x > wb[k]:
                    length = min(x, we[k]) - wb[k]
                    v_val -= ws[k] * length
                    m_val -= (ws[k] * length) * (x - (wb[k] + min(x, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 11))
        plt.subplots_adjust(hspace=0.5)

        # Sistem Şeması
        axes[0].hlines(0, 0, L, color='black', lw=4)
        for r in reaks:
            if r['type'] == 'Ry':
                axes[0].plot(r['pos'], -0.2, '^', ms=12, color='gray')
        if maf.size > 0:
            axes[0].scatter(maf, [0]*len(maf), c='white', edgecolors='black', s=80, zorder=5)
        axes[0].axis('off')

        # V ve M Diyagramları
        for ax, data, title, color in zip(axes[1:], [V, M], ["V (Kesme) - kN", "M (Moment) - kNm"], ["blue", "red"]):
            ax.plot(x_plot, data, color=color, lw=2)
            ax.fill_between(x_plot, data, color=color, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if "Moment" in title: ax.invert_yaxis()
            
            kp = np.unique(np.concatenate(([0, L], m_pos, maf, pk)))
            for p in kp:
                val = data[np.abs(x_plot - p).argmin()]
                ax.text(p, val, f' {val:.1f}', fontsize=8, fontweight='bold')

        st.pyplot(fig)

        # Reaksiyonlar
        st.subheader("📊 Reaksiyon Sonuçları")
        cols = st.columns(len(reaks))
        for i, r in enumerate(reaks):
            cols[i].metric(f"{r['label']} ({r['pos']}m)", f"{res[i]:.2f}")

    except Exception as e:
        st.error(f"Sistem belirsiz veya hatalı veri: {e}")

if __name__ == "__main__":
    analiz_motoru()
