import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Gerber Analiz Sistemi", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    h1 {color: #1E3A8A; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Profesyonel Gerber Kiriş Analiz Sistemi")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📐 Sistem Parametreleri")
    L = st.number_input("Kiriş Toplam Boyu (m)", value=8.0)
    mafsallar_in = st.text_input("Mafsal Konumları (m)", "4")
    m_pos_in = st.text_input("Mesnet Konumları", "0, 8")
    m_type_in = st.text_input("Türler (1:Sabit, 2:Hark, 3:Ankastre)", "3, 2")
    
    st.subheader("🔴 Yükler")
    ps_in = st.text_input("Tekil Yükler (kN)", "4")
    pk_in = st.text_input("Konumları (m)", "6")
    ws_in = st.text_input("Yayılı Yükler (kN/m)", "2")
    wb_in = st.text_input("Başlangıç (m)", "0")
    we_in = st.text_input("Bitiş (m)", "4")

def analiz_motoru():
    try:
        def parse(raw):
            return np.array([float(i.strip()) for i in raw.split(',') if i.strip()])

        m_pos = parse(m_pos_in)
        m_type = parse(m_type_in).astype(int)
        maf = parse(mafsallar_in)
        ps, pk = parse(ps_in), parse(pk_in)
        ws, wb, we = parse(ws_in), parse(wb_in), parse(we_in)

        # Reaksiyon Tanımları
        reaks = []
        for i, t in enumerate(m_type):
            reaks.append({'pos': m_pos[i], 'type': 'Ry'})
            if t == 3: reaks.append({'pos': m_pos[i], 'type': 'Ma'})
        
        n = len(reaks)
        A, B = np.zeros((n, n)), np.zeros(n)

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
            if i + 2 >= n: break
            row = i + 2
            for j, r in enumerate(reaks):
                if r['pos'] <= mx:
                    if r['type'] == 'Ry': A[row, j] = (mx - r['pos'])
                    if r['type'] == 'Ma': A[row, j] = 1
            
            B[row] = np.sum(ps[pk <= mx] * (mx - pk[pk <= mx]))
            for k in range(len(ws)):
                if wb[k] < mx:
                    ex = min(we[k], mx)
                    B[row] += (ws[k] * (ex - wb[k])) * (mx - (wb[k] + ex)/2)

        X_res = np.linalg.solve(A, B)

        # --- DİYAGRAM HESABI ---
        x_vals = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x_vals), np.zeros_like(x_vals)

        for i, x in enumerate(x_vals):
            v_tmp, m_tmp = 0, 0
            for j, r in enumerate(reaks):
                if x >= r['pos']:
                    if r['type'] == 'Ry':
                        v_tmp += X_res[j]
                        m_tmp += X_res[j] * (x - r['pos'])
                    if r['type'] == 'Ma':
                        m_tmp -= X_res[j] # İşaret düzeltmesi (El hesabı uyumu)
            
            v_tmp -= np.sum(ps[pk <= x])
            m_tmp -= np.sum(ps[pk <= x] * (x - pk[pk <= x]))
            
            for k in range(len(ws)):
                if x > wb[k]:
                    dist = min(x, we[k]) - wb[k]
                    v_tmp -= ws[k] * dist
                    m_tmp -= (ws[k] * dist) * (x - (wb[k] + min(x, we[k]))/2)
            
            V[i], M[i] = v_tmp, m_tmp

        # --- GRAFİKLER ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for ax, data, title, color in zip(axes, [V, M], ["V (Kesme) - kN", "M (Moment) - kNm"], ["blue", "red"]):
            ax.plot(x_vals, data, color=color, lw=2)
            ax.fill_between(x_vals, data, color=color, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(title, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.2)
            if "Moment" in title: ax.invert_yaxis()
            
            k_pts = np.unique(np.concatenate(([0, L], m_pos, maf, pk)))
            for p in k_pts:
                val = data[np.abs(x_vals - p).argmin()]
                ax.text(p, val, f' {val:.1f}', fontsize=9, fontweight='bold')

        st.pyplot(fig)

        # SONUÇLAR
        st.subheader("📊 Mesnet Tepkileri")
        cols = st.columns(len(reaks))
        for i, r in enumerate(reaks):
            unit = "kN" if r['type'] == 'Ry' else "kNm"
            cols[i].metric(f"{r['pos']}m - {r['type']}", f"{X_res[i]:.1f} {unit}")

    except Exception as e:
        st.error(f"Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
