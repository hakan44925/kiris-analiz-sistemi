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

        # Girdileri İşleme
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

        # Bilinmeyen Sayısı Belirleme
        n_mesnet = len(m_pos)
        n_ankastre = np.count_nonzero(m_type == 3)
        n_bilinmeyen = n_mesnet + n_ankastre
        
        A = np.zeros((n_bilinmeyen, n_bilinmeyen))
        B = np.zeros(n_bilinmeyen)

        # Matris İndis Yönetimi: [R1, R2, ..., M_ank]
        # 1. Denklemler: Toplam Fy = 0
        A[0, :n_mesnet] = 1
        total_p_y = np.sum(ps * np.sin(np.deg2rad(pa))) if ps.size > 0 else 0
        total_w_y = np.sum(ws * (we - wb))
        B[0] = total_p_y + total_w_y

        # 2. Denklemler: Toplam M (x=0'a göre) = 0
        for i in range(n_mesnet):
            A[1, i] = m_pos[i]
        ank_idx_list = np.where(m_type == 3)[0]
        if ank_idx_list.size > 0:
            A[1, n_mesnet] = 1 # Ankastre Momenti
        
        moment_load = np.sum(ps * np.sin(np.deg2rad(pa)) * pk) if ps.size > 0 else 0
        moment_load += np.sum(ms_val)
        for i in range(len(ws)):
            moment_load += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)
        B[1] = moment_load

        # 3. Mafsal Denklemleri (Mafsalın bir tarafındaki moment = 0)
        for i, maf_x in enumerate(mafsallar):
            row = 2 + i
            if row >= n_bilinmeyen: break
            # Mafsalın SOL tarafındaki kuvvetlerin mafsala göre momenti
            for j in range(n_mesnet):
                if m_pos[j] < maf_x:
                    A[row, j] = (maf_x - m_pos[j])
            if ank_idx_list.size > 0 and m_pos[ank_idx_list[0]] < maf_x:
                A[row, n_mesnet] = 1
            
            m_load_maf = np.sum(ps[pk < maf_x] * np.sin(np.deg2rad(pa[pk < maf_x])) * (maf_x - pk[pk < maf_x])) if ps.size > 0 else 0
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    eff_we = min(we[k], maf_x)
                    w_len = eff_we - wb[k]
                    m_load_maf += (ws[k] * w_len) * (maf_x - (wb[k] + eff_we)/2)
            B[row] = m_load_maf

        reaksiyonlar = np.linalg.solve(A, B)

        # --- DİYAGRAMLAR ---
        x = np.linspace(0, L, 1000)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            # Mesnetler
            for j in range(n_mesnet):
                if xi >= m_pos[j]:
                    v_val += reaksiyonlar[j]
                    m_val += reaksiyonlar[j] * (xi - m_pos[j])
            # Ankastre Momenti (Genelde x=0'dadır)
            if ank_idx_list.size > 0 and xi >= m_pos[ank_idx_list[0]]:
                m_val -= reaksiyonlar[n_mesnet]

            # Dış Yükler
            if ps.size > 0:
                for j in range(len(ps)):
                    if xi >= pk[j]:
                        py = ps[j] * np.sin(np.deg2rad(pa[j]))
                        v_val -= py
                        m_val -= py * (xi - pk[j])
            
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_len = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * w_len
                    m_val -= (ws[k] * w_len) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 14), gridspec_kw={'height_ratios': [1, 1.2, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.6)

        # Şema
        axes[0].hlines(0, 0, L, color='black', lw=5)
        for p, t in zip(m_pos, m_type):
            if t == 1: axes[0].plot(p, -0.2, '^', ms=15, color='gray')
            if t == 2: axes[0].plot(p, -0.2, 'o', ms=12, color='gray')
            if t == 3: axes[0].vlines(p, -0.5, 0.5, color='black', lw=8)
        if mafsallar.size > 0:
            axes[0].scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=80, zorder=5)
        axes[0].axis('off')

        # Diyagramlar ve Etiketler
        # Kritik noktaları belirle (Sıçramaların olduğu yerler)
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk, wb, we)))
        
        titles = ["N (Normal Kuvvet)", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        data_list = [np.zeros_like(x), V, M]
        colors = ['green', 'blue', 'red']

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors, data_list)):
            ax.plot(x, d, color=c, lw=2)
            ax.fill_between(x, d, color=c, alpha=0.1)
            ax.axhline(0, color='black', lw=1)
            ax.set_title(t, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            
            # Etiketleme Mantığı
            if i > 0: # V ve M için
                for kx in kritik_x:
                    idx = np.abs(x - kx).argmin()
                    val = d[idx]
                    # Kesme kuvvetinde çift taraflı etiketleme (sıçramalar için)
                    if i == 1: # V diyagramı
                        val_plus = d[min(idx+1, len(d)-1)]
                        val_minus = d[max(idx-1, 0)]
                        ax.text(kx, val_minus, f'{val_minus:.1f}', color=c, fontsize=8, ha='right', fontweight='bold')
                        if abs(val_plus - val_minus) > 0.1:
                            ax.text(kx, val_plus, f'{val_plus:.1f}', color=c, fontsize=8, ha='left', fontweight='bold')
                    else: # M diyagramı
                        ax.text(kx, val, f'{val:.1f}', color=c, fontsize=8, ha='center', va='bottom' if val>0 else 'top', fontweight='bold')

            if "M" in t: ax.invert_yaxis()

        st.pyplot(fig)
        st.success("Hesaplama Tamamlandı.")

    except Exception as e:
        st.error(f"Matematiksel Hata: {e}")

if __name__ == "__main__":
    analiz_motoru()
