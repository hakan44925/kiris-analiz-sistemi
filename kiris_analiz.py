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

# --- SIDEBAR ---
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

        # Bilinmeyenler: Mesnet Reaksiyonları (R1, R2...) ve Ankastre Momenti (MA)
        # SIRA: [R1, R2, ..., MA]
        ankastre_var mı = 3 in m_type
        n_mesnet = len(m_pos)
        n_vars = n_mesnet + (1 if ankastre_var mı else 0)
        
        A = np.zeros((n_vars, n_vars))
        B = np.zeros(n_vars)

        # 1. Denklemler: Toplam Fy = 0
        A[0, :n_mesnet] = 1
        B[0] = np.sum(ps) + np.sum(ws * (we - wb))

        # 2. Denklemler: Toplam Moment (Sistemin en soluna x=0'a göre)
        for i in range(n_mesnet):
            A[1, i] = m_pos[i]
        if ankastre_var mı:
            # MA (Ankastre Momenti) bilinmeyeni genellikle matrisin sonunda olur
            A[1, n_vars-1] = 1
        
        # Dış yüklerin x=0'a göre momenti
        B[1] = np.sum(ps * pk)
        for i in range(len(ws)):
            B[1] += (ws[i] * (we[i] - wb[i])) * ((wb[i] + we[i])/2)

        # 3. Denklemler: Mafsal Şartı (M_mafsal = 0)
        # Mafsalın SOLUNDAKİ kuvvetlerin mafsal noktasına göre momenti
        for i, maf_x in enumerate(mafsallar):
            row = 2 + i
            if row >= n_vars: break
            for j in range(n_mesnet):
                if m_pos[j] < maf_x:
                    A[row, j] = (maf_x - m_pos[j])
            if ankastre_var mı and m_pos[0] < maf_x:
                A[row, n_vars-1] = 1 # Ankastre momenti MA denkleme dahil
            
            # Mafsalın solundaki dış yüklerin momenti
            m_load = np.sum(ps[pk < maf_x] * (maf_x - pk[pk < maf_x]))
            for k in range(len(ws)):
                if wb[k] < maf_x:
                    payda = min(we[k], maf_x)
                    w_L = payda - wb[k]
                    m_load += (ws[k] * w_L) * (maf_x - (wb[k] + payda)/2)
            B[row] = m_load

        reak = np.linalg.solve(A, B)

        # --- DİYAGRAM HESAPLARI ---
        x = np.linspace(0, L, 1001)
        V, M = np.zeros_like(x), np.zeros_like(x)

        for i, xi in enumerate(x):
            v_val, m_val = 0, 0
            # Mesnet Reaksiyonları
            for j in range(n_mesnet):
                if xi >= m_pos[j] - 1e-9:
                    v_val += reak[j]
                    m_val += reak[j] * (xi - m_pos[j])
            # Ankastre Momenti Katkısı
            if ankastre_var mı and xi >= m_pos[0] - 1e-9:
                m_val -= reak[n_vars-1] # Reaksiyon momenti dengeleyici
            
            # Tekil Yükler
            for j in range(len(ps)):
                if xi >= pk[j] - 1e-9:
                    v_val -= ps[j]
                    m_val -= ps[j] * (xi - pk[j])
            
            # Yayılı Yükler
            for k in range(len(ws)):
                if xi > wb[k]:
                    w_eff_L = min(xi, we[k]) - wb[k]
                    v_val -= ws[k] * w_eff_L
                    m_val -= (ws[k] * w_eff_L) * (xi - (wb[k] + min(xi, we[k]))/2)
            
            V[i], M[i] = v_val, m_val

        # --- GÖRSELLEŞTİRME ---
        fig, axes = plt.subplots(4, 1, figsize=(11, 14))
        plt.subplots_adjust(hspace=0.6)

        # Şema (Sadeleştirilmiş)
        ax0 = axes[0]
        ax0.hlines(0, 0, L, color='black', lw=6)
        for p, t in zip(m_pos, m_type):
            if t == 3: ax0.vlines(p, -0.6, 0.6, color='black', lw=10)
            else: ax0.plot(p, -0.25, '^' if t==1 else 'o', ms=15, color='gray')
        if mafsallar.size > 0:
            ax0.scatter(mafsallar, [0]*len(mafsallar), color='white', edgecolor='black', s=100, zorder=5)
        ax0.axis('off')

        # Diyagramlar
        kritik_x = np.unique(np.concatenate(([0, L], m_pos, mafsallar, pk, wb, we)))
        titles = ["N (Normal Kuvvet)", "V (Kesme Kuvveti) - kN", "M (Eğilme Momenti) - kNm"]
        data_list = [np.zeros_like(x), V, M]
        colors = ['green', 'blue', 'red']

        for i, (ax, t, c, d) in enumerate(zip(axes[1:], titles, colors
