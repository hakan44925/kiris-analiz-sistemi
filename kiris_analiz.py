import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Hakan Çırak - Kiriş Analiz Portalı", layout="wide")

st.title("🏗️ Profesyonel Kiriş Analiz Sistemi")
st.markdown("---")

# --- PARAMETRELER ---
L = st.sidebar.number_input("Kiriş Boyu (m)", value=11.0)
m_pos = np.array([0.0, 8.0]) # Mesnetler
ps, pk = np.array([20.0]), np.array([11.0]) # Tekil Yük
ms, mk = np.array([-150.0]), np.array([11.0]) # Tekil Moment
ws, wb, we = np.array([40.0]), np.array([0.0]), np.array([8.0]) # Yayılı Yük

def analiz_motoru():
    try:
        # 1. REAKSİYON HESABI (m1=0'a göre moment)
        W_total = ws[0] * (we[0] - wb[0])
        W_arm = (we[0] + wb[0]) / 2
        # Sum(M_0) = 0 => R2*8 - W*4 - P*11 + M_tekil = 0
        # R2*8 = (40*8)*4 + 20*11 - (-150)
        R2y = (W_total * W_arm + ps[0] * pk[0] - ms[0]) / m_pos[1]
        R1y = (W_total + ps[0]) - R2y

        # 2. SÜREKSİZLİK İÇİN ÖZEL NOKTA YÖNETİMİ
        # Momentin olduğu mk noktasında iki değer tanımlıyoruz: Öncesi ve Sonrası
        x_list = []
        v_list = []
        m_list = []
        
        # 0'dan L'ye çok ince adımlarla tara
        steps = np.linspace(0, L, 5000)
        # Kritik noktaları (mesnet, yük, moment) listeye dahil et
        check_points = np.sort(np.unique(np.concatenate(([0, L], m_pos, pk, mk))))
        
        for xi in steps:
            # Kesme Hesabı
            v = R1y if xi >= m_pos[0] else 0
            if xi >= m_pos[1]: v += R2y
            if xi >= pk[0]: v -= ps[0]
            if wb[0] <= xi <= we[0]: v -= ws[0] * (xi - wb[0])
            elif xi > we[0]: v -= ws[0] * (we[0] - wb[0])
            
            # Moment Hesabı (Tekil moment eklenmeden önceki saf hali)
            m = 0
            if xi >= m_pos[0]: m += R1y * (xi - m_pos[0])
            if xi >= m_pos[1]: m += R2y * (xi - m_pos[1])
            if xi >= pk[0]: m -= ps[0] * (xi - pk[0])
            if xi > wb[0]:
                active_w = min(xi, we[0]) - wb[0]
                m -= (ws[0] * active_w) * (xi - (wb[0] + active_w/2))
            
            # Tekil momentin olduğu tam noktada sıçramayı manuel yap
            if xi < mk[0]:
                x_list.append(xi); v_list.append(v); m_list.append(m)
            elif xi == mk[0]:
                # Momentten hemen önce
                x_list.append(xi); v_list.append(v); m_list.append(m)
                # Momentten hemen sonra (Sıfıra kapanış)
                x_list.append(xi); v_list.append(v); m_list.append(m + ms[0])
            else:
                x_list.append(xi); v_list.append(v); m_list.append(m + ms[0])

        # --- GÖRSELLEŞTİRME ---
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        
        # Kesme Diyagramı
        ax[0].plot(x_list, v_list, color='blue', lw=2)
        ax[0].fill_between(x_list, v_list, color='blue', alpha=0.1)
        ax[0].set_title(f"Kesme Kuvveti (V) - Max: {round(max(v_list),1)} kN")
        ax[0].grid(True, alpha=0.3)

        # Moment Diyagramı (Ucu dikey kapatan versiyon)
        ax[1].plot(x_list, m_list, color='red', lw=2)
        ax[1].fill_between(x_list, m_list, color='red', alpha=0.1)
        ax[1].set_title(f"Eğilme Momenti (M) - Mesnet: {round(m_list[np.argmin(np.abs(np.array(x_list)-8))],1)} kNm")
        ax[1].invert_yaxis() # Mühendislik standardı
        ax[1].grid(True, alpha=0.3)

        st.pyplot(fig)
        st.write(f"**Hesaplanan Reaksiyonlar:** R1: {round(R1y,1)} kN | R2: {round(R2y,1)} kN")

    except Exception as e:
        st.error(f"Sistem Hatası: {e}")

analiz_motoru()
