# =========================================================
# Raman Tab — SurfaceXLab (FINAL FUNCIONAL)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter1d

# =========================================================
# IMPORT SEGURO
# =========================================================
try:
    from raman_processing import process_raman_spectrum_with_groups
    RAMAN_OK = True
except:
    RAMAN_OK = False

# =========================================================
# IMPORT MAPPING
# =========================================================
from raman_mapping_tab import render_mapeamento_molecular_tab


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab(supabase=None):

    st.header("🧬 Análises Moleculares")

    subtabs = st.tabs([
        "📐 Processamento",
        "📊 PCA",
        "🗺️ Mapping"
    ])

    # =====================================================
    # PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        files = st.file_uploader(
            "Upload espectros Raman",
            type=["csv", "txt"],
            accept_multiple_files=True
        )

        if not RAMAN_OK:
            st.error("⚠️ raman_processing não encontrado")
            st.info("Apenas o mapping está disponível")
            return

        if files:

            for f in files:

                st.markdown(f"### {f.name}")

                try:
                    result = process_raman_spectrum_with_groups(f)
                except Exception as e:
                    st.error("Erro no processamento")
                    st.exception(e)
                    continue

                df = result["spectrum_df"]
                peaks = result["peaks_df"]

                st.pyplot(result["figures"]["raw"])
                st.pyplot(result["figures"]["baseline"])

                st.dataframe(peaks)

                st.session_state.raman_peaks[f.name] = peaks

    # =====================================================
    # PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.get("raman_peaks", {})) < 2:
            st.info("Carregue pelo menos 2 amostras")
            return

        df = pd.DataFrame(st.session_state.raman_peaks).T

        df = df.select_dtypes(include=[np.number]).fillna(0)

        X = StandardScaler().fit_transform(df.values)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        fig, ax = plt.subplots()

        ax.scatter(scores[:,0], scores[:,1])

        for i, name in enumerate(df.index):
            ax.text(scores[i,0], scores[i,1], name)

        st.pyplot(fig)

    # =====================================================
    # MAPPING
    # =====================================================
    with subtabs[2]:

        render_mapeamento_molecular_tab()
