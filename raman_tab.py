# =========================================================
# Raman Tab — SurfaceXLab (ARQUITETURA FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.ndimage import gaussian_filter1d

# =========================================================
# IMPORTS
# =========================================================
try:
    from raman_processing import process_raman_spectrum_with_groups
    RAMAN_OK = True
except:
    RAMAN_OK = False

from raman_mapping_tab import render_mapeamento_molecular_tab


# =========================================================
# LEITOR UNIVERSAL
# =========================================================
def read_any_file(file):

    name = file.name.lower()

    if name.endswith(".xlsx"):
        df = pd.read_excel(file)

    elif name.endswith(".txt") or name.endswith(".log"):
        df = pd.read_csv(file, sep=None, engine="python")

    elif name.endswith(".csv"):
        df = pd.read_csv(file)

    else:
        raise ValueError("Formato não suportado")

    return df


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab(supabase=None):

    st.header("🧬 Análises Moleculares")

    subtabs = st.tabs([
        "📐 Espectros (Paper)",
        "🗺️ Mapping Raman",
        "📊 PCA"
    ])

    # =====================================================
    # SUBABA 1 — PAPER RAMAN
    # =====================================================
    with subtabs[0]:

        st.subheader("📐 Processamento — Dados do Paper")

        files = st.file_uploader(
            "Upload dados Raman (xlsx, csv, txt, log)",
            accept_multiple_files=True
        )

        if not RAMAN_OK:
            st.error("⚠️ raman_processing não encontrado")
            return

        if files:

            for f in files:

                st.markdown(f"### {f.name}")

                try:
                    df = read_any_file(f)
                except Exception as e:
                    st.error("Erro na leitura")
                    st.exception(e)
                    continue

                try:
                    result = process_raman_spectrum_with_groups(f)
                except Exception as e:
                    st.error("Erro no processamento")
                    st.exception(e)
                    continue

                # plots
                st.pyplot(result["figures"]["raw"])
                st.pyplot(result["figures"]["baseline"])

                # peaks
                st.dataframe(result["peaks_df"])

                st.session_state.raman_peaks[f.name] = result["peaks_df"]

    # =====================================================
    # SUBABA 2 — MAPPING (PACIENTE)
    # =====================================================
    with subtabs[1]:

        st.subheader("🗺️ Mapping Raman — Paciente")

        st.info("Upload do arquivo de mapeamento (ex: paciente 180)")

        render_mapeamento_molecular_tab()

    # =====================================================
    # SUBABA 3 — PCA
    # =====================================================
    with subtabs[2]:

        st.subheader("📊 Análise PCA")

        if len(st.session_state.get("raman_peaks", {})) < 2:
            st.warning("Carregue pelo menos 2 espectros do paper")
            return

        df = pd.DataFrame(st.session_state.raman_peaks).T

        df = df.select_dtypes(include=[np.number]).fillna(0)

        try:
            X = StandardScaler().fit_transform(df.values)

            pca = PCA(n_components=2)
            scores = pca.fit_transform(X)

        except Exception as e:
            st.error("Erro no PCA")
            st.exception(e)
            return

        fig, ax = plt.subplots()

        ax.scatter(scores[:, 0], scores[:, 1])

        for i, name in enumerate(df.index):
            ax.text(scores[i, 0], scores[i, 1], name)

        ax.set_title("PCA — Raman")

        st.pyplot(fig)
