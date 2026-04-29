# =========================================================
# Raman Tab — SurfaceXLab (FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# IMPORT DO MAPPING (COM PROTEÇÃO)
# =========================================================
try:
    from raman_mapping_tab import render_mapeamento_molecular_tab
except:
    def render_mapeamento_molecular_tab():
        st.warning("Módulo de mapeamento não encontrado.")


# =========================================================
# LEITURA DE ESPECTRO SIMPLES
# =========================================================
def read_raman(file):

    df = pd.read_csv(file, sep=r"\s+|\t+|;", engine="python")

    df.columns = df.columns.str.lower()

    if "wave" not in df.columns:
        df.columns = ["wave", "intensity"]

    df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    return df.dropna()


# =========================================================
# PLOT
# =========================================================
def plot_raman(df):

    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    ax.plot(df["wave"], df["intensity"], linewidth=1.5)

    ax.set_xlabel("Número de onda (cm⁻¹)")
    ax.set_ylabel("Intensidade")
    ax.set_title("Espectro Raman")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# EXTRAÇÃO DE PARÂMETROS (SIMPLES)
# =========================================================
def extract_basic_metrics(df):

    intensity_max = df["intensity"].max()
    peak_pos = df.loc[df["intensity"].idxmax(), "wave"]

    return {
        "Intensidade máxima": intensity_max,
        "Pico principal (cm⁻¹)": peak_pos
    }


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab():

    st.header("🧬 Espectroscopia Raman")

    if "raman_peaks" not in st.session_state:
        st.session_state.raman_peaks = {}

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📈 Espectros",
        "🧬 Mapping"
    ])

    # =====================================================
    # SUBABA 1 — ESPECTRO
    # =====================================================
    with subtabs[0]:

        files = st.file_uploader(
            "Upload arquivos Raman",
            type=["txt", "csv"],
            accept_multiple_files=True
        )

        if files:

            for f in files:

                st.markdown(f"### 📄 {f.name}")

                try:
                    df = read_raman(f)

                    fig = plot_raman(df)
                    st.pyplot(fig)

                    metrics = extract_basic_metrics(df)

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Intensidade máxima",
                        f"{metrics['Intensidade máxima']:.2f}"
                    )

                    col2.metric(
                        "Pico principal (cm⁻¹)",
                        f"{metrics['Pico principal (cm⁻¹)']:.2f}"
                    )

                    # salva no session_state
                    st.session_state.raman_peaks[f.name] = metrics

                except Exception as e:
                    st.error("Erro ao processar arquivo")
                    st.exception(e)

    # =====================================================
    # SUBABA 2 — MAPPING
    # =====================================================
    with subtabs[1]:

        render_mapeamento_molecular_tab()
