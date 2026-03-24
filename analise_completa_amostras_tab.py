# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PROCESSAMENTOS
from tensiometria_processing import process_tensiometry
from resistividade_processing import process_resistivity
from raman_processing import process_raman_spectrum_with_groups

# MAPEAMENTO RAMAN
from mapeamento_molecular_tab import (
    read_mapping,
    process_spectrum,
    plot_heatmap,
    run_pca as run_pca_raman,
    plot_raman_groups_annotated
)


# =========================================================
# CLASSIFICADOR INTELIGENTE
# =========================================================
def classify_file(file):

    name = file.name.lower()

    # por nome
    if "tensio" in name:
        return "tensiometria"

    if "resist" in name or "iv" in name:
        return "eletrico"

    if "perfil" in name:
        return "perfilometria"

    if "raman" in name:
        return "raman"

    # fallback por conteúdo
    try:

        if name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        cols = [c.lower() for c in df.columns]

        if any("voltage" in c or "v" == c for c in cols):
            return "eletrico"

        if any("theta" in c or "angle" in c for c in cols):
            return "tensiometria"

        if any("height" in c or "z" in c for c in cols):
            return "perfilometria"

    except:
        pass

    return "desconhecido"


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca_global(df):

    df = df.copy()

    if "Amostra" not in df.columns:
        df["Amostra"] = df.index.astype(str)

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.warning("Poucas variáveis para PCA.")
        return None

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

    for i, label in enumerate(df["Amostra"]):
        ax.text(scores[i,0], scores[i,1], label)

    scale = np.max(np.abs(scores)) * 0.8

    for i, var in enumerate(X.columns):

        ax.arrow(
            0, 0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            head_width=0.05,
            color="black"
        )

        ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var,
            fontsize=9
        )

    ax.axhline(0,color="gray",lw=0.5)
    ax.axvline(0,color="gray",lw=0.5)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

    ax.set_title("PCA Global — Multimodal")

    return fig


# =========================================================
# DATASET GLOBAL
# =========================================================
def build_global_dataset():

    dfs = []

    if "tensiometry_samples" in st.session_state:
        df = pd.DataFrame(st.session_state.tensiometry_samples.values())
        df["Amostra"] = list(st.session_state.tensiometry_samples.keys())
        dfs.append(df)

    if "electrical_features" in st.session_state:
        dfs.append(st.session_state.electrical_features)

    if "raman_peaks" in st.session_state:
        df_r = pd.DataFrame(st.session_state.raman_peaks).T.fillna(0)
        df_r["Amostra"] = df_r.index
        dfs.append(df_r)

    if len(dfs) == 0:
        return None

    df_final = dfs[0]

    for df in dfs[1:]:
        df_final = pd.merge(df_final, df, on="Amostra", how="outer")

    return df_final


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    st.info("""
    Upload universal:
    - Resistividade (I–V)
    - Tensiometria (OWRK)
    - Perfilometria
    - Raman
    """)

    subtabs = st.tabs([
        "📥 Upload & Processamento",
        "📊 PCA Global",
        "🧬 Mapeamento Raman"
    ])


# =========================================================
# 📥 UPLOAD
# =========================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload múltiplo",
            type=["csv","txt","xls","xlsx","log"],
            accept_multiple_files=True
        )

        if uploaded_files:

            for file in uploaded_files:

                tipo = classify_file(file)

                st.markdown(f"### {file.name} → {tipo}")

                try:

                    # ==========================
                    # TENSIOMETRIA
                    # ==========================
                    if tipo == "tensiometria":

                        result = process_tensiometry(
                            file,
                            {"water":70,"diiodomethane":50,"formamide":60},
                            0.5,
                            0.3
                        )

                        summary = result["summary"]
                        summary["Amostra"] = file.name

                        if "tensiometry_samples" not in st.session_state:
                            st.session_state.tensiometry_samples = {}

                        st.session_state.tensiometry_samples[file.name] = summary

                        st.pyplot(result["figure"])


                    # ==========================
                    # ELÉTRICO
                    # ==========================
                    elif tipo == "eletrico":

                        result = process_resistivity(
                            file,
                            1e-6,
                            "four_point_film"
                        )

                        summary = result["summary"]
                        summary["Amostra"] = file.name

                        if "electrical_features" not in st.session_state:
                            st.session_state.electrical_features = pd.DataFrame()

                        st.session_state.electrical_features = pd.concat([
                            st.session_state.electrical_features,
                            pd.DataFrame([summary])
                        ])

                        st.pyplot(result["figure"])


                    # ==========================
                    # RAMAN
                    # ==========================
                    elif tipo == "raman":

                        result = process_raman_spectrum_with_groups(file)

                        peaks = result["peaks_df"]

                        fingerprint = (
                            peaks.groupby("chemical_group")["amplitude"]
                            .mean()
                        )

                        if "raman_peaks" not in st.session_state:
                            st.session_state.raman_peaks = {}

                        st.session_state.raman_peaks[file.name] = fingerprint

                        st.success("Raman processado")


                    # ==========================
                    # PERFILOMETRIA
                    # ==========================
                    elif tipo == "perfilometria":

                        df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

                        st.dataframe(df)

                        if "profilometry" not in st.session_state:
                            st.session_state.profilometry = {}

                        st.session_state.profilometry[file.name] = df

                        st.success("Perfilometria carregada")


                    else:
                        st.warning("Tipo não identificado")

                except Exception as e:
                    st.error("Erro no processamento")
                    st.exception(e)


# =========================================================
# 📊 PCA GLOBAL
# =========================================================
    with subtabs[1]:

        df = build_global_dataset()

        if df is None:
            st.warning("Nenhum dado processado ainda.")
            return

        st.dataframe(df, use_container_width=True)

        if df.shape[0] >= 2:

            fig = run_pca_global(df)

            if fig:
                st.pyplot(fig)


# =========================================================
# 🧬 MAPEAMENTO
# =========================================================
    with subtabs[2]:

        file = st.file_uploader("Upload mapping", type=["csv","txt"])

        if not file:
            return

        df = read_mapping(file)

        st.pyplot(plot_heatmap(df))

        spectra = []

        for y_val, group in df.groupby("y"):

            x,y = process_spectrum(
                group["wave"].values,
                group["intensity"].values
            )

            spectra.append({
                "y":y_val,
                "wave":x,
                "intensity":y
            })

        st.pyplot(run_pca_raman(spectra))

        fig,tables = plot_raman_groups_annotated(spectra)

        st.pyplot(fig)

        for g,t in tables.items():
            st.markdown(f"### {g}")
            st.dataframe(t)
