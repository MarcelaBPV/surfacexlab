# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PROCESSAMENTOS (já existentes)
from tensiometria_processing import process_tensiometry
from resistividade_processing import process_resistivity
from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca_global(df):

    X = df.select_dtypes(include=[np.number])

    if X.shape[1] < 2:
        st.warning("Poucas variáveis para PCA.")
        return

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

    for i, label in enumerate(df["Amostra"]):
        ax.text(scores[i,0], scores[i,1], label)

    scale = np.max(np.abs(scores))*0.8

    for i, var in enumerate(X.columns):

        ax.arrow(
            0,0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            head_width=0.05
        )

        ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var,
            fontsize=9
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA Global — Multimodal")

    return fig


# =========================================================
# BUILD DATASET POR AMOSTRA
# =========================================================
def build_dataset():

    if "samples_unified" not in st.session_state:
        return None

    data = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        # junta tudo
        for key, val in content.items():

            if isinstance(val, dict):
                row.update(val)

        data.append(row)

    return pd.DataFrame(data)


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Unificada de Amostras")

    if "samples_unified" not in st.session_state:
        st.session_state.samples_unified = {}

    subtabs = st.tabs([
        "📥 Upload & Processamento",
        "📊 PCA Global"
    ])

# =========================================================
# 📥 SUBABA 1 — UPLOAD ORGANIZADO
# =========================================================
    with subtabs[0]:

        st.subheader("Cadastro da Amostra")

        sample_id = st.text_input("ID da Amostra (ex: A1.5, B1.5)")

        technique = st.selectbox(
            "Tipo de Análise",
            [
                "Raman",
                "Resistividade",
                "Tensiometria",
                "Perfilometria"
            ]
        )

        file = st.file_uploader(
            "Upload arquivo",
            type=["csv","txt","xls","xlsx","log"]
        )

        process = st.button("Processar")

        if process and file and sample_id:

            if sample_id not in st.session_state.samples_unified:
                st.session_state.samples_unified[sample_id] = {}

            try:

                # =====================================
                # RAMAN
                # =====================================
                if technique == "Raman":

                    result = process_raman_spectrum_with_groups(file)

                    peaks = result["peaks_df"]

                    fingerprint = (
                        peaks.groupby("chemical_group")["amplitude"]
                        .mean()
                    )

                    st.session_state.samples_unified[sample_id]["raman"] = fingerprint.to_dict()

                    st.success("Raman processado")

                # =====================================
                # ELÉTRICO
                # =====================================
                elif technique == "Resistividade":

                    result = process_resistivity(file,1e-6,"four_point_film")

                    st.pyplot(result["figure"])

                    st.session_state.samples_unified[sample_id]["eletrico"] = result["summary"]

                    st.success("Resistividade processada")

                # =====================================
                # TENSIOMETRIA
                # =====================================
                elif technique == "Tensiometria":

                    result = process_tensiometry(
                        file,
                        {"water":70,"diiodomethane":50,"formamide":60},
                        0.5,
                        0.3
                    )

                    st.pyplot(result["figure"])

                    st.session_state.samples_unified[sample_id]["tensiometria"] = result["summary"]

                    st.success("Tensiometria processada")

                # =====================================
                # PERFILOMETRIA
                # =====================================
                elif technique == "Perfilometria":

                    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

                    st.dataframe(df)

                    # exemplo simples: média de altura
                    if "height" in df.columns:
                        roughness = df["height"].std()
                    else:
                        roughness = np.std(df.values)

                    st.session_state.samples_unified[sample_id]["perfilometria"] = {
                        "Rugosidade (std)": roughness
                    }

                    st.success("Perfilometria processada")

                # =====================================
                # SALVA NO SUPABASE
                # =====================================
                if supabase:

                    supabase.table("samples_unified").insert({
                        "sample_code": sample_id,
                        "technique": technique
                    }).execute()

            except Exception as e:

                st.error("Erro no processamento")
                st.exception(e)

        # =====================================================
        # PREVIEW
        # =====================================================
        if st.session_state.samples_unified:

            st.subheader("Amostras carregadas")

            st.json(st.session_state.samples_unified)


# =========================================================
# 📊 SUBABA 2 — PCA GLOBAL
# =========================================================
    with subtabs[1]:

        df = build_dataset()

        if df is None or df.empty:
            st.warning("Nenhuma amostra completa ainda.")
            return

        st.dataframe(df, use_container_width=True)

        if df.shape[0] >= 2:
            fig = run_pca_global(df)
            st.pyplot(fig)
