# tensiometria_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensiometria_processing import process_tensiometry


# =========================================================
# ABA TENSIOMETRIA
# =========================================================
def render_tensiometria_tab(supabase=None):

    st.header("💧 Físico-Mecânica — Tensiometria Óptica")

    if "tensiometry_samples" not in st.session_state:
        st.session_state.tensiometry_samples = {}

    subtabs = st.tabs([
        "📐 Processamento",
        "📊 PCA"
    ])

# =========================================================
# SUBABA 1 — PROCESSAMENTO
# =========================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload arquivos do tensiômetro (.LOG)",
            type=["log","txt"],
            accept_multiple_files=True
        )

        st.subheader("Ângulos médios para OWRK")

        col1,col2,col3 = st.columns(3)

        with col1:
            theta_water = st.number_input("Água (°)", value=70.0)

        with col2:
            theta_diiodo = st.number_input("Diiodometano (°)", value=50.0)

        with col3:
            theta_formamide = st.number_input("Formamida (°)", value=60.0)

        theta_by_liquid = {
            "water":theta_water,
            "diiodomethane":theta_diiodo,
            "formamide":theta_formamide
        }

        st.subheader("Parâmetros Raman")

        col1,col2 = st.columns(2)

        with col1:
            id_ig = st.number_input("ID/IG", value=0.5)

        with col2:
            i2d_ig = st.number_input("I2D/IG", value=0.3)

        if uploaded_files:

            for file in uploaded_files:

                if file.name in st.session_state.tensiometry_samples:
                    st.warning(f"{file.name} já processado.")
                    continue

                st.markdown(f"### Amostra: {file.name}")

                try:

                    result = process_tensiometry(
                        file,
                        theta_by_liquid,
                        id_ig,
                        i2d_ig
                    )

                    st.pyplot(result["figure"])

                    summary = result["summary"]
                    summary["Amostra"] = file.name

                    st.dataframe(
                        pd.DataFrame([summary]).set_index("Amostra"),
                        use_container_width=True
                    )

                    st.session_state.tensiometry_samples[file.name] = summary

                    st.success("✔ Amostra processada")

                except Exception as e:

                    st.error("Erro no processamento")
                    st.exception(e)

        if st.session_state.tensiometry_samples:

            st.subheader("Resumo das amostras")

            df_all = pd.DataFrame(
                st.session_state.tensiometry_samples.values()
            )

            df_all["Amostra"] = list(st.session_state.tensiometry_samples.keys())

            st.dataframe(df_all,use_container_width=True)

            if st.button("Limpar dados"):
                st.session_state.tensiometry_samples = {}
                st.experimental_rerun()

# =========================================================
# FUNÇÃO PCA
# =========================================================

    def run_pca(df_pca):

        if "Amostra" not in df_pca.columns:
            df_pca = df_pca.reset_index()

        feature_cols = [
            "Rrms (mm)",
            "q* (°)",
            "Energia superficial (mJ/m²)",
            "Componente dispersiva (mJ/m²)",
            "Componente polar (mJ/m²)",
            "ID/IG",
            "I2D/IG"
        ]

        feature_cols = [c for c in feature_cols if c in df_pca.columns]

        if len(feature_cols) < 2:
            st.error("Dados insuficientes para PCA")
            st.write("Colunas disponíveis:", df_pca.columns.tolist())
            return

        X = df_pca[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

        labels = df_pca["Amostra"]

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)

        scores = pca.fit_transform(X_scaled)

        loadings = pca.components_.T

        explained = pca.explained_variance_ratio_ * 100

        fig,ax = plt.subplots(figsize=(7,7),dpi=300)

        ax.scatter(scores[:,0],scores[:,1],s=100,edgecolor="black")

        for i,label in enumerate(labels):
            ax.text(scores[i,0]+0.03,scores[i,1]+0.03,label)

        scale = np.max(np.abs(scores))*0.8

        for i,var in enumerate(feature_cols):

            ax.arrow(
                0,0,
                loadings[i,0]*scale,
                loadings[i,1]*scale,
                head_width=0.05,
                color="black"
            )

            ax.text(
                loadings[i,0]*scale*1.1,
                loadings[i,1]*scale*1.1,
                var
            )

        ax.axhline(0,color="gray",lw=0.5)
        ax.axvline(0,color="gray",lw=0.5)

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

        ax.set_title("PCA — Tensiometria")

        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({
            "Componente":["PC1","PC2"],
            "Variância (%)":explained.round(2)
        }))

# =========================================================
# SUBABA 2 — PCA
# =========================================================

    with subtabs[1]:

        option = st.radio(
            "Fonte dos dados",
            [
                "Usar amostras processadas",
                "Carregar arquivo (CSV / XLS / XLSX)"
            ]
        )

        if option == "Usar amostras processadas":

            if len(st.session_state.tensiometry_samples) < 2:
                st.info("Processar ao menos duas amostras.")
                return

            df_pca = pd.DataFrame(
                st.session_state.tensiometry_samples.values()
            )

            df_pca["Amostra"] = list(st.session_state.tensiometry_samples.keys())

            run_pca(df_pca)

        else:

            uploaded_pca = st.file_uploader(
                "Upload dados PCA",
                type=["csv","xls","xlsx"]
            )

            if uploaded_pca:

                if uploaded_pca.name.endswith(".csv"):
                    df_pca = pd.read_csv(uploaded_pca)
                else:
                    df_pca = pd.read_excel(uploaded_pca)

                st.dataframe(df_pca,use_container_width=True)

                run_pca(df_pca)
