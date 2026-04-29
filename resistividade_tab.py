# resistividade_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from resistividade_processing import process_resistivity


# =========================================================
# ABA RESISTIVIDADE ELÉTRICA
# =========================================================
def render_resistividade_tab(supabase=None):

    st.header("⚡ Propriedades Elétricas — Resistividade (4 Pontas)")

    st.markdown("""
    Medição elétrica via método de quatro pontas (Smits, 1958).
    
    O sistema realiza:
    - Ajuste linear V × I
    - Cálculo de resistividade
    - Sheet resistance
    - Diagnóstico físico automático
    - Classificação do material
    """)

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "electrical_samples" not in st.session_state:
        st.session_state.electrical_samples = {}

    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Elétrica"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos elétricos",
            type=["csv", "txt", "xls", "xlsx"],
            accept_multiple_files=True
        )

        thickness_um = st.number_input(
            "Espessura do filme (µm)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        geometry = "four_point_film"

        if st.button("▶ Processar amostras"):

            if not uploaded_files:
                st.warning("Nenhum arquivo selecionado.")
                return

            for file in uploaded_files:

                if file.name in st.session_state.electrical_samples:
                    st.warning(f"{file.name} já processado.")
                    continue

                st.markdown(f"---\n### 📄 Amostra: `{file.name}`")

                try:
                    result = process_resistivity(
                        file_like=file,
                        thickness_m=thickness_um * 1e-6,
                        geometry=geometry
                    )

                    # -----------------------------
                    # GRÁFICO
                    # -----------------------------
                    st.pyplot(result["figure"])

                    summary = result["summary"].copy()

                    summary["Amostra"] = file.name
                    summary["Espessura (µm)"] = thickness_um

                    st.session_state.electrical_samples[file.name] = summary

                    # -----------------------------
                    # DIAGNÓSTICO
                    # -----------------------------
                    st.markdown("### 🧠 Diagnóstico físico")

                    st.write(f"**Regime:** {summary['Regime']}")
                    st.write(f"**Classe:** {summary['Classe']}")

                    # -----------------------------
                    # MÉTRICAS
                    # -----------------------------
                    st.markdown("### 📊 Parâmetros elétricos")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Resistividade (Ω·m)",
                        f"{summary['Resistividade (Ω·m)']:.2e}"
                    )

                    col2.metric(
                        "Condutividade (S/m)",
                        f"{summary['Condutividade (S/m)']:.2e}"
                    )

                    col3.metric(
                        "Sheet Resistance (Ω/sq)",
                        f"{summary['Sheet Resistance (Ω/sq)']:.2e}"
                    )

                    # -----------------------------
                    # TABELA
                    # -----------------------------
                    st.dataframe(
                        pd.DataFrame([summary]).set_index("Amostra"),
                        use_container_width=True
                    )

                    st.success("✔ Processado com sucesso")

                except Exception as e:
                    st.error("Erro ao processar")
                    st.exception(e)

        # =====================================================
        # RESUMO GLOBAL
        # =====================================================
        if st.session_state.electrical_samples:

            st.markdown("---")
            st.subheader("📋 Resumo consolidado")

            df_all = pd.DataFrame(st.session_state.electrical_samples.values())

            st.dataframe(df_all, use_container_width=True)

            # =====================================================
            # EXPORTAÇÃO PARA ML / PCA (CORRIGIDO)
            # =====================================================
            df_ml = df_all.copy()

            # Conversão segura (PANDAS FIX)
            df_ml = df_ml.apply(lambda col: pd.to_numeric(col, errors="coerce"))

            # Remove colunas totalmente inválidas
            df_ml = df_ml.dropna(axis=1, how="all")

            # Define índice
            if "Amostra" in df_ml.columns:
                df_ml = df_ml.set_index("Amostra")

            # Preenche NaN para PCA
            df_ml = df_ml.fillna(0)

            st.session_state.electrical_features = df_ml

            if st.button("🗑 Limpar dados"):
                st.session_state.electrical_samples = {}
                st.session_state.electrical_features = None
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if "electrical_features" not in st.session_state or st.session_state.electrical_features is None:
            st.info("Sem dados para PCA.")
            return

        df_pca = st.session_state.electrical_features.copy()

        if df_pca.shape[0] < 2:
            st.info("Mínimo de 2 amostras.")
            return

        st.subheader("Dataset PCA")
        st.dataframe(df_pca)

        X = StandardScaler().fit_transform(df_pca.values)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        fig, ax = plt.subplots(figsize=(7,7), dpi=300)

        ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

        for i, label in enumerate(df_pca.index):
            ax.text(scores[i,0], scores[i,1], label)

        scale = np.max(np.abs(scores)) * 0.8

        for i, var in enumerate(df_pca.columns):

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

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Propriedades Elétricas")

        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")

        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
