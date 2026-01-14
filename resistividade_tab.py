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

    st.header("⚡ Propriedades Elétricas — Resistividade")

    st.markdown(
        """
        **Subaba 1**  
        Upload da amostra elétrica → ajuste **V × I** → cálculo de resistividade  

        **Subaba 2**  
        PCA multivariada usando **apenas os parâmetros físicos calculados**
        """
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "electrical_samples" not in st.session_state:
        st.session_state.electrical_samples = {}

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Upload & Processamento",
        "📊 PCA — Elétrica"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        uploaded_files = st.file_uploader(
            "Upload dos arquivos elétricos (CSV / TXT)",
            type=["csv", "txt"],
            accept_multiple_files=True
        )

        thickness_um = st.number_input(
            "Espessura do filme (µm)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

        geometry = st.selectbox(
            "Geometria da medição",
            ["four_point_film", "bulk"]
        )

        process_clicked = st.button("▶ Processar amostras")

        if uploaded_files and process_clicked:
            for file in uploaded_files:

                if file.name in st.session_state.electrical_samples:
                    st.warning(f"{file.name} já foi processado.")
                    continue

                st.markdown(f"### 📄 Amostra: `{file.name}`")

                try:
                    result = process_resistivity(
                        file_like=file,
                        thickness_m=thickness_um * 1e-6,
                        geometry=geometry
                    )

                    # -----------------------------
                    # Gráfico I × V
                    # -----------------------------
                    st.pyplot(result["figure"])

                    # -----------------------------
                    # Resumo físico
                    # -----------------------------
                    summary = {
                        "Amostra": file.name,
                        "Resistência (Ω)": result["R_ohm"],
                        "Resistividade (Ω·m)": result["rho_ohm_m"],
                        "Condutividade (S/m)": result["sigma_S_m"],
                        "Classe": result["classe"],
                        "R²": result["fit"]["R2"],
                        "Espessura (µm)": thickness_um,
                    }

                    st.session_state.electrical_samples[file.name] = summary

                    st.success("✔ Amostra processada com sucesso")

                except Exception as e:
                    st.error("Erro ao processar a amostra")
                    st.exception(e)

        if st.session_state.electrical_samples:
            st.subheader("Resumo elétrico das amostras")
            st.dataframe(
                pd.DataFrame(st.session_state.electrical_samples.values()),
                use_container_width=True
            )

            if st.button("🗑 Limpar amostras"):
                st.session_state.electrical_samples = {}
                st.experimental_rerun()

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        if len(st.session_state.electrical_samples) < 2:
            st.info("Carregue ao menos duas amostras na subaba de processamento.")
            return

        df_pca = pd.DataFrame(st.session_state.electrical_samples.values())

        # Garante apenas dados numéricos
        numeric_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()

        st.subheader("Dados de entrada da PCA")
        st.dataframe(df_pca, use_container_width=True)

        feature_cols = st.multiselect(
            "Variáveis elétricas para PCA",
            options=numeric_cols,
            default=[
                "Resistividade (Ω·m)",
                "Condutividade (S/m)",
                "R²",
            ]
        )

        if len(feature_cols) < 2:
            st.warning("Selecione ao menos duas variáveis.")
            return

        X = df_pca[feature_cols].values
        labels = df_pca["Amostra"].values

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_ * 100

        # ---------------------------
        # BIPLOT PADRONIZADO
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

        ax.scatter(scores[:, 0], scores[:, 1], s=90, edgecolor="black")

        for i, label in enumerate(labels):
            ax.text(
                scores[i, 0] + 0.03,
                scores[i, 1] + 0.03,
                label,
                fontsize=9
            )

        scale = np.max(np.abs(scores)) * 0.85
        for i, var in enumerate(feature_cols):
            ax.arrow(
                0, 0,
                loadings[i, 0] * scale,
                loadings[i, 1] * scale,
                color="black",
                alpha=0.7,
                head_width=0.08,
                length_includes_head=True
            )
            ax.text(
                loadings[i, 0] * scale * 1.1,
                loadings[i, 1] * scale * 1.1,
                var,
                fontsize=9
            )

        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA — Propriedades Elétricas")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.subheader("Variância explicada")
        st.dataframe(pd.DataFrame({
            "Componente": ["PC1", "PC2"],
            "Variância (%)": explained.round(2)
        }))
