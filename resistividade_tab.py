# =========================================================
# resistividade_tab.py
# SurfaceXLab — Electrical Module (F200 READY)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from resistividade_processing import process_resistivity


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_resistividade_tab(supabase=None):

    st.header("⚡ Caracterização Elétrica Superficial")

    st.markdown("""
    Plataforma para caracterização elétrica interfacial via método
    de quatro pontas aplicada à análise de materiais metálicos,
    superfícies tratadas quimicamente e filmes funcionais.

    O pipeline realiza automaticamente:

    - processamento I–V robusto;
    - detecção da região ôhmica;
    - regressão linear científica;
    - cálculo de resistividade;
    - análise de regimes interfaciais;
    - extração de features elétricas;
    - PCA multivariado.
    """)

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "electrical_samples" not in st.session_state:
        st.session_state.electrical_samples = {}

    if "electrical_features" not in st.session_state:
        st.session_state.electrical_features = None

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📐 Processamento Elétrico",
        "📊 PCA Elétrico"
    ])

    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        st.subheader("📐 Upload e Processamento")

        uploaded_files = st.file_uploader(
            "Upload arquivos elétricos",
            type=["csv", "txt", "xlsx", "xls"],
            accept_multiple_files=True
        )

        thickness_um = st.number_input(
            "Espessura da amostra (µm)",
            min_value=0.01,
            value=1.00,
            step=0.1
        )

        sample_group = st.selectbox(
            "Grupo experimental",
            [
                "F200 - Ácido",
                "F200 - Alcalino",
                "F200 - Controle",
                "Nanotubos",
                "Outro"
            ]
        )

        # =================================================
        # PROCESSAMENTO
        # =================================================
        if st.button("▶ Processar amostras"):

            if not uploaded_files:

                st.warning("Selecione ao menos um arquivo.")
                return

            for file in uploaded_files:

                st.markdown("---")

                st.markdown(
                    f"## 📄 {file.name}"
                )

                try:

                    # =====================================
                    # PIPELINE PRINCIPAL
                    # =====================================
                    result = process_resistivity(
                        file_like=file,
                        thickness_m=thickness_um * 1e-6,
                        sample_name=file.name
                    )

                    summary = result["summary"]

                    # =====================================
                    # FIGURA
                    # =====================================
                    st.pyplot(result["figure"])

                    # =====================================
                    # DIAGNÓSTICO
                    # =====================================
                    st.subheader("🧠 Diagnóstico Interfacial")

                    col1, col2 = st.columns(2)

                    col1.info(
                        f"""
                        **Regime**

                        {summary['Regime']}
                        """
                    )

                    col2.info(
                        f"""
                        **Classe**

                        {summary['Classe']}
                        """
                    )

                    # =====================================
                    # MÉTRICAS
                    # =====================================
                    st.subheader(
                        "📊 Propriedades Elétricas"
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    m1.metric(
                        "ρ (Ω·m)",
                        f"{summary['Resistivity_Ohm_m']:.2e}"
                    )

                    m2.metric(
                        "σ (S/m)",
                        f"{summary['Conductivity_S_m']:.2e}"
                    )

                    m3.metric(
                        "Rs (Ω/sq)",
                        f"{summary['Sheet_Resistance_Ohm_sq']:.2e}"
                    )

                    m4.metric(
                        "R²",
                        f"{summary['R_squared']:.4f}"
                    )

                    # =====================================
                    # FEATURES
                    # =====================================
                    st.subheader(
                        "📈 Features Elétricas"
                    )

                    feature_df = pd.DataFrame(
                        [summary]
                    )

                    st.dataframe(
                        feature_df,
                        use_container_width=True
                    )

                    # =====================================
                    # SESSION STORAGE
                    # =====================================
                    feature_df["Sample_Group"] = sample_group
                    feature_df["Thickness_um"] = thickness_um

                    st.session_state.electrical_samples[
                        file.name
                    ] = feature_df.iloc[0].to_dict()

                    st.success(
                        "✔ Amostra processada"
                    )

                except Exception as e:

                    st.error(
                        "Erro durante processamento"
                    )

                    st.exception(e)

        # =================================================
        # CONSOLIDADO
        # =================================================
        if st.session_state.electrical_samples:

            st.markdown("---")

            st.subheader(
                "📋 Dataset Consolidado"
            )

            df_all = pd.DataFrame(
                st.session_state.electrical_samples.values()
            )

            st.dataframe(
                df_all,
                use_container_width=True
            )

            # =============================================
            # FEATURES PCA
            # =============================================
            df_pca = df_all.copy()

            # remove texto
            df_pca = df_pca.select_dtypes(
                include=[np.number]
            )

            df_pca = df_pca.fillna(0)

            st.session_state.electrical_features = df_pca

            # =============================================
            # EXPORT
            # =============================================
            csv = df_all.to_csv(index=False)

            st.download_button(
                label="⬇ Exportar Dataset",
                data=csv,
                file_name="electrical_dataset.csv",
                mime="text/csv"
            )

            # =============================================
            # RESET
            # =============================================
            if st.button(
                "🗑 Limpar Dataset"
            ):

                st.session_state.electrical_samples = {}

                st.session_state.electrical_features = None

                st.rerun()

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        st.subheader(
            "📊 PCA — Caracterização Elétrica"
        )

        if (
            st.session_state.electrical_features
            is None
        ):

            st.info(
                "Nenhum dataset disponível."
            )

            return

        df_pca = (
            st.session_state.electrical_features.copy()
        )

        if len(df_pca) < 2:

            st.warning(
                "Mínimo de 2 amostras."
            )

            return

        # =================================================
        # PCA
        # =================================================
        X = StandardScaler().fit_transform(
            df_pca.values
        )

        pca = PCA(n_components=2)

        scores = pca.fit_transform(X)

        loadings = pca.components_.T

        explained = (
            pca.explained_variance_ratio_ * 100
        )

        # =================================================
        # FIGURA PCA
        # =================================================
        fig, ax = plt.subplots(
            figsize=(7,7),
            dpi=300
        )

        ax.scatter(
            scores[:,0],
            scores[:,1],
            s=120,
            edgecolor="black"
        )

        # labels
        labels = list(
            st.session_state.electrical_samples.keys()
        )

        for i, label in enumerate(labels):

            ax.text(
                scores[i,0],
                scores[i,1],
                label,
                fontsize=9
            )

        # loading vectors
        scale = np.max(np.abs(scores)) * 0.7

        for i, feature in enumerate(df_pca.columns):

            ax.arrow(
                0,
                0,
                loadings[i,0] * scale,
                loadings[i,1] * scale,
                head_width=0.05
            )

            ax.text(
                loadings[i,0] * scale * 1.1,
                loadings[i,1] * scale * 1.1,
                feature,
                fontsize=8
            )

        ax.set_xlabel(
            f"PC1 ({explained[0]:.1f}%)"
        )

        ax.set_ylabel(
            f"PC2 ({explained[1]:.1f}%)"
        )

        ax.set_title(
            "PCA — Propriedades Elétricas"
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        st.pyplot(fig)

        # =================================================
        # VARIÂNCIA
        # =================================================
        st.subheader(
            "📈 Variância Explicada"
        )

        explained_df = pd.DataFrame({

            "Componente": ["PC1", "PC2"],

            "Variância (%)": explained.round(2)

        })

        st.dataframe(
            explained_df,
            use_container_width=True
        )

        # =================================================
        # LOADINGS
        # =================================================
        st.subheader(
            "🧬 Importância das Variáveis"
        )

        loading_df = pd.DataFrame(
            loadings,
            columns=["PC1", "PC2"],
            index=df_pca.columns
        )

        st.dataframe(
            loading_df,
            use_container_width=True
        )
