# =========================================================
# resistividade_tab.py
# SurfaceXLab — Electrical Module FINAL
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from resistividade_processing import (

    process_resistivity,

    build_acid_group_plot,

    build_alkaline_group_plot
)

# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_resistividade_tab(supabase=None):

    st.header(
        "⚡ Caracterização Elétrica Superficial"
    )

    st.markdown("""
    Plataforma científica para caracterização
    elétrica interfacial aplicada à engenharia
    de superfícies e integração multimodal.
    """)

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "electrical_samples" not in st.session_state:

        st.session_state[
            "electrical_samples"
        ] = {}

    if "electrical_features" not in st.session_state:

        st.session_state[
            "electrical_features"
        ] = None

    # =====================================================
    # SUBTABS
    # =====================================================
    subtabs = st.tabs([

        "📐 Processamento Elétrico",

        "📊 PCA Elétrico"
    ])

    # =====================================================
    # SUBTAB 1
    # =====================================================
    with subtabs[0]:

        st.subheader(
            "📐 Upload e Processamento"
        )

        # =================================================
        # UPLOAD
        # =================================================
        uploaded_files = st.file_uploader(

            "Upload arquivos elétricos",

            type=[

                "csv",

                "txt",

                "xlsx",

                "xls"
            ],

            accept_multiple_files=True
        )

        # =================================================
        # ESPESSURA
        # =================================================
        thickness_um = st.number_input(

            "Espessura da amostra (µm)",

            min_value=0.01,

            value=1.00,

            step=0.1
        )

        # =================================================
        # MODO
        # =================================================
        measurement_mode_ui = st.selectbox(

            "Modo experimental",

            [

                "Voltage Sweep (SMU)",

                "Current Sweep (4-Point Probe)"
            ]
        )

        # =================================================
        # CONVERSÃO
        # =================================================
        if "Voltage" in measurement_mode_ui:

            measurement_mode = (
                "voltage_sweep"
            )

        else:

            measurement_mode = (
                "current_sweep"
            )

        # =================================================
        # PROCESSAMENTO
        # =================================================
        if st.button("▶ Processar amostras"):

            if not uploaded_files:

                st.warning(
                    "Selecione ao menos um arquivo."
                )

                return

            # =============================================
            # DICIONÁRIO
            # =============================================
            processed = {}

            # =============================================
            # LOOP
            # =============================================
            for file in uploaded_files:

                st.markdown("---")

                st.markdown(
                    f"## 📄 {file.name}"
                )

                try:

                    # =====================================
                    # PROCESSA
                    # =====================================
                    result = process_resistivity(

                        file_like=file,

                        thickness_m=
                            thickness_um * 1e-6,

                        sample_name=
                            file.name,

                        mode=
                            measurement_mode
                    )

                    # =====================================
                    # ARMAZENA
                    # =====================================
                    processed[
                        file.name
                    ] = result

                    summary = result["summary"]

                    # =====================================
                    # MÉTRICAS
                    # =====================================
                    st.subheader(
                        "📊 Propriedades Elétricas"
                    )

                    c1, c2, c3, c4, c5 = st.columns(5)

                    c1.metric(

                        "ρ (Ω·m)",

                        f"{summary['Resistivity_Ohm_m']:.2e}"
                    )

                    c2.metric(

                        "σ (S/m)",

                        f"{summary['Conductivity_S_m']:.2e}"
                    )

                    c3.metric(

                        "R (Ω)",

                        f"{summary['Resistance_Ohm']:.2e}"
                    )

                    c4.metric(

                        "R²",

                        f"{summary['R_squared']:.4f}"
                    )

                    c5.metric(

                        "Histerese",

                        f"{summary['Hysteresis']:.2e}"
                    )

                    # =====================================
                    # DATAFRAME
                    # =====================================
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
                    st.session_state[
                        "electrical_samples"
                    ][file.name] = (
                        feature_df.iloc[0].to_dict()
                    )

                    st.success(
                        "✔ Amostra processada"
                    )

                except Exception as e:

                    st.error(
                        "⚠ Falha no processamento"
                    )

                    st.warning(str(e))

            # =============================================
            # FIGURA 28 — ÁCIDO
            # =============================================
            acid_exists = any(

                (

                    k.upper().startswith("A")

                    and

                    "_D" in k.upper()

                )

                for k in processed.keys()
            )

            if acid_exists:

                st.markdown("---")

                st.subheader(
                    "📈 Figura 28 — Grupo Ácido"
                )

                fig_acid = build_acid_group_plot(
                    processed
                )

                st.pyplot(fig_acid)

                st.caption(
                    """
                    Figura 28 – Curvas I×V das amostras
                    de FC200 submetidas ao tratamento
                    ácido em diferentes tempos de
                    exposição, evidenciando alterações
                    progressivas da resposta elétrica
                    superficial associadas aos processos
                    de reorganização interfacial e
                    passivação química.
                    """
                )

            # =============================================
            # FIGURA 29 — ALCALINO
            # =============================================
            alkaline_exists = any(

                (

                    k.upper().startswith("B")

                    and

                    "_D" in k.upper()

                )

                for k in processed.keys()
            )

            if alkaline_exists:

                st.markdown("---")

                st.subheader(
                    "📈 Figura 29 — Grupo Alcalino"
                )

                fig_alk = build_alkaline_group_plot(
                    processed
                )

                st.pyplot(fig_alk)

                st.caption(
                    """
                    Figura 29 – Curvas I×V das amostras
                    de FC200 submetidas ao tratamento
                    alcalino em diferentes tempos de
                    exposição, mostrando comportamento
                    elétrico mais estável e menor
                    variabilidade interfacial em
                    comparação ao grupo tratado
                    em meio ácido.
                    """
                )

        # =================================================
        # DATASET CONSOLIDADO
        # =================================================
        if st.session_state.electrical_samples:

            st.markdown("---")

            st.subheader(
                "📋 Dataset Consolidado"
            )

            df_all = pd.DataFrame(

                st.session_state[
                    "electrical_samples"
                ].values()
            )

            st.dataframe(

                df_all,

                use_container_width=True
            )

            # =============================================
            # PCA
            # =============================================
            df_pca = df_all.copy()

            df_pca = df_pca.select_dtypes(
                include=[np.number]
            )

            df_pca = df_pca.fillna(0)

            st.session_state[
                "electrical_features"
            ] = df_pca

            # =============================================
            # EXPORTA
            # =============================================
            csv = df_all.to_csv(index=False)

            st.download_button(

                label="⬇ Exportar Dataset",

                data=csv,

                file_name=
                    "electrical_dataset.csv",

                mime="text/csv"
            )

            # =============================================
            # RESET
            # =============================================
            if st.button(
                "🗑 Limpar Dataset"
            ):

                st.session_state[
                    "electrical_samples"
                ] = {}

                st.session_state[
                    "electrical_features"
                ] = None

                st.rerun()

    # =====================================================
    # SUBTAB PCA
    # =====================================================
    with subtabs[1]:

        st.subheader(
            "📊 PCA — Caracterização Elétrica"
        )

        if (

            st.session_state[
                "electrical_features"
            ] is None
        ):

            st.info(
                "Nenhum dataset disponível."
            )

            return

        # =================================================
        # DATASET
        # =================================================
        df_pca = (

            st.session_state[
                "electrical_features"
            ].copy()
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

            pca.explained_variance_ratio_

            * 100
        )

        # =================================================
        # FIGURA PCA
        # =================================================
        fig, ax = plt.subplots(

            figsize=(7,6),

            dpi=600
        )

        # =================================================
        # SCATTER
        # =================================================
        ax.scatter(

            scores[:,0],

            scores[:,1],

            s=60,

            edgecolor='black'
        )

        labels = list(

            st.session_state[
                "electrical_samples"
            ].keys()
        )

        for i, label in enumerate(labels):

            ax.text(

                scores[i,0] + 0.03,

                scores[i,1] + 0.03,

                label,

                fontsize=10
            )

        # =================================================
        # LOADINGS
        # =================================================
        scale = (

            np.max(np.abs(scores))

            * 0.7
        )

        for i, feature in enumerate(df_pca.columns):

            ax.arrow(

                0,
                0,

                loadings[i,0] * scale,

                loadings[i,1] * scale,

                head_width=0.04,

                linewidth=1.2
            )

            ax.text(

                loadings[i,0] * scale * 1.05,

                loadings[i,1] * scale * 1.05,

                feature,

                fontsize=9
            )

        # =================================================
        # EIXOS
        # =================================================
        ax.axhline(

            y=0,

            linewidth=0.8
        )

        ax.axvline(

            x=0,

            linewidth=0.8
        )

        # =================================================
        # REMOVE GRID
        # =================================================
        ax.grid(False)

        # =================================================
        # REMOVE BORDAS
        # =================================================
        ax.spines['top'].set_visible(False)

        ax.spines['right'].set_visible(False)

        # =================================================
        # LABELS
        # =================================================
        ax.set_xlabel(

            f"PC1 [{explained[0]:.1f}%]",

            fontsize=12
        )

        ax.set_ylabel(

            f"PC2 [{explained[1]:.1f}%]",

            fontsize=12
        )

        plt.tight_layout()

        st.pyplot(fig)

        # =================================================
        # VARIÂNCIA
        # =================================================
        st.subheader(
            "📈 Variância Explicada"
        )

        explained_df = pd.DataFrame({

            "Componente":
                ["PC1", "PC2"],

            "Variância (%)":
                explained.round(2)
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
