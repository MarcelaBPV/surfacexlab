# =========================================================
# pca_tab.py
# SurfaceXLab — PCA Multimodal
# CORRIGIDO DEFINITIVAMENTE
# REPRODUÇÃO VISUAL IGUAL AO PAPER
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


# =========================================================
# TAB PCA
# =========================================================
def render_pca_tab():

    st.header("📊 PCA Multimodal")

    st.markdown("""
    Integração multimodal de parâmetros
    físico-químicos.
    """)

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================
    uploaded = st.file_uploader(

        "Upload matriz PCA (.xlsx, .xls ou .ods)",

        type=[
            "xlsx",
            "xls",
            "ods"
        ]
    )

    if uploaded is None:

        st.info(
            "Faça upload da matriz experimental."
        )

        return

    # =====================================================
    # LEITURA
    # =====================================================
    try:

        if uploaded.name.endswith(".xlsx"):

            df_raw = pd.read_excel(uploaded)

        elif uploaded.name.endswith(".xls"):

            df_raw = pd.read_excel(uploaded)

        elif uploaded.name.endswith(".ods"):

            df_raw = pd.read_excel(
                uploaded,
                engine="odf"
            )

        else:

            st.error(
                "Formato não suportado."
            )

            return

    except Exception as e:

        st.error(
            "Erro na leitura do arquivo."
        )

        st.exception(e)

        return

    # =====================================================
    # VISUALIZAÇÃO
    # =====================================================
    st.subheader(
        "📋 Tabela Importada"
    )

    st.dataframe(
        df_raw,
        use_container_width=True
    )

    # =====================================================
    # PCA
    # =====================================================
    run_pca(df_raw)


# =========================================================
# EXTRAI MÉDIA
# =========================================================
def extract_mean(value):

    if pd.isna(value):

        return np.nan

    value = str(value)

    value = value.replace(",", ".")

    value = value.replace("−", "-")

    value = value.replace("±", "+-")

    value = value.replace(" ", "")

    try:

        return float(
            value.split("+-")[0]
        )

    except:

        return np.nan


# =========================================================
# PCA
# =========================================================
def run_pca(df_raw):

    try:

        # =================================================
        # PRIMEIRA COLUNA
        # =================================================
        col0 = df_raw.columns[0]

        df_raw = df_raw.rename(
            columns={col0: "Variavel"}
        )

        # =================================================
        # REMOVE LINHAS INDESEJADAS
        # =================================================
        df_raw = df_raw[
            ~df_raw["Variavel"]
            .astype(str)
            .str.contains(
                "Temp",
                case=False,
                na=False
            )
        ]

        # =================================================
        # NORMALIZA NOMES DAS COLUNAS
        # =================================================
        new_cols = []

        counter = {}

        for col in df_raw.columns:

            col = str(col).strip()

            col = col.replace(" ", "")

            if col in counter:

                counter[col] += 1

                col = f"{col}_{counter[col]}"

            else:

                counter[col] = 0

            new_cols.append(col)

        df_raw.columns = new_cols

        # =================================================
        # DEBUG
        # =================================================
        st.subheader("📌 Colunas Detectadas")

        st.write(df_raw.columns.tolist())

        # =================================================
        # COLUNAS AUTOMÁTICAS
        # =================================================
        sample_columns = []

        for col in df_raw.columns:

            if col == "Variavel":

                continue

            sample_columns.append(col)

        # =================================================
        # LABELS
        # =================================================
        labels = []

        for col in sample_columns:

            base = col.split("_")[0]

            labels.append(base)

        # =================================================
        # MATRIZ
        # =================================================
        variables = []

        matrix = []

        for _, row in df_raw.iterrows():

            var_name = str(
                row["Variavel"]
            )

            variables.append(var_name)

            values = []

            for col in sample_columns:

                val = extract_mean(

                    row.get(col, np.nan)

                )

                values.append(val)

            matrix.append(values)

        # =================================================
        # MATRIZ FINAL
        # =================================================
        X = np.array(matrix).T

        X = np.nan_to_num(X)

        # =================================================
        # NORMALIZAÇÃO
        # =================================================
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # =================================================
        # PCA
        # =================================================
        pca = PCA(
            n_components=2
        )

        scores = pca.fit_transform(
            X_scaled
        )

        loadings = pca.components_.T

        explained = (
            pca.explained_variance_ratio_ * 100
        )

        # =================================================
        # CORREÇÃO VISUAL
        # INVERTE SOMENTE PC1
        # =================================================

        scores[:, 0] = -scores[:, 0]

        loadings[:, 0] = -loadings[:, 0]

        # =================================================
        # FIGURA
        # =================================================
        fig, ax = plt.subplots(

            figsize=(8, 5),

            dpi=600
        )

        fig.patch.set_facecolor("white")

        ax.set_facecolor("white")

        # =================================================
        # SCORES
        # =================================================
        for i in range(len(scores)):

            ax.scatter(

                scores[i, 0],

                scores[i, 1],

                color="black",

                s=20,

                zorder=3
            )

            ax.text(

                scores[i, 0] + 0.07,

                scores[i, 1] + 0.04,

                labels[i],

                fontsize=6,

                color="blue",

                fontweight="bold"
            )

        # =================================================
        # LOADINGS
        # =================================================
        scale = 2.5

        for i, var in enumerate(variables):

            x = loadings[i, 0] * scale

            y = loadings[i, 1] * scale

            ax.arrow(

                0,
                0,

                x,
                y,

                color="forestgreen",

                linewidth=1.0,

                head_width=0.08,

                length_includes_head=True,

                zorder=2
            )

            ax.text(

                x * 1.32,

                y * 1.32,

                var,

                color="red",

                fontsize=6,

                fontweight="bold"
            )

        # =================================================
        # EIXOS
        # =================================================
        ax.axhline(

            0,

            color="gray",

            linewidth=1.5
        )

        ax.axvline(

            0,

            color="gray",

            linewidth=1.5
        )

        # =================================================
        # LABELS
        # =================================================
        ax.set_xlabel(

            f"PC1 ({explained[0]:.1f}%)",

            fontsize=6
        )

        ax.set_ylabel(

            f"PC2 ({explained[1]:.1f}%)",

            fontsize=6
        )

        # =================================================
        # ESTILO
        # =================================================
        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.tick_params(

            axis="both",

            labelsize=6,

            width=1.5,

            length=6
        )

        ax.grid(False)

        # =================================================
        # LIMITES
        # =================================================
        margin = 0.7

        ax.set_xlim(

            scores[:,0].min() - margin,

            scores[:,0].max() + margin
        )

        ax.set_ylim(

            scores[:,1].min() - margin,

            scores[:,1].max() + margin
        )

        # =================================================
        # MOSTRAR FIGURA
        # =================================================
        st.divider()

        st.subheader(
            "📈 PCA Scores + Loadings"
        )

        st.pyplot(fig)

        # =================================================
        # VARIÂNCIA
        # =================================================
        col1, col2 = st.columns(2)

        col1.metric(

            "PC1",

            f"{explained[0]:.1f}%"
        )

        col2.metric(

            "PC2",

            f"{explained[1]:.1f}%"
        )

        # =================================================
        # SCORES
        # =================================================
        scores_df = pd.DataFrame({

            "Amostra": labels,

            "PC1": np.round(
                scores[:,0], 4
            ),

            "PC2": np.round(
                scores[:,1], 4
            )
        })

        st.subheader(
            "📌 Scores"
        )

        st.dataframe(

            scores_df,

            use_container_width=True
        )

        # =================================================
        # LOADINGS
        # =================================================
        loadings_df = pd.DataFrame({

            "Variavel": variables,

            "PC1": np.round(
                loadings[:,0], 4
            ),

            "PC2": np.round(
                loadings[:,1], 4
            )
        })

        st.subheader(
            "📌 Loadings"
        )

        st.dataframe(

            loadings_df,

            use_container_width=True
        )

        # =================================================
        # SALVAR FIGURA
        # =================================================
        fig.savefig(

            "pca_multimodal_corrigido.png",

            dpi=600,

            bbox_inches="tight"
        )

        # =================================================
        # DOWNLOAD
        # =================================================
        with open(

            "pca_multimodal_corrigido.png",

            "rb"

        ) as f:

            st.download_button(

                "📥 Download Figura PCA",

                f,

                file_name="PCA_multimodal_corrigido.png"
            )

    except Exception as e:

        st.error(
            "Erro no processamento PCA"
        )

        st.exception(e)
