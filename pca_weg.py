# =========================================================
# PCA_WEG.PY
# SurfaceXLab — PCA FC200 WEG
# STREAMLIT CLOUD COMPATÍVEL
# VERSÃO FINAL CORRIGIDA
# =========================================================

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# =========================================================
# FUNÇÃO PRINCIPAL
# =========================================================

def render_pca_weg():

    st.header("📊 PCA Multimodal — FC200 WEG")

    st.markdown(
        """
        Integração multimodal de parâmetros
        físico-químicos experimentais.
        """
    )

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================

    uploaded = st.file_uploader(

        "Upload matriz PCA WEG (.xlsx, .xls ou .ods)",

        type=[
            "xlsx",
            "xls",
            "ods"
        ],

        key="pca_weg_uploader"
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
    # MOSTRA TABELA
    # =====================================================

    st.subheader(
        "📋 Tabela Importada"
    )

    st.dataframe(
        df_raw,
        use_container_width=True
    )

    # =====================================================
    # EXECUTA PCA
    # =====================================================

    run_pca_weg(df_raw)


# =========================================================
# FUNÇÃO LIMPEZA
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

def run_pca_weg(df_raw):

    try:

        # =================================================
        # PRIMEIRA COLUNA
        # =================================================

        col0 = df_raw.columns[0]

        df_raw = df_raw.rename(
            columns={col0: "Variavel"}
        )

        # =================================================
        # REMOVE TEMPERATURA
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
        # REMOVE CONDUTIVIDADE
        # =================================================

        df_raw = df_raw[
            ~df_raw["Variavel"]
            .astype(str)
            .str.contains(
                "Condut",
                case=False,
                na=False
            )
        ]

        # =================================================
        # NORMALIZA COLUNAS
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

        st.subheader(
            "📌 Colunas Detectadas"
        )

        st.write(
            df_raw.columns.tolist()
        )

        # =================================================
        # AMOSTRAS
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

            labels.append(col)

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

        # =================================================
        # IMPUTAÇÃO
        # =================================================

        imputer = SimpleImputer(
            strategy="mean"
        )

        X = imputer.fit_transform(X)

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
        # CORREÇÃO ESPELHAMENTO
        # =================================================

        scores[:,0] = -scores[:,0]

        loadings[:,0] = -loadings[:,0]

        # =================================================
        # ESCALA LOADINGS
        # =================================================

        scale = 3.5

        # =================================================
        # FIGURA
        # =================================================

        plt.rcParams["font.family"] = "Arial"

        fig, ax = plt.subplots(

            figsize=(10,6),

            dpi=600
        )

        fig.patch.set_facecolor("white")

        ax.set_facecolor("white")

        # =================================================
        # SCORES
        # =================================================

        for i in range(len(scores)):

            ax.scatter(

                scores[i,0],

                scores[i,1],

                color="black",

                s=55,

                zorder=3
            )

            ax.text(

                scores[i,0] + 0.04,

                scores[i,1] + 0.02,

                labels[i],

                fontsize=9,

                color="blue",

                fontweight="bold"
            )

        # =================================================
        # LOADINGS
        # =================================================

        for i, var in enumerate(variables):

            x = loadings[i,0] * scale

            y = loadings[i,1] * scale

            ax.arrow(

                0,
                0,

                x,
                y,

                color="forestgreen",

                linewidth=2.2,

                head_width=0.08,

                length_includes_head=True,

                zorder=2
            )

            ax.text(

                x * 1.15,

                y * 1.15,

                var,

                color="red",

                fontsize=11,

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

            fontsize=13
        )

        ax.set_ylabel(

            f"PC2 ({explained[1]:.1f}%)",

            fontsize=13
        )

        # =================================================
        # ESTILO
        # =================================================

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.tick_params(

            axis="both",

            labelsize=11,

            width=1.5,

            length=7
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
        # LAYOUT
        # =================================================

        plt.tight_layout()

        # =================================================
        # SALVAR
        # =================================================

        fig.savefig(

            "PCA_WEG_FC200.tiff",

            dpi=600,

            bbox_inches="tight",

            transparent=True
        )

        fig.savefig(

            "PCA_WEG_FC200.png",

            dpi=600,

            bbox_inches="tight"
        )

        # =================================================
        # MOSTRAR
        # =================================================

        st.pyplot(fig)

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

        st.subheader("📌 Scores")

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

        st.subheader("📌 Loadings")

        st.dataframe(
            loadings_df,
            use_container_width=True
        )

        # =================================================
        # EXPORTA CSV
        # =================================================

        scores_df.to_csv(

            "export_scores.csv",

            index=False
        )

        loadings_df.to_csv(

            "export_loadings.csv",

            index=False
        )

        # =================================================
        # DOWNLOADS STREAMLIT
        # =================================================

        st.divider()

        with open(
            "PCA_WEG_FC200.png",
            "rb"
        ) as f:

            st.download_button(

                label="📥 Download PCA PNG",

                data=f,

                file_name="PCA_WEG_FC200.png",

                mime="image/png"
            )

        with open(
            "PCA_WEG_FC200.tiff",
            "rb"
        ) as f:

            st.download_button(

                label="📥 Download PCA TIFF",

                data=f,

                file_name="PCA_WEG_FC200.tiff",

                mime="image/tiff"
            )

        with open(
            "export_scores.csv",
            "rb"
        ) as f:

            st.download_button(

                label="📥 Download Scores",

                data=f,

                file_name="export_scores.csv",

                mime="text/csv"
            )

        with open(
            "export_loadings.csv",
            "rb"
        ) as f:

            st.download_button(

                label="📥 Download Loadings",

                data=f,

                file_name="export_loadings.csv",

                mime="text/csv"
            )

    except Exception as e:

        st.error(
            "Erro no processamento PCA"
        )

        st.exception(e)
