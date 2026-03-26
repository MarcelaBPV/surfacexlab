# analise_completa_amostras_tab.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# módulos existentes
from tensiometria_processing import process_tensiometry

# =========================================================
# LEITURA UNIVERSAL (resolve ParserError)
# =========================================================
def read_any_file(file):

    try:
        name = file.name.lower()

        if name.endswith(".csv"):
            try:
                return pd.read_csv(file)
            except:
                file.seek(0)
                return pd.read_csv(file, sep=";")

        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)

        elif name.endswith((".txt", ".log")):
            try:
                return pd.read_csv(file, sep=r"\s+|,|;", engine="python")
            except:
                file.seek(0)
                return pd.read_csv(file, sep="\t")

        else:
            st.warning(f"Formato não suportado: {file.name}")
            return None

    except Exception as e:
        st.error(f"Erro ao ler {file.name}")
        st.exception(e)
        return None


# =========================================================
# PROCESSAMENTO ELÉTRICO
# =========================================================
def process_iv(df):

    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    if df.shape[1] < 2:
        return {}

    V = df.iloc[:, 0].values
    I = df.iloc[:, 1].values

    if len(V) < 2:
        return {}

    slope = np.polyfit(V, I, 1)[0]

    resistivity = 1 / slope if slope != 0 else np.nan

    return {
        "Resistividade": resistivity
    }


# =========================================================
# PROCESSAMENTO PERFILOMETRIA
# =========================================================
def process_profilometry(df):

    df = df.apply(pd.to_numeric, errors="coerce")

    z = df.values.flatten()
    z = z[~np.isnan(z)]

    if len(z) == 0:
        return {}

    return {
        "Rugosidade (std)": float(np.std(z))
    }


# =========================================================
# BUILD DATASET (com triplicata)
# =========================================================
def build_dataset():

    if "samples_unified" not in st.session_state:
        return None

    data = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        for key, val in content.items():

            # =========================
            # TENSIOMETRIA (triplicata)
            # =========================
            if key == "tensiometria" and isinstance(val, list):

                df_rep = pd.DataFrame(val)

                mean_vals = df_rep.mean(numeric_only=True)
                std_vals = df_rep.std(numeric_only=True)

                for col in mean_vals.index:
                    row[f"{col} (mean)"] = mean_vals[col]
                    row[f"{col} (std)"] = std_vals[col]

            # =========================
            # OUTROS
            # =========================
            elif isinstance(val, dict):
                row.update(val)

        data.append(row)

    return pd.DataFrame(data)


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca(df):

    df = df.copy()

    if "Amostra" not in df.columns:
        return

    features = df.drop(columns=["Amostra"])

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    if features.shape[1] < 2:
        st.warning("Dados insuficientes para PCA")
        return

    X = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7,7))

    ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

    for i, label in enumerate(df["Amostra"]):
        ax.text(scores[i,0], scores[i,1], label)

    ax.set_title("PCA Global — Análise Completa")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# MAIN TAB
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    if "samples_unified" not in st.session_state:
        st.session_state.samples_unified = {}

    subtabs = st.tabs([
        "📥 Upload & Processamento",
        "📊 PCA Global"
    ])

# =========================================================
# SUBABA 1
# =========================================================
    with subtabs[0]:

        sample_id = st.text_input("ID da amostra (ex: A1.5)")

        technique = st.selectbox(
            "Tipo de análise",
            ["Raman", "Resistividade", "Tensiometria", "Perfilometria"]
        )

        files = st.file_uploader(
            "Upload arquivos",
            type=["csv","xlsx","xls","txt","log"],
            accept_multiple_files=True
        )

        if files and sample_id:

            if sample_id not in st.session_state.samples_unified:
                st.session_state.samples_unified[sample_id] = {}

            for file in files:

                st.write(f"Processando: {file.name}")

                df = read_any_file(file)

                if df is None:
                    continue

                try:

                    # =====================================================
                    # RESISTIVIDADE
                    # =====================================================
                    if technique == "Resistividade":

                        result = process_iv(df)

                        st.session_state.samples_unified[sample_id]["resistividade"] = result

                        st.success("✔ Resistividade processada")

                    # =====================================================
                    # PERFILOMETRIA
                    # =====================================================
                    elif technique == "Perfilometria":

                        result = process_profilometry(df)

                        st.session_state.samples_unified[sample_id]["perfilometria"] = result

                        st.success("✔ Perfilometria processada")

                    # =====================================================
                    # TENSIOMETRIA (TRIPLICATA)
                    # =====================================================
                    elif technique == "Tensiometria":

                        result = process_tensiometry(
                            file,
                            {"water":70,"diiodomethane":50,"formamide":60},
                            0.5,
                            0.3
                        )

                        st.pyplot(result["figure"])

                        summary = result["summary"]

                        if "tensiometria" not in st.session_state.samples_unified[sample_id]:
                            st.session_state.samples_unified[sample_id]["tensiometria"] = []

                        st.session_state.samples_unified[sample_id]["tensiometria"].append(summary)

                        st.success(f"✔ Replicata adicionada ({len(st.session_state.samples_unified[sample_id]['tensiometria'])})")

                except Exception as e:

                    st.error("Erro no processamento")
                    st.exception(e)

        if st.session_state.samples_unified:

            st.subheader("Dados consolidados")

            st.write(st.session_state.samples_unified)

            df_all = build_dataset()

            if df_all is not None:
                st.dataframe(df_all, use_container_width=True)

# =========================================================
# SUBABA 2 PCA
# =========================================================
    with subtabs[1]:

        df_all = build_dataset()

        if df_all is None or len(df_all) < 2:
            st.info("Necessário ao menos 2 amostras")
            return

        st.dataframe(df_all)

        run_pca(df_all)
