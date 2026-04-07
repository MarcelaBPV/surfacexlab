# =========================================================
# SurfaceXLab — Análise Integrada FINAL
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensiometria_processing import process_tensiometry
from resistividade_processing import process_resistivity
from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# PCA GLOBAL (CORRIGIDO)
# =========================================================
def run_pca_global(df):

    df = df.copy()

    if "Amostra" not in df.columns:
        df["Amostra"] = df.index.astype(str)

    X = df.select_dtypes(include=[np.number])

    # remove colunas de erro (melhora PCA)
    X = X.drop(columns=[c for c in X.columns if "erro" in c.lower()], errors="ignore")

    if X.shape[1] < 2:
        st.warning("Dados insuficientes para PCA.")
        return None

    X = X.fillna(0)

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    ax.scatter(scores[:,0], scores[:,1], s=100, edgecolor="black")

    labels = df["Amostra"].astype(str).values

    for i, label in enumerate(labels):
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

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("PCA Global — Integração Multimodal")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# DATASET GLOBAL (PADRONIZADO)
# =========================================================
def build_dataset():

    if "samples_unified" not in st.session_state:
        return None

    data = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        for tech, values in content.items():

            if isinstance(values, dict):
                row.update(values)

        data.append(row)

    if len(data) == 0:
        return None

    return pd.DataFrame(data)


# =========================================================
# PERFILOMETRIA (CORRIGIDO REAL)
# =========================================================
def process_profilometry(file):

    file.seek(0)

    try:
        df = pd.read_excel(file)
    except:
        file.seek(0)
        df = pd.read_csv(file)

    # manter apenas linhas numéricas
    df = df[df.iloc[:,0].astype(str).str.replace(",", "").str.isnumeric()]

    df["Pa"] = pd.to_numeric(df["Pa"], errors="coerce")
    df["Pq"] = pd.to_numeric(df["Pq"], errors="coerce")

    pa = df["Pa"].dropna()
    pq = df["Pq"].dropna()

    if len(pa) < 3:
        raise ValueError("Dados insuficientes")

    def calc(x):
        media = np.mean(x)
        desvio = np.std(x, ddof=1)
        erro = desvio / np.sqrt(len(x))
        return media, erro

    pa_m, pa_e = calc(pa)
    pq_m, pq_e = calc(pq)

    return {
        "Pa": float(pa_m),
        "Pq": float(pq_m),
        "Pa_erro": float(pa_e),
        "Pq_erro": float(pq_e)
    }


# =========================================================
# RESISTIVIDADE (CORRIGIDO)
# =========================================================
def process_resistivity_fixed(file):

    df = pd.read_csv(file, sep=";")

    df = df.apply(lambda col: col.astype(str).str.replace(",", "."))
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.dropna()

    V = df["CH1 Voltage"]
    I = df["CH1 Current"]

    # regressão linear (melhor que média)
    coef = np.polyfit(V, I, 1)
    slope = coef[0]

    resistencia = 1 / slope if slope != 0 else np.nan

    return {
        "Resistencia": float(resistencia)
    }


# =========================================================
# TENSIOMETRIA (CORRIGIDO)
# =========================================================
def process_tensiometry_fixed(file):

    df = pd.read_excel(file)

    ang = pd.to_numeric(df["angulo"], errors="coerce").dropna()

    media = np.mean(ang)
    desvio = np.std(ang, ddof=1)
    erro = desvio / np.sqrt(len(ang))

    return {
        "Angulo": float(media),
        "Angulo_erro": float(erro)
    }


# =========================================================
# TAB PRINCIPAL
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
# UPLOAD
# =========================================================
    with subtabs[0]:

        sample_id = st.text_input("ID da Amostra")

        technique = st.selectbox(
            "Técnica",
            ["Raman", "Resistividade", "Tensiometria", "Perfilometria"]
        )

        file = st.file_uploader(
            "Arquivo",
            type=["csv","txt","xls","xlsx","log"]
        )

        if st.button("Processar"):

            if not sample_id or not file:
                st.warning("Preencha tudo")
                return

            if sample_id not in st.session_state.samples_unified:
                st.session_state.samples_unified[sample_id] = {}

            try:

                if technique == "Raman":

                    result = process_raman_spectrum_with_groups(file)

                    peaks = result["peaks_df"]
                    fingerprint = peaks.groupby("chemical_group")["amplitude"].mean()

                    st.session_state.samples_unified[sample_id]["raman"] = fingerprint.to_dict()

                elif technique == "Resistividade":

                    result = process_resistivity_fixed(file)
                    st.session_state.samples_unified[sample_id]["eletrico"] = result

                elif technique == "Tensiometria":

                    result = process_tensiometry_fixed(file)
                    st.session_state.samples_unified[sample_id]["tensiometria"] = result

                elif technique == "Perfilometria":

                    result = process_profilometry(file)
                    st.session_state.samples_unified[sample_id]["perfilometria"] = result

                st.success("Processado com sucesso!")

            except Exception as e:
                st.error("Erro")
                st.exception(e)

        if st.session_state.samples_unified:
            st.json(st.session_state.samples_unified)

# =========================================================
# PCA
# =========================================================
    with subtabs[1]:

        df = build_dataset()

        if df is None or df.empty:
            st.warning("Sem dados")
            return

        st.dataframe(df)

        if df.shape[0] >= 2:

            fig = run_pca_global(df)

            if fig:
                st.pyplot(fig)
