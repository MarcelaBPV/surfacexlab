# =========================================================
# PERFILOMETRIA — SurfaceXLab (VERSÃO FINAL CIENTÍFICA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# ESTATÍSTICA
# =========================================================
def analisar_estatistica(dados):

    dados = np.array(dados)

    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    erro = desvio / np.sqrt(len(dados))

    return media, desvio, erro


# =========================================================
# LEITURA
# =========================================================
def ler_arquivo(file):

    if file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)

        if any("Unnamed" in str(c) for c in df.columns):
            df = pd.read_excel(file, header=1)

    else:
        df = pd.read_csv(file)

    return df


# =========================================================
# DETECÇÃO DE COLUNAS
# =========================================================
def detectar_colunas(df):

    pa_col = None
    pq_col = None

    for c in df.columns:
        c_clean = str(c).strip().lower()

        if c_clean == "pa":
            pa_col = c

        elif c_clean == "pq":
            pq_col = c

    return pa_col, pq_col


# =========================================================
# EXTRAÇÃO
# =========================================================
def extrair_medidas(df, pa_col, pq_col):

    Pa = pd.to_numeric(df[pa_col], errors="coerce").dropna().values
    Pq = pd.to_numeric(df[pq_col], errors="coerce").dropna().values

    if len(Pa) > 10:
        Pa = Pa[:10]

    if len(Pq) > 10:
        Pq = Pq[:10]

    return Pa, Pq


# =========================================================
# CLASSIFICAÇÃO
# =========================================================
def classificar_rugosidade(pa):

    if pa < 0.1:
        return "Lisa"
    elif pa < 1:
        return "Moderada"
    else:
        return "Rugosa"


# =========================================================
# PCA BIPLOT
# =========================================================
def plot_pca_biplot(df):

    X = df.values
    labels = df.index
    features = df.columns

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T

    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    # pontos
    ax.scatter(scores[:,0], scores[:,1], s=90, edgecolor="black")

    for i, label in enumerate(labels):
        ax.text(scores[i,0], scores[i,1], label)

    # vetores
    scale = np.max(np.abs(scores)) * 0.8

    for i, var in enumerate(features):

        ax.arrow(
            0, 0,
            loadings[i,0]*scale,
            loadings[i,1]*scale,
            color="black",
            head_width=0.08
        )

        ax.text(
            loadings[i,0]*scale*1.1,
            loadings[i,1]*scale*1.1,
            var
        )

    ax.axhline(0, color="gray")
    ax.axvline(0, color="gray")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")

    ax.set_title("PCA — Perfilometria")

    ax.grid(alpha=0.3)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def plot_ra_rq_comparison(df):

    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    x = np.arange(len(df.index))
    width = 0.35

    ax.bar(x - width/2, df["Ra"], width, label="Ra")
    ax.bar(x + width/2, df["Rq"], width, label="Rq")

    ax.set_xlabel("Amostras")
    ax.set_ylabel("Rugosidade (µm)")
    ax.set_title("Comparação dos parâmetros de rugosidade")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index)

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    return fig

def render_perfilometria_tab():

    st.header("📏 Perfilometria")

    if "perfilometria_samples" not in st.session_state:
        st.session_state.perfilometria_samples = {}

    tabs = st.tabs(["📊 Upload & Análise", "🧠 PCA"])

    # =====================================================
    # SUBABA 1
    # =====================================================
    with tabs[0]:

        files = st.file_uploader(
            "Upload dos arquivos de perfilometria",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True
        )

        if not files:
            st.info("Envie os arquivos.")
            return

        resultados = {}

        for f in files:

            st.markdown(f"### 📄 {f.name}")

            df = ler_arquivo(f)

            pa_col, pq_col = detectar_colunas(df)

            if pa_col is None:
                pa_col = st.selectbox(f"Pa ({f.name})", df.columns, key=f"pa_{f.name}")

            if pq_col is None:
                pq_col = st.selectbox(f"Pq ({f.name})", df.columns, key=f"pq_{f.name}")

            Pa, Pq = extrair_medidas(df, pa_col, pq_col)

            if len(Pa) == 0:
                st.error("Erro nos dados")
                continue

            pa_mean, pa_std, _ = analisar_estatistica(Pa)
            pq_mean, pq_std, _ = analisar_estatistica(Pq)

            # métricas extras
            cv_pa = pa_std / pa_mean
            anisotropia = pa_std / pa_mean

            classe = classificar_rugosidade(pa_mean)

            diagnostico = classe

            if cv_pa > 0.3:
                diagnostico += " + heterogênea"

            if anisotropia > 0.2:
                diagnostico += " + anisotrópica"

            # métricas
            col1, col2 = st.columns(2)

            col1.metric("Pa (µm)", f"{pa_mean:.3f}")
            col2.metric("Pq (µm)", f"{pq_mean:.3f}")

            st.markdown("### 🧠 Diagnóstico")
            st.write(f"**Classe:** {classe}")
            st.write(f"**Diagnóstico:** {diagnostico}")

            resultados[f.name] = {
                "Ra": pa_mean,
                "Rq": pq_mean,
                "CV": cv_pa,
                "Anisotropia": anisotropia
            }

        if resultados:
            st.session_state.perfilometria_samples = resultados

            df_final = pd.DataFrame(resultados).T

            st.markdown("### 📊 Dataset consolidado")
            st.dataframe(df_final)

    # =====================================================
    # GRÁFICO COMPARATIVO
    # =====================================================
    st.markdown("### 📊 Comparação de Rugosidade (Ra vs Rq)")

    fig_bar = plot_ra_rq_comparison(df_final)

    st.pyplot(fig_bar)    

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with tabs[1]:

        if "perfilometria_samples" not in st.session_state:
            st.warning("Execute a análise primeiro")
            return

        df = pd.DataFrame(st.session_state.perfilometria_samples).T

        if df.shape[0] < 2:
            st.info("Mínimo de 2 amostras")
            return

        st.dataframe(df)

        fig = plot_pca_biplot(df)

        st.pyplot(fig)

        st.markdown("### 🔬 Interpretação")

        st.write("""
        - PC1 geralmente associado à rugosidade (Ra, Rq)
        - PC2 associado à heterogeneidade e anisotropia
        - Vetores próximos → correlação direta
        - Vetores opostos → relação inversa
        """)
