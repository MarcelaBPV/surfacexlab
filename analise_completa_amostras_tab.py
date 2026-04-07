# -- coding: utf-8 --

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
# PCA GLOBAL
# =========================================================
def run_pca_global(df):

    df = df.copy()

    # garante coluna de identificação
    if "Amostra" not in df.columns:
        df["Amostra"] = df.index.astype(str)

    # seleciona apenas dados numéricos
    X = df.select_dtypes(include=[np.number])

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
# CONSTRUIR DATASET GLOBAL
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
# PROCESSAMENTO PERFILOMETRIA (ROBUSTO)
# =========================================================
def process_profilometry(file):

    file.seek(0)

    try:
        df = pd.read_excel(file)
    except:
        file.seek(0)
        df = pd.read_csv(file)

    # remover linhas tipo "Média", "Desvio", etc
    df = df[df.iloc[:,0].astype(str).str.replace(',', '').str.isnumeric()]

    # converter colunas
    df["Pa"] = pd.to_numeric(df["Pa"], errors="coerce")
    df["Pq"] = pd.to_numeric(df["Pq"], errors="coerce")

    pa = df["Pa"].dropna()
    pq = df["Pq"].dropna()

    if len(pa) < 3 or len(pq) < 3:
        raise ValueError("Dados insuficientes de Pa/Pq")

    # estatística correta
    def calc(x):
        media = np.mean(x)
        desvio = np.std(x, ddof=1)
        erro = desvio / np.sqrt(len(x))
        return media, erro

    pa_media, pa_erro = calc(pa)
    pq_media, pq_erro = calc(pq)

    return {
        "Pa": float(pa_media),
        "Pq": float(pq_media),
        "Pa_erro": float(pa_erro),
        "Pq_erro": float(pq_erro)
    }

# =========================================================
# 📥 SUBABA 1
# =========================================================
    with subtabs[0]:

        st.subheader("Cadastro e Upload por Técnica")

        sample_id = st.text_input("ID da Amostra (ex: A1.5, B1.5)")

        technique = st.selectbox(
            "Selecione a técnica",
            ["Raman", "Resistividade", "Tensiometria", "Perfilometria"]
        )

        file = st.file_uploader(
            "Upload do arquivo",
            type=["csv","txt","xls","xlsx","log"]
        )

        if st.button("Processar"):

            if not sample_id or not file:
                st.warning("Preencha todos os campos.")
                return

            # garante estrutura
            if sample_id not in st.session_state.samples_unified:
                st.session_state.samples_unified[sample_id] = {}

            try:

                # =========================
                # RAMAN
                # =========================
                if technique == "Raman":

                    result = process_raman_spectrum_with_groups(file)

                    peaks = result["peaks_df"]

                    fingerprint = peaks.groupby("chemical_group")["amplitude"].mean()

                    st.session_state.samples_unified[sample_id]["raman"] = fingerprint.to_dict()

                    st.success("Raman processado")

                # =========================
                # ELÉTRICO
                # =========================
                elif technique == "Resistividade":

                    result = process_resistivity(file, 1e-6, "four_point_film")

                    st.pyplot(result["figure"])

                    st.session_state.samples_unified[sample_id]["eletrico"] = result["summary"]

                    st.success("Resistividade processada")

                # =========================
                # TENSIOMETRIA
                # =========================
                elif technique == "Tensiometria":

                    result = process_tensiometry(
                        file,
                        {"water":70,"diiodomethane":50,"formamide":60},
                        0.5,
                        0.3
                    )

                    st.pyplot(result["figure"])

                    st.session_state.samples_unified[sample_id]["tensiometria"] = result["summary"]

                    st.success("Tensiometria processada")

                # =========================
                # PERFILOMETRIA
                # =========================
                elif technique == "Perfilometria":

                    result = process_profilometry(file)

                    st.session_state.samples_unified[sample_id]["perfilometria"] = result

                    st.success(f"Rugosidade = {result['Rugosidade (std)']:.4f}")

                # =========================
                # SUPABASE
                # =========================
                if supabase:
                    supabase.table("samples_unified").insert({
                        "sample_code": sample_id,
                        "technique": technique
                    }).execute()

            except Exception as e:

                st.error("Erro no processamento")
                st.exception(e)

                # remove amostra vazia
                if st.session_state.samples_unified[sample_id] == {}:
                    del st.session_state.samples_unified[sample_id]

        # =========================
        # VISUALIZAÇÃO
        # =========================
        if st.session_state.samples_unified:

            st.subheader("Amostras carregadas")

            st.json(st.session_state.samples_unified)


# =========================================================
# 📊 SUBABA 2 — PCA
# =========================================================
    with subtabs[1]:

        df = build_dataset()

        if df is None or df.empty:
            st.warning("Nenhuma amostra completa ainda.")
            return

        st.dataframe(df, use_container_width=True)

        if df.shape[0] >= 2:

            fig = run_pca_global(df)

            if fig:
                st.pyplot(fig)
X = df.select_dtypes(include=[np.number])

# remover colunas de erro (opcional)
X = X.drop(columns=[col for col in X.columns if "erro" in col], errors="ignore")                
