# =========================================================
# PERFILOMETRIA TAB — SurfaceXLab (PA/PQ + PCA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =========================================================
# FUNÇÃO ESTATÍSTICA
# =========================================================
def analisar_estatistica(dados):

    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    erro = desvio / np.sqrt(len(dados))

    erro_arred = round(erro, 2)
    valor_arred = round(media, 2)

    valor_real = f"{valor_arred} ± {erro_arred}"

    return media, desvio, erro, erro_arred, valor_arred, valor_real


# =========================================================
# LEITURA INTELIGENTE
# =========================================================
def ler_arquivo(file):

    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, delimiter="\t")


# =========================================================
# DETECÇÃO DE COLUNAS
# =========================================================
def detectar_colunas(df):

    pa_col = None
    pq_col = None

    for c in df.columns:
        c_lower = c.lower()

        if "pa" in c_lower:
            pa_col = c

        if "pq" in c_lower:
            pq_col = c

    return pa_col, pq_col


# =========================================================
# PCA
# =========================================================
def executar_pca(df):

    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])

    return df_pca, pca.explained_variance_ratio_


def plot_pca(df_pca, labels):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(df_pca["PC1"], df_pca["PC2"])

    for i, txt in enumerate(labels):
        ax.annotate(txt, (df_pca["PC1"][i], df_pca["PC2"][i]))

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Análise de Componentes Principais (PCA)")

    ax.grid(True)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_perfilometria_tab():

    st.subheader("📏 Perfilometria de Superfície")
    st.caption("Análise estatística de rugosidade + PCA")

    st.divider()

    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([
        "📊 Estatística",
        "🧠 PCA"
    ])

    # =====================================================
    # SUBABA 1 — ESTATÍSTICA
    # =====================================================
    with subtabs[0]:

        files = st.file_uploader(
            "📂 Envie arquivos (csv, txt, log, xls, xlsx)",
            type=["csv", "txt", "log", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if not files:
            st.info("Envie um ou mais arquivos para iniciar.")
        else:

            resultados = {}

            for f in files:

                st.markdown(f"### 📄 {f.name}")

                try:
                    df = ler_arquivo(f)
                    st.dataframe(df.head())

                    # detectar colunas
                    pa_col, pq_col = detectar_colunas(df)

                    if pa_col is None:
                        pa_col = st.selectbox(
                            f"Selecione coluna Pa ({f.name})",
                            df.columns,
                            key=f"pa_{f.name}"
                        )

                    if pq_col is None:
                        pq_col = st.selectbox(
                            f"Selecione coluna Pq ({f.name})",
                            df.columns,
                            key=f"pq_{f.name}"
                        )

                    # dados
                    Pa = pd.to_numeric(df[pa_col], errors="coerce").dropna().values
                    Pq = pd.to_numeric(df[pq_col], errors="coerce").dropna().values

                    # cálculo
                    pa_res = analisar_estatistica(Pa)
                    pq_res = analisar_estatistica(Pq)

                    # tabela
                    tabela = pd.DataFrame({
                        "Parâmetro": [
                            "Média",
                            "Desvio P",
                            "Erro Aleatório",
                            "Erro (Arred.)",
                            "Valor (Arred.)",
                            "Valor (Real)"
                        ],
                        "Pa": pa_res,
                        "Pq": pq_res
                    })

                    st.markdown("### 📊 Estatística (Pa / Pq)")
                    st.dataframe(tabela)

                    # métricas
                    col1, col2 = st.columns(2)
                    col1.metric("📏 Pa Final", pa_res[5])
                    col2.metric("📏 Pq Final", pq_res[5])

                    # salvar para PCA
                    resultados[f.name] = {
                        "Pa_media": pa_res[0],
                        "Pq_media": pq_res[0],
                        "Pa_std": pa_res[1],
                        "Pq_std": pq_res[1],
                    }

                except Exception as e:
                    st.error(f"Erro no arquivo {f.name}: {e}")

            # salvar no session_state
            if resultados:
                st.session_state["perfilometria_samples"] = resultados

                st.divider()
                st.markdown("### 📊 Resumo das amostras")

                df_final = pd.DataFrame(resultados).T
                st.dataframe(df_final)

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        st.subheader("🧠 Análise de Componentes Principais (PCA)")
        st.caption("Identificação de padrões entre amostras")

        if "perfilometria_samples" not in st.session_state:
            st.warning("Execute a análise na aba Estatística primeiro.")
        else:

            dados = st.session_state["perfilometria_samples"]

            if not dados:
                st.warning("Sem dados disponíveis.")
            else:

                df = pd.DataFrame(dados).T

                st.markdown("### 📊 Dados utilizados")
                st.dataframe(df)

                try:
                    df_pca, var_exp = executar_pca(df)

                    st.markdown("### 📉 Variância explicada")
                    st.write(f"PC1: {var_exp[0]*100:.2f}%")
                    st.write(f"PC2: {var_exp[1]*100:.2f}%")

                    fig = plot_pca(df_pca, df.index)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Erro no PCA: {e}")
