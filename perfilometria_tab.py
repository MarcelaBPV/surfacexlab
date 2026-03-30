# =========================================================
# PERFILOMETRIA TAB — SurfaceXLab (CORRIGIDO + PCA)
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

    dados = np.array(dados)

    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)
    erro = desvio / np.sqrt(len(dados))

    erro_arred = round(erro, 2)
    valor_arred = round(media, 2)

    valor_real = f"{valor_arred} ± {erro_arred}"

    return media, desvio, erro, erro_arred, valor_arred, valor_real


# =========================================================
# LEITURA ROBUSTA
# =========================================================
def ler_arquivo(file):

    try:
        if file.name.endswith((".xlsx", ".xls")):
            # tenta leitura padrão
            df = pd.read_excel(file)

            # se vier bugado, tenta outro header
            if "Unnamed" in str(df.columns):
                df = pd.read_excel(file, header=1)

        else:
            try:
                df = pd.read_csv(file)
            except:
                df = pd.read_csv(file, delimiter="\t")

        return df

    except Exception as e:
        st.error(f"Erro na leitura: {e}")
        return None


# =========================================================
# DETECÇÃO SEGURA DE COLUNAS
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
    ax.set_title("PCA — Perfilometria")

    ax.grid(True)

    return fig


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_perfilometria_tab():

    st.subheader("📏 Perfilometria de Superfície")
    st.caption("Análise estatística de rugosidade + PCA")

    st.divider()

    subtabs = st.tabs(["📊 Estatística", "🧠 PCA"])

    # =====================================================
    # SUBABA 1 — ESTATÍSTICA
    # =====================================================
    with subtabs[0]:

        files = st.file_uploader(
            "📂 Envie arquivos",
            type=["csv", "txt", "log", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if not files:
            st.info("Envie arquivos para iniciar.")
            return

        resultados = {}

        for f in files:

            st.markdown(f"## 📄 {f.name}")

            df = ler_arquivo(f)

            if df is None:
                continue

            st.write("Colunas detectadas:", df.columns)
            st.dataframe(df.head())

            # detectar colunas
            pa_col, pq_col = detectar_colunas(df)

            # fallback manual
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

            # dados numéricos
            Pa = pd.to_numeric(df[pa_col], errors="coerce").dropna()
            Pq = pd.to_numeric(df[pq_col], errors="coerce").dropna()

            # DEBUG (ESSENCIAL)
            st.write("🔍 DEBUG Pa:", Pa.head())
            st.write("🔍 DEBUG Pq:", Pq.head())

            if len(Pa) == 0 or len(Pq) == 0:
                st.error("Dados inválidos — verifique colunas.")
                continue

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

            col1, col2 = st.columns(2)
            col1.metric("📏 Pa Final", pa_res[5])
            col2.metric("📏 Pq Final", pq_res[5])

            # salvar corretamente
            resultados[f.name] = {
                "Pa_media": pa_res[0],
                "Pq_media": pq_res[0],
                "Pa_std": pa_res[1],
                "Pq_std": pq_res[1],
            }

        # salvar session
        if resultados:
            st.session_state["perfilometria_samples"] = resultados

            st.divider()
            st.markdown("### 📊 Resumo")

            df_final = pd.DataFrame(resultados).T
            st.dataframe(df_final)

    # =====================================================
    # SUBABA 2 — PCA
    # =====================================================
    with subtabs[1]:

        st.subheader("🧠 PCA — Perfilometria")

        if "perfilometria_samples" not in st.session_state:
            st.warning("Execute a análise primeiro.")
            return

        df = pd.DataFrame(st.session_state["perfilometria_samples"]).T

        st.write("Dados usados no PCA:")
        st.dataframe(df)

        try:
            df_pca, var_exp = executar_pca(df)

            st.write(f"PC1: {var_exp[0]*100:.2f}%")
            st.write(f"PC2: {var_exp[1]*100:.2f}%")

            st.pyplot(plot_pca(df_pca, df.index))

        except Exception as e:
            st.error(f"Erro no PCA: {e}")
