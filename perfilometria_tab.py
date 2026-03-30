# =========================================================
# PERFILOMETRIA TAB — SurfaceXLab (PA + PQ + ESTATÍSTICA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np


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
# LEITURA INTELIGENTE DE ARQUIVO
# =========================================================
def ler_arquivo(file):

    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, delimiter="\t")


# =========================================================
# DETECÇÃO AUTOMÁTICA DE COLUNAS
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
# TAB PRINCIPAL
# =========================================================
def render_perfilometria_tab():

    st.subheader("📏 Perfilometria de Superfície")
    st.caption("Análise estatística de rugosidade (Pa e Pq)")

    st.divider()

    # =====================================================
    # UPLOAD
    # =====================================================
    files = st.file_uploader(
        "📂 Envie arquivos (csv, txt, log, xls, xlsx)",
        type=["csv", "txt", "log", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if not files:
        st.info("Envie um ou mais arquivos para iniciar.")
        return

    resultados = []

    # =====================================================
    # LOOP DOS ARQUIVOS
    # =====================================================
    for f in files:

        st.markdown(f"### 📄 {f.name}")

        try:
            # ============================
            # LEITURA
            # ============================
            df = ler_arquivo(f)

            st.dataframe(df.head())

            # ============================
            # DETECÇÃO DE COLUNAS
            # ============================
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

            # ============================
            # EXTRAIR DADOS
            # ============================
            Pa = pd.to_numeric(df[pa_col], errors="coerce").dropna().values
            Pq = pd.to_numeric(df[pq_col], errors="coerce").dropna().values

            # ============================
            # CÁLCULO
            # ============================
            pa_res = analisar_estatistica(Pa)
            pq_res = analisar_estatistica(Pq)

            # ============================
            # TABELA RESULTADO
            # ============================
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

            # ============================
            # RESULTADO FINAL DESTACADO
            # ============================
            col1, col2 = st.columns(2)

            col1.metric("📏 Pa Final", pa_res[5])
            col2.metric("📏 Pq Final", pq_res[5])

            # ============================
            # SALVAR RESULTADO
            # ============================
            resultados.append({
                "arquivo": f.name,
                "Pa": pa_res[5],
                "Pq": pq_res[5]
            })

        except Exception as e:
            st.error(f"Erro no arquivo {f.name}: {e}")

    # =====================================================
    # COMPARAÇÃO FINAL
    # =====================================================
    if resultados:

        st.divider()
        st.markdown("### 📊 Comparação entre amostras")

        df_final = pd.DataFrame(resultados)
        st.dataframe(df_final)

        if "perfilometria_samples" not in st.session_state:
            st.session_state["perfilometria_samples"] = {}

        for r in resultados:
            st.session_state["perfilometria_samples"][r["arquivo"]] = r
