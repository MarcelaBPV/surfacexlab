# -- coding: utf-8 --

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensiometria_processing import owkr_surface_energy


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_any_file(file):

    name = file.name.lower()

    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, sep=None, engine="python")

        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)

        elif name.endswith((".txt", ".log")):
            return pd.read_csv(file, sep=r"\s+|,|;", engine="python")

    except:
        return None


# =========================================================
# IDENTIFICAR AMOSTRA
# =========================================================
def detect_sample_and_type(filename):

    name = filename.lower()

    match = re.search(r'([ab]\d+\.?\d*)', name)
    sample = match.group(1).upper() if match else "UNKNOWN"

    if "resistividade" in name:
        return sample, "resistividade"
    elif "tensiometria" in name:
        return sample, "tensiometria"
    elif "perfilometria" in name:
        return sample, "perfilometria"

    return sample, "unknown"


# =========================================================
# RESISTIVIDADE ROBUSTA
# =========================================================
def process_iv(df):

    df.columns = [str(c).lower() for c in df.columns]

    v_col = None
    i_col = None

    for col in df.columns:
        if "v" in col:
            v_col = col
        if "i" in col:
            i_col = col

    if v_col is None or i_col is None:
        df = df.select_dtypes(include=np.number)

        if df.shape[1] < 2:
            raise ValueError("Arquivo inválido para I-V")

        V = df.iloc[:, 0]
        I = df.iloc[:, 1]
    else:
        V = df[v_col]
        I = df[i_col]

    V = pd.to_numeric(V, errors="coerce")
    I = pd.to_numeric(I, errors="coerce")

    mask = (~V.isna()) & (~I.isna())

    V = V[mask].values
    I = I[mask].values

    slope = np.polyfit(V, I, 1)[0]

    return {
        "Resistividade": float(1 / slope) if slope != 0 else np.nan,
        "V": V,
        "I": I
    }


# =========================================================
# PERFILOMETRIA
# =========================================================
def process_profilometry(df):

    df = df.select_dtypes(include=np.number)

    z = df.values.flatten()
    z = z[~np.isnan(z)]

    return {
        "Rugosidade": float(np.std(z))
    }


# =========================================================
# TENSIOMETRIA INTELIGENTE
# =========================================================
def process_tensiometry_excel(df):

    df.columns = [str(c).lower() for c in df.columns]

    water = diiodo = formamide = None

    for col in df.columns:

        if "water" in col or "agua" in col:
            water = df[col].mean()

        elif "diiodo" in col:
            diiodo = df[col].mean()

        elif "formamide" in col:
            formamide = df[col].mean()

    if water is None:
        nums = df.select_dtypes(include=np.number).iloc[0]
        water, diiodo, formamide = nums.iloc[:3]

    return owkr_surface_energy({
        "water": float(water),
        "diiodomethane": float(diiodo),
        "formamide": float(formamide)
    })


# =========================================================
# PCA
# =========================================================
def run_pca(df, title):

    if len(df) < 2:
        st.info("Poucas amostras")
        return

    labels = df["Amostra"]

    X = df.drop(columns=["Amostra"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(scores[:,0], scores[:,1], s=80)

    for i, label in enumerate(labels):
        ax.text(scores[i,0], scores[i,1], label)

    ax.set_title(title)
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# MAIN
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    if "samples" not in st.session_state:
        st.session_state.samples = {}

    subtabs = st.tabs([
        "🔬 Análise por Técnica",
        "🌐 PCA Global"
    ])

# =========================================================
# SUBABA 1
# =========================================================
    with subtabs[0]:

        files = st.file_uploader(
            "Upload das amostras",
            accept_multiple_files=True
        )

        if files:

            for file in files:

                sample, tech = detect_sample_and_type(file.name)

                df = read_any_file(file)

                if df is None:
                    continue

                if sample not in st.session_state.samples:
                    st.session_state.samples[sample] = {}

                try:

                    if tech == "resistividade":
                        st.session_state.samples[sample]["resistividade"] = process_iv(df)

                    elif tech == "perfilometria":
                        st.session_state.samples[sample]["perfilometria"] = process_profilometry(df)

                    elif tech == "tensiometria":
                        st.session_state.samples[sample]["tensiometria"] = process_tensiometry_excel(df)

                except Exception as e:
                    st.error(f"Erro em {file.name}")
                    st.exception(e)

        tecnica = st.selectbox(
            "Selecionar técnica",
            ["resistividade", "tensiometria", "perfilometria"]
        )

        rows = []

        for sample, data in st.session_state.samples.items():

            if tecnica in data:

                row = {"Amostra": sample}
                row.update({k:v for k,v in data[tecnica].items() if not isinstance(v, np.ndarray)})
                rows.append(row)

        df = pd.DataFrame(rows)

        st.dataframe(df)

        # =====================================================
        # GRÁFICOS
        # =====================================================
        if tecnica == "resistividade":

            cols = st.columns(3)

            for i,(sample,data) in enumerate(st.session_state.samples.items()):

                if "resistividade" not in data:
                    continue

                V = data["resistividade"]["V"]
                I = data["resistividade"]["I"]

                with cols[i%3]:

                    fig,ax = plt.subplots(figsize=(3,2))
                    ax.plot(V,I)
                    ax.set_title(sample)

                    st.pyplot(fig)

                    if st.button(f"Expandir {sample}"):
                        fig2,ax2 = plt.subplots()
                        ax2.plot(V,I)
                        st.pyplot(fig2)

        if tecnica == "tensiometria":

            fig,ax = plt.subplots()

            ax.scatter(
                df["Componente dispersiva (mJ/m²)"],
                df["Componente polar (mJ/m²)"]
            )

            for i,row in df.iterrows():
                ax.text(row[1], row[2], row[0])

            ax.set_xlabel("γd")
            ax.set_ylabel("γp")

            st.pyplot(fig)

        # PCA
        run_pca(df, f"PCA — {tecnica}")

# =========================================================
# SUBABA 2
# =========================================================
    with subtabs[1]:

        rows = []

        for sample,data in st.session_state.samples.items():

            row = {"Amostra": sample}

            for tech in data:
                for k,v in data[tech].items():
                    if not isinstance(v, np.ndarray):
                        row[k] = v

            rows.append(row)

        df = pd.DataFrame(rows)

        st.dataframe(df)

        run_pca(df, "PCA GLOBAL")
