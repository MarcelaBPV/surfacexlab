# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensiometria_processing import process_tensiometry, owkr_surface_energy


# =========================================================
# LEITURA UNIVERSAL
# =========================================================
def read_any_file(file):

    name = file.name.lower()

    try:
        if name.endswith(".csv"):
            try:
                return pd.read_csv(file)
            except:
                file.seek(0)
                return pd.read_csv(file, sep=";")

        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)

        elif name.endswith((".txt", ".log")):
            return pd.read_csv(file, sep=r"\s+|,|;", engine="python")

    except Exception as e:
        st.error(f"Erro ao ler {file.name}")
        st.exception(e)
        return None


# =========================================================
# EXTRAÇÃO INTELIGENTE (TENSIOMETRIA)
# =========================================================
def extract_tensiometry_angles(df):

    cols = [c.lower() for c in df.columns]

    water = None
    diiodo = None
    formamide = None

    for i, col in enumerate(cols):

        if "water" in col or "agua" in col:
            water = df.iloc[0, i]

        elif "diiodo" in col:
            diiodo = df.iloc[0, i]

        elif "formamide" in col or "formamida" in col:
            formamide = df.iloc[0, i]

    if water is None or diiodo is None or formamide is None:
        row = df.iloc[0]
        water = row.iloc[0]
        diiodo = row.iloc[1]
        formamide = row.iloc[2]

    return float(water), float(diiodo), float(formamide)


# =========================================================
# PROCESSAMENTO IV
# =========================================================
def process_iv(df):

    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    V = df.iloc[:, 0].values
    I = df.iloc[:, 1].values

    slope = np.polyfit(V, I, 1)[0]

    return {"Resistividade": 1 / slope if slope != 0 else np.nan}


# =========================================================
# PERFILOMETRIA
# =========================================================
def process_profilometry(df):

    df = df.apply(pd.to_numeric, errors="coerce")

    z = df.values.flatten()
    z = z[~np.isnan(z)]

    return {"Rugosidade (std)": float(np.std(z))}


# =========================================================
# BUILD DATASET
# =========================================================
def build_dataset():

    data = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        for key, val in content.items():

            if key == "tensiometria":

                df_rep = pd.DataFrame(val)

                mean_vals = df_rep.mean(numeric_only=True)
                std_vals = df_rep.std(numeric_only=True)

                for col in mean_vals.index:
                    row[f"{col} (mean)"] = mean_vals[col]
                    row[f"{col} (std)"] = std_vals[col]

            elif isinstance(val, dict):
                row.update(val)

        data.append(row)

    return pd.DataFrame(data)


# =========================================================
# OWRK PLOT
# =========================================================
def plot_owrk(df):

    if "Componente polar (mJ/m²) (mean)" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6,5), dpi=300)

    ax.scatter(
        df["Componente dispersiva (mJ/m²) (mean)"],
        df["Componente polar (mJ/m²) (mean)"],
        s=120,
        edgecolor="black"
    )

    for i, label in enumerate(df["Amostra"]):
        ax.text(
            df.iloc[i]["Componente dispersiva (mJ/m²) (mean)"],
            df.iloc[i]["Componente polar (mJ/m²) (mean)"],
            label,
            fontsize=9
        )

    ax.set_xlabel("γᵈ (mJ/m²)")
    ax.set_ylabel("γᵖ (mJ/m²)")
    ax.set_title("OWRK — Energia de Superfície")
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# PCA CLUSTER
# =========================================================
def run_pca_cluster(df):

    df["Grupo"] = df["Amostra"].str.extract(r"([AB])")

    features = df.drop(columns=["Amostra", "Grupo"], errors="ignore")
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    for g in df["Grupo"].unique():
        mask = df["Grupo"] == g
        ax.scatter(scores[mask,0], scores[mask,1], label=g, s=100)

    for i, label in enumerate(df["Amostra"]):
        ax.text(scores[i,0], scores[i,1], label)

    ax.legend()
    ax.set_title("PCA — Cluster por tratamento")
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# MAIN
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    if "samples_unified" not in st.session_state:
        st.session_state.samples_unified = {}

    tabs = st.tabs(["Upload", "PCA"])

# =========================================================
# UPLOAD
# =========================================================
    with tabs[0]:

        sample_id = st.text_input("Amostra (ex: A1.5)")

        technique = st.selectbox(
            "Técnica",
            ["Resistividade", "Tensiometria", "Perfilometria"]
        )

        files = st.file_uploader(
            "Upload",
            accept_multiple_files=True,
            type=["csv","xlsx","xls","txt","log"]
        )

        if files and sample_id:

            if sample_id not in st.session_state.samples_unified:
                st.session_state.samples_unified[sample_id] = {}

            for file in files:

                df = read_any_file(file)

                if df is None:
                    continue

                if technique == "Resistividade":
                    st.session_state.samples_unified[sample_id]["resistividade"] = process_iv(df)

                elif technique == "Perfilometria":
                    st.session_state.samples_unified[sample_id]["perfilometria"] = process_profilometry(df)

                elif technique == "Tensiometria":

                    name = file.name.lower()

                    if name.endswith((".log", ".txt")):

                        result = process_tensiometry(
                            file,
                            {"water":70,"diiodomethane":50,"formamide":60},
                            0.5,
                            0.3
                        )

                        summary = result["summary"]

                    else:

                        df = df.apply(pd.to_numeric, errors="coerce")
                        theta_water, theta_diiodo, theta_formamide = extract_tensiometry_angles(df)

                        owkr = owkr_surface_energy({
                            "water": theta_water,
                            "diiodomethane": theta_diiodo,
                            "formamide": theta_formamide
                        })

                        summary = {
                            "Ângulo água": theta_water,
                            "Ângulo diiodo": theta_diiodo,
                            "Ângulo formamida": theta_formamide,
                            **owkr
                        }

                    if "tensiometria" not in st.session_state.samples_unified[sample_id]:
                        st.session_state.samples_unified[sample_id]["tensiometria"] = []

                    st.session_state.samples_unified[sample_id]["tensiometria"].append(summary)

        st.write(st.session_state.samples_unified)

# =========================================================
# PCA
# =========================================================
    with tabs[1]:

        df = build_dataset()

        if df is None or len(df) < 2:
            st.info("Carregue ao menos duas amostras")
            return

        st.dataframe(df)

        st.subheader("OWRK")
        plot_owrk(df)

        st.subheader("PCA")
        run_pca_cluster(df)
