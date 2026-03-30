# -- coding: utf-8 --

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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
# IDENTIFICAÇÃO AUTOMÁTICA
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
# EXTRAÇÃO INTELIGENTE (OWRK)
# =========================================================
def extract_tensiometry_angles(df):

    cols = [c.lower() for c in df.columns]

    water = diiodo = formamide = None

    for i, col in enumerate(cols):

        if "water" in col or "agua" in col:
            water = df.iloc[0, i]

        elif "diiodo" in col:
            diiodo = df.iloc[0, i]

        elif "formamide" in col:
            formamide = df.iloc[0, i]

    if water is None or diiodo is None or formamide is None:
        row = df.iloc[0]
        water, diiodo, formamide = row.iloc[0:3]

    return float(water), float(diiodo), float(formamide)


# =========================================================
# PROCESSAMENTO
# =========================================================
def process_iv(df):

    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    V = df.iloc[:, 0].values
    I = df.iloc[:, 1].values

    slope = np.polyfit(V, I, 1)[0]

    return {"Resistividade": 1 / slope if slope != 0 else np.nan}


def process_profilometry(df):

    df = df.apply(pd.to_numeric, errors="coerce")

    z = df.values.flatten()
    z = z[~np.isnan(z)]

    return {"Rugosidade": float(np.std(z))}


# =========================================================
# SPLIT POR TÉCNICA
# =========================================================
def split_by_technique():

    res, tens, perf = [], [], []

    for sample, content in st.session_state.samples_unified.items():

        if "resistividade" in content:
            row = {"Amostra": sample}
            row.update(content["resistividade"])
            res.append(row)

        if "perfilometria" in content:
            row = {"Amostra": sample}
            row.update(content["perfilometria"])
            perf.append(row)

        if "tensiometria" in content:

            df_rep = pd.DataFrame(content["tensiometria"])
            mean_vals = df_rep.mean(numeric_only=True)

            row = {"Amostra": sample}
            row.update(mean_vals.to_dict())

            tens.append(row)

    return pd.DataFrame(res), pd.DataFrame(tens), pd.DataFrame(perf)


# =========================================================
# PCA
# =========================================================
def run_pca(df, title):

    if df is None or len(df) < 2:
        st.info("Dados insuficientes")
        return

    labels = df["Amostra"]
    groups = labels.str.extract(r'([AB])')

    X = df.drop(columns=["Amostra"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7,7), dpi=300)

    for g in groups[0].unique():

        mask = groups[0] == g

        ax.scatter(scores[mask,0], scores[mask,1], s=100, label=f"Grupo {g}")

    for i, label in enumerate(labels):
        ax.text(scores[i,0], scores[i,1], label, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================================================
# PCA GLOBAL
# =========================================================
def run_pca_global():

    rows = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        if "resistividade" in content:
            row.update(content["resistividade"])

        if "perfilometria" in content:
            row.update(content["perfilometria"])

        if "tensiometria" in content:
            df_rep = pd.DataFrame(content["tensiometria"])
            row.update(df_rep.mean(numeric_only=True).to_dict())

        rows.append(row)

    df = pd.DataFrame(rows)

    st.subheader("Dataset Global")
    st.dataframe(df)

    run_pca(df, "PCA GLOBAL — Multitécnica")


# =========================================================
# OWRK PLOT
# =========================================================
def plot_owrk(df):

    if "Componente polar (mJ/m²)" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6,5), dpi=300)

    ax.scatter(
        df["Componente dispersiva (mJ/m²)"],
        df["Componente polar (mJ/m²)"],
        s=120,
        edgecolor="black"
    )

    for i, label in enumerate(df["Amostra"]):
        ax.text(
            df.iloc[i]["Componente dispersiva (mJ/m²)"],
            df.iloc[i]["Componente polar (mJ/m²)"],
            label
        )

    ax.set_xlabel("γᵈ")
    ax.set_ylabel("γᵖ")
    ax.set_title("OWRK — Energia de Superfície")

    st.pyplot(fig)


# =========================================================
# MAIN
# =========================================================
def render_analise_completa_amostras_tab(supabase=None):

    st.header("🧠 Análise Completa de Amostras")

    if "samples_unified" not in st.session_state:
        st.session_state.samples_unified = {}

    tabs = st.tabs(["📥 Upload", "📊 PCA", "🌐 PCA Global"])

# =========================================================
# UPLOAD
# =========================================================
    with tabs[0]:

        files = st.file_uploader(
            "Upload de todas as amostras",
            accept_multiple_files=True,
            type=["csv","xlsx","xls","txt","log"]
        )

        if files:

            for file in files:

                sample, tech = detect_sample_and_type(file.name)

                st.write(f"{file.name} → {sample} ({tech})")

                if sample not in st.session_state.samples_unified:
                    st.session_state.samples_unified[sample] = {}

                df = read_any_file(file)

                if df is None:
                    continue

                try:

                    if tech == "resistividade":
                        st.session_state.samples_unified[sample]["resistividade"] = process_iv(df)

                    elif tech == "perfilometria":
                        st.session_state.samples_unified[sample]["perfilometria"] = process_profilometry(df)

                    elif tech == "tensiometria":

                        if file.name.endswith((".log",".txt")):

                            result = process_tensiometry(
                                file,
                                {"water":70,"diiodomethane":50,"formamide":60},
                                0.5,
                                0.3
                            )

                            summary = result["summary"]

                        else:

                            df = df.apply(pd.to_numeric, errors="coerce")

                            w, d, f = extract_tensiometry_angles(df)

                            summary = {
                                **owkr_surface_energy({
                                    "water": w,
                                    "diiodomethane": d,
                                    "formamide": f
                                })
                            }

                        if "tensiometria" not in st.session_state.samples_unified[sample]:
                            st.session_state.samples_unified[sample]["tensiometria"] = []

                        st.session_state.samples_unified[sample]["tensiometria"].append(summary)

                except Exception as e:
                    st.error(f"Erro em {file.name}")
                    st.exception(e)

        st.write(st.session_state.samples_unified)


# =========================================================
# PCA
# =========================================================
    with tabs[1]:

        df_res, df_tens, df_perf = split_by_technique()

        st.subheader("⚡ Resistividade")
        st.dataframe(df_res)
        run_pca(df_res, "PCA — Resistividade")

        st.subheader("💧 Tensiometria")
        st.dataframe(df_tens)
        plot_owrk(df_tens)
        run_pca(df_tens, "PCA — Tensiometria")

        st.subheader("📏 Perfilometria")
        st.dataframe(df_perf)
        run_pca(df_perf, "PCA — Perfilometria")


# =========================================================
# PCA GLOBAL
# =========================================================
    with tabs[2]:

        run_pca_global()
