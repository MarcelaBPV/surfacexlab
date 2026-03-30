# =========================================================
# ANÁLISE COMPLETA DE AMOSTRAS — VERSÃO FINAL ROBUSTA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import plotly.express as px

from tensiometria_processing import owkr_surface_energy

# =========================================================
# LEITURA UNIVERSAL ROBUSTA
# =========================================================

def read_any_file(file):

```
name = file.name.lower()

try:
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except:
            file.seek(0)
            df = pd.read_csv(file, sep=";")

    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file)

    elif name.endswith((".txt", ".log")):
        df = pd.read_csv(file, sep=r"\s+|,|;", engine="python")

    else:
        return None

    # 🔥 limpeza global
    df = df.applymap(lambda x: str(x).replace(" ", "") if isinstance(x, str) else x)
    df = df.replace(",", ".", regex=True)

    return df

except Exception as e:
    st.error(f"Erro ao ler {file.name}")
    st.exception(e)
    return None
```

# =========================================================
# IDENTIFICAÇÃO
# =========================================================

def detect_sample_and_type(filename):

```
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
```

# =========================================================
# RESISTIVIDADE
# =========================================================

def process_iv(df):

```
df.columns = [str(c).lower() for c in df.columns]

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(how="all")

v_col, i_col = None, None

for col in df.columns:
    if "v" in col:
        v_col = col
    if "i" in col:
        i_col = col

if v_col and i_col:
    V = df[v_col]
    I = df[i_col]
else:
    df_num = df.select_dtypes(include=np.number)

    if df_num.shape[1] < 2:
        raise ValueError("Dados insuficientes para I-V")

    V = df_num.iloc[:, 0]
    I = df_num.iloc[:, 1]

mask = (~V.isna()) & (~I.isna())
V = V[mask].values
I = I[mask].values

if len(V) < 2:
    raise ValueError("Poucos dados após limpeza")

slope = np.polyfit(V, I, 1)[0]

return {
    "Resistividade": float(1/slope) if slope != 0 else np.nan,
    "Slope": float(slope),
    "V": V,
    "I": I
}
```

# =========================================================
# PERFILOMETRIA
# =========================================================

def process_profilometry(df):

```
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(how="all")

if df.empty:
    raise ValueError("Sem dados válidos")

z = df.values.flatten()
z = z[~np.isnan(z)]

if len(z) < 5:
    raise ValueError("Poucos pontos")

return {
    "Rugosidade (Rq)": float(np.std(z)),
    "Rugosidade (Ra)": float(np.mean(np.abs(z - np.mean(z))))
}
```

# =========================================================
# TENSIOMETRIA
# =========================================================

def process_tensiometry_excel(df):

```
df.columns = [str(c).lower() for c in df.columns]
df = df.apply(pd.to_numeric, errors="coerce")

water = diiodo = formamide = None

for col in df.columns:
    if "water" in col or "agua" in col:
        water = df[col].mean()
    elif "diiodo" in col:
        diiodo = df[col].mean()
    elif "formamide" in col:
        formamide = df[col].mean()

if water is None:
    nums = df.select_dtypes(include=np.number).dropna()

    if nums.shape[1] >= 3:
        water, diiodo, formamide = nums.iloc[0, :3]
    else:
        raise ValueError("Não identificou líquidos")

return owkr_surface_energy({
    "water": float(water),
    "diiodomethane": float(diiodo),
    "formamide": float(formamide)
})
```

# =========================================================
# PCA INTERATIVO
# =========================================================

def run_pca_plotly(df, title):

```
if df.empty or len(df) < 2:
    st.warning("Dados insuficientes")
    return

labels = df["Amostra"]

X = df.drop(columns=["Amostra"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all").dropna()

if X.shape[0] < 2 or X.shape[1] < 2:
    st.warning("PCA inválido")
    return

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)

df_plot = pd.DataFrame({
    "PC1": scores[:,0],
    "PC2": scores[:,1],
    "Amostra": labels.iloc[:len(scores)]
})

fig = px.scatter(df_plot, x="PC1", y="PC2", text="Amostra", title=title)
fig.update_traces(textposition='top center')

st.plotly_chart(fig, use_container_width=True)
```

# =========================================================
# CLUSTERING
# =========================================================

def run_clustering(df):

```
if df.empty or len(df) < 3:
    return df

X = df.drop(columns=["Amostra"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all").dropna()

if X.shape[0] < 3 or X.shape[1] < 2:
    return df

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df = df.iloc[:len(clusters)].copy()
df["Cluster"] = clusters

fig = px.scatter(df, x=df.columns[1], y=df.columns[2],
                 color="Cluster", text="Amostra",
                 title="Clustering")

st.plotly_chart(fig, use_container_width=True)

return df
```

# =========================================================
# CORRELAÇÃO
# =========================================================

def run_correlation(df):

```
if "Resistividade" not in df.columns:
    return

if "Energia Superficial Total (mJ/m²)" not in df.columns:
    return

x = df["Resistividade"]
y = df["Energia Superficial Total (mJ/m²)"]

mask = (~x.isna()) & (~y.isna())
x = x[mask]
y = y[mask]

if len(x) < 2:
    return

coef = np.polyfit(x, y, 1)
y_pred = np.polyval(coef, x)

ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)

fig = px.scatter(x=x, y=y, trendline="ols",
                 title=f"Correlação Resistividade vs Energia (R²={r2:.3f})")

st.plotly_chart(fig, use_container_width=True)
```

# =========================================================
# SUPABASE
# =========================================================

def save_to_supabase(df, supabase):

```
if supabase is None:
    st.warning("Supabase não conectado")
    return

data = []

for _, row in df.iterrows():

    data.append({
        "sample": row.get("Amostra"),
        "resistividade": row.get("Resistividade"),
        "rugosidade": row.get("Rugosidade (Rq)"),
        "energia_total": row.get("Energia Superficial Total (mJ/m²)"),
        "energia_polar": row.get("Componente polar (mJ/m²)"),
        "energia_dispersiva": row.get("Componente dispersiva (mJ/m²)")
    })

try:
    supabase.table("samples_data").insert(data).execute()
    st.success("Dados salvos no Supabase 🚀")
except Exception as e:
    st.error("Erro ao salvar no Supabase")
    st.exception(e)
```

# =========================================================
# MAIN
# =========================================================

def render_analise_completa_amostras_tab(supabase=None):

```
st.header("🧠 Análise Completa de Amostras")

if "samples" not in st.session_state:
    st.session_state.samples = {}

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

# =====================================================
# DATAFRAME FINAL
# =====================================================
rows = []

for sample, data in st.session_state.samples.items():

    row = {"Amostra": sample}

    for tech in data:
        for k, v in data[tech].items():
            if not isinstance(v, np.ndarray):
                row[k] = v

    rows.append(row)

df = pd.DataFrame(rows)

st.dataframe(df)

# =====================================================
# INTELIGÊNCIA
# =====================================================
st.subheader("📊 PCA Interativo")
run_pca_plotly(df, "PCA")

st.subheader("🧠 Clustering")
df = run_clustering(df)

st.subheader("📈 Correlação entre técnicas")
run_correlation(df)

if st.button("💾 Salvar no banco"):
    save_to_supabase(df, supabase)
```
