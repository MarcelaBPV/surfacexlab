# =========================================================
# Raman Molecular Mapping — Independent Module
# SurfaceXLab
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Raman Molecular Mapping",
    layout="wide"
)

st.title("🧬 Raman Molecular Mapping")

st.markdown("""
Módulo independente para reconstrução espacial de mapas
moleculares Raman a partir de espectros adquiridos ao
longo da superfície da amostra.

Fluxo aplicado:
- Upload dos espectros;
- Correção de linha de base;
- Suavização espectral;
- Normalização;
- Identificação automática de picos;
- Reconstrução espacial do mapa molecular.
""")

# =========================================================
# BASELINE ALS
# =========================================================
def baseline_als(y, lam=1e5, p=0.01, niter=10):

    L = len(y)

    D = sparse.diags(
        [1, -2, 1],
        [0, -1, -2],
        shape=(L, L)
    )

    D = lam * D.dot(D.transpose())

    w = np.ones(L)

    for i in range(niter):

        W = sparse.spdiags(w, 0, L, L)

        Z = W + D

        z = spsolve(Z, w * y)

        w = p * (y > z) + (1 - p) * (y < z)

    return z

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "Upload arquivo Raman",
    type=["csv", "txt", "xlsx"]
)

if uploaded_file is None:

    st.info("Faça upload do arquivo Raman Mapping.")
    st.stop()

# =========================================================
# READ FILE
# =========================================================
try:

    if uploaded_file.name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(uploaded_file)

    else:

        try:

            df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

        except:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                delim_whitespace=True
            )

except Exception as e:

    st.error("Erro ao carregar arquivo.")
    st.exception(e)
    st.stop()

# =========================================================
# COLUMN CLEAN
# =========================================================
df.columns = [
    str(c).strip().lower()
    for c in df.columns
]

# =========================================================
# REQUIRED COLUMNS
# =========================================================
required = ["x", "y", "wave", "intensity"]

for c in required:

    if c not in df.columns:

        st.error(f"Coluna obrigatória ausente: {c}")
        st.stop()

# =========================================================
# NUMERIC CONVERSION
# =========================================================
for c in required:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

df = df.dropna()

# =========================================================
# GROUP SPECTRA
# =========================================================
grouped = df.groupby(["x", "y"])

st.success(
    f"{len(grouped)} espectros detectados."
)

# =========================================================
# FIGURE — ALL SPECTRA
# =========================================================
fig, ax = plt.subplots(
    figsize=(12,7),
    dpi=300
)

heatmap = []

features = []

# =========================================================
# LOOP
# =========================================================
for idx, ((x, y), group) in enumerate(grouped):

    group = group.sort_values("wave")

    wave = group["wave"].values
    intensity = group["intensity"].values

    # =====================================================
    # BASELINE
    # =====================================================
    baseline = baseline_als(intensity)

    corrected = intensity - baseline

    # =====================================================
    # SMOOTH
    # =====================================================
    if len(corrected) > 21:

        smooth = savgol_filter(
            corrected,
            21,
            3
        )

    else:

        smooth = corrected

    # =====================================================
    # NORMALIZATION
    # =====================================================
    norm = (
        smooth - np.min(smooth)
    ) / (
        np.max(smooth) - np.min(smooth) + 1e-9
    )

    # =====================================================
    # PEAKS
    # =====================================================
    peaks, props = find_peaks(
        norm,
        prominence=0.05
    )

    # =====================================================
    # MAIN PEAK
    # =====================================================
    if len(peaks) > 0:

        peak_idx = peaks[
            np.argmax(norm[peaks])
        ]

        peak_wave = wave[peak_idx]
        peak_intensity = norm[peak_idx]

    else:

        peak_wave = 0
        peak_intensity = 0

    # =====================================================
    # STORE FEATURES
    # =====================================================
    features.append({

        "Spectrum": idx + 1,
        "X": x,
        "Y": y,
        "Main Peak": peak_wave,
        "Max Intensity": peak_intensity
    })

    # =====================================================
    # HEATMAP
    # =====================================================
    heatmap.append([

        x,
        y,
        peak_intensity
    ])

    # =====================================================
    # PLOT
    # =====================================================
    ax.plot(
        wave,
        norm,
        linewidth=1
    )

    # =====================================================
    # PEAK MARKERS
    # =====================================================
    ax.scatter(
        wave[peaks],
        norm[peaks],
        s=10
    )

# =========================================================
# FINAL FIGURE
# =========================================================
ax.set_title(
    "Raman Spectra Mapping"
)

ax.set_xlabel(
    "Raman Shift (cm⁻¹)"
)

ax.set_ylabel(
    "Normalized Intensity"
)

ax.grid(alpha=0.2)

st.pyplot(fig)

# =========================================================
# FEATURES TABLE
# =========================================================
features_df = pd.DataFrame(features)

st.subheader("📊 Spectral Features")

st.dataframe(features_df)

# =========================================================
# HEATMAP
# =========================================================
st.subheader("🔥 Raman Molecular Map")

heat_df = pd.DataFrame(
    heatmap,
    columns=[
        "X",
        "Y",
        "Intensity"
    ]
)

pivot = heat_df.pivot(
    index="Y",
    columns="X",
    values="Intensity"
)

fig2, ax2 = plt.subplots(
    figsize=(7,6),
    dpi=300
)

im = ax2.imshow(
    pivot,
    cmap="inferno",
    origin="lower",
    aspect="auto"
)

cbar = plt.colorbar(
    im,
    ax=ax2
)

cbar.set_label(
    "Relative Raman Intensity"
)

ax2.set_title(
    "Spatial Molecular Distribution"
)

ax2.set_xlabel("X Position")

ax2.set_ylabel("Y Position")

st.pyplot(fig2)

# =========================================================
# DOWNLOAD
# =========================================================
csv = features_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Features",
    data=csv,
    file_name="raman_mapping_features.csv",
    mime="text/csv"
)
