# =========================================================
# raman_mapping.py
# SurfaceXLab — Raman Molecular Mapping
# FINAL REAL MAPPING VERSION
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.signal import savgol_filter
from scipy.signal import find_peaks


# =========================================================
# LEITOR UNIVERSAL
# =========================================================
def read_mapping_file(file_like):

    name = file_like.name.lower()

    # =====================================================
    # EXCEL
    # =====================================================
    if name.endswith(

        (".xlsx", ".xls")
    ):

        df = pd.read_excel(file_like)

    # =====================================================
    # CSV / TXT
    # =====================================================
    else:

        try:

            df = pd.read_csv(

                file_like,

                sep=None,

                engine="python",

                encoding="utf-8"
            )

        except:

            file_like.seek(0)

            df = pd.read_csv(

                file_like,

                delim_whitespace=True,

                encoding="latin1"
            )

    # =====================================================
    # LIMPEZA
    # =====================================================
    df.columns = [

        str(c).strip().lower()

        for c in df.columns
    ]

    return df


# =========================================================
# TAB MAPEAMENTO
# =========================================================
def render_mapeamento_molecular_tab():

    st.subheader(
        "🗺️ Raman Molecular Mapping"
    )

    st.markdown("""
    Processamento espacial de espectros Raman
    adquiridos em múltiplas posições da amostra.
    """)

    uploaded_file = st.file_uploader(

        "Upload arquivo Raman Mapping",

        type=[

            "csv",
            "txt",
            "xlsx",
            "xls"
        ]
    )

    # =====================================================
    # SEM ARQUIVO
    # =====================================================
    if uploaded_file is None:

        st.info("""
        Faça upload do arquivo Raman Mapping.
        """)

        return

    # =====================================================
    # LEITURA
    # =====================================================
    try:

        df = read_mapping_file(
            uploaded_file
        )

    except Exception as e:

        st.error(
            "Erro ao ler arquivo."
        )

        st.exception(e)

        return

    # =====================================================
    # DATAFRAME
    # =====================================================
    st.subheader(
        "📄 Dataset Importado"
    )

    st.dataframe(

        df.head(),

        width="stretch"
    )

    # =====================================================
    # IDENTIFICA COLUNAS
    # =====================================================
    x_col = None
    y_col = None
    wave_col = None
    intensity_col = None

    for c in df.columns:

        if c == "x":

            x_col = c

        elif c == "y":

            y_col = c

        elif "wave" in c:

            wave_col = c

        elif "intensity" in c:

            intensity_col = c

    # =====================================================
    # VALIDAÇÃO
    # =====================================================
    if any(

        v is None

        for v in [

            x_col,
            y_col,
            wave_col,
            intensity_col
        ]
    ):

        st.error("""
        O arquivo precisa conter:

        x | y | wave | intensity
        """)

        return

    # =====================================================
    # CONVERSÃO NUMÉRICA
    # =====================================================
    for c in [

        x_col,
        y_col,
        wave_col,
        intensity_col
    ]:

        df[c] = pd.to_numeric(

            df[c],

            errors="coerce"
        )

    df = df.dropna()

    # =====================================================
    # GRUPOS ESPECTRAIS
    # =====================================================
    grouped = df.groupby(

        [x_col, y_col]
    )

    total_spectra = len(grouped)

    st.success(
        f"{total_spectra} espectros detectados."
    )

    # =====================================================
    # FIGURA ESPECTROS
    # =====================================================
    st.subheader(
        "📈 Raman Spectra"
    )

    fig_spec, ax_spec = plt.subplots(

        figsize=(10,6),

        dpi=300
    )

    features = []

    heatmap_data = []

    # =====================================================
    # LOOP DOS ESPECTROS
    # =====================================================
    for idx, (

        (x_pos, y_pos),

        group

    ) in enumerate(grouped):

        group = group.sort_values(
            wave_col
        )

        wave = group[
            wave_col
        ].values

        intensity = group[
            intensity_col
        ].values

        # =================================================
        # SUAVIZAÇÃO
        # =================================================
        if len(intensity) > 21:

            intensity = savgol_filter(

                intensity,

                21,

                3
            )

        # =================================================
        # NORMALIZA
        # =================================================
        intensity_norm = (

            intensity -
            np.min(intensity)

        ) / (

            np.max(intensity) -
            np.min(intensity) + 1e-9
        )

        # =================================================
        # PLOT
        # =================================================
        ax_spec.plot(

            wave,

            intensity_norm,

            linewidth=1,

            alpha=0.8
        )

        # =================================================
        # PICOS
        # =================================================
        peaks, props = find_peaks(

            intensity_norm,

            prominence=0.05
        )

        # =================================================
        # FEATURE EXTRACTION
        # =================================================
        if len(peaks) > 0:

            peak_wave = wave[
                peaks[np.argmax(
                    intensity_norm[peaks]
                )]
            ]

            peak_intensity = np.max(
                intensity_norm
            )

        else:

            peak_wave = 0
            peak_intensity = 0

        features.append({

            "Spectrum":

                idx + 1,

            "X":

                x_pos,

            "Y":

                y_pos,

            "Main Peak":

                peak_wave,

            "Max Intensity":

                peak_intensity
        })

        heatmap_data.append([

            x_pos,

            y_pos,

            peak_intensity
        ])

    # =====================================================
    # FIGURA FINAL
    # =====================================================
    ax_spec.set_title(
        "Raman Spectral Mapping"
    )

    ax_spec.set_xlabel(
        "Raman Shift (cm⁻¹)"
    )

    ax_spec.set_ylabel(
        "Normalized Intensity"
    )

    ax_spec.grid(alpha=0.2)

    st.pyplot(fig_spec)

    # =====================================================
    # FEATURES DF
    # =====================================================
    features_df = pd.DataFrame(
        features
    )

    # =====================================================
    # TABELA
    # =====================================================
    st.subheader(
        "📊 Spectral Features"
    )

    st.dataframe(

        features_df,

        width="stretch"
    )

    # =====================================================
    # HEATMAP
    # =====================================================
    st.subheader(
        "🔥 Molecular Distribution"
    )

    heat_df = pd.DataFrame(

        heatmap_data,

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

    fig_heat, ax_heat = plt.subplots(

        figsize=(6,5),

        dpi=300
    )

    im = ax_heat.imshow(

        pivot,

        cmap="inferno",

        origin="lower",

        aspect="auto"
    )

    cbar = plt.colorbar(

        im,

        ax=ax_heat
    )

    cbar.set_label(
        "Intensity"
    )

    ax_heat.set_title(
        "Molecular Heatmap"
    )

    st.pyplot(fig_heat)

    # =====================================================
    # PCA
    # =====================================================
    st.subheader(
        "📊 PCA Mapping"
    )

    try:

        X = features_df[[

            "Main Peak",

            "Max Intensity"
        ]]

        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(
            n_components=2
        )

        scores = pca.fit_transform(
            X_scaled
        )

        explained = (
            pca.explained_variance_ratio_
            * 100
        )

        fig_pca, ax_pca = plt.subplots(

            figsize=(6,5),

            dpi=300
        )

        ax_pca.scatter(

            scores[:,0],

            scores[:,1]
        )

        for i in range(len(scores)):

            ax_pca.text(

                scores[i,0],

                scores[i,1],

                f"S{i+1}",

                fontsize=7
            )

        ax_pca.set_xlabel(
            f"PC1 ({explained[0]:.1f}%)"
        )

        ax_pca.set_ylabel(
            f"PC2 ({explained[1]:.1f}%)"
        )

        ax_pca.set_title(
            "PCA Raman Mapping"
        )

        ax_pca.grid(alpha=0.2)

        st.pyplot(fig_pca)

    except Exception as e:

        st.warning(
            "PCA não pôde ser executado."
        )

        st.exception(e)

    # =====================================================
    # EXPORT
    # =====================================================
    st.subheader(
        "⬇ Export"
    )

    csv = features_df.to_csv(
        index=False
    )

    st.download_button(

        label="Download Mapping Features",

        data=csv,

        file_name="raman_mapping_features.csv",

        mime="text/csv"
    )
