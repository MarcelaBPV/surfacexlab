# =========================================================
# Raman Tab — SurfaceXLab
# Interface Principal Raman
# =========================================================

import streamlit as st
import pandas as pd


# =========================================================
# IMPORTS
# =========================================================
try:

    from raman_processing import (
        process_raman_spectrum_with_groups,
        run_raman_pca
    )

    RAMAN_OK = True

except Exception:

    RAMAN_OK = False


from raman_mapping import (
    render_mapeamento_molecular_tab
)


# =========================================================
# LEITOR UNIVERSAL
# =========================================================
def read_any_file(file):

    name = file.name.lower()

    if name.endswith(".xlsx"):

        df = pd.read_excel(file)

    elif name.endswith(".txt") or name.endswith(".log"):

        df = pd.read_csv(
            file,
            sep=None,
            engine="python"
        )

    elif name.endswith(".csv"):

        df = pd.read_csv(file)

    else:

        raise ValueError(
            "Formato não suportado"
        )

    return df


# =========================================================
# TAB PRINCIPAL
# =========================================================
def render_raman_tab(samples):

    st.header("🧬 Análises Moleculares")

    if not RAMAN_OK:

        st.error(
            "⚠️ raman_processing não encontrado"
        )

        return


    # =====================================================
    # SELEÇÃO DE AMOSTRAS
    # =====================================================
    sample_ids = list(samples.keys())

    if not sample_ids:

        st.warning(
            "Nenhuma amostra cadastrada."
        )

        return


    selected_sample = st.selectbox(
        "🧪 Selecionar amostra",
        sample_ids
    )


    # =====================================================
    # METADADOS
    # =====================================================
    metadata = samples[selected_sample]["metadata"]

    st.info(
        f"""
        Material: {metadata.get('material', '-')}

        Tratamento: {metadata.get('treatment', '-')}

        ID: {metadata.get('sample_id', '-')}
        """
    )


    # =====================================================
    # SUBABAS
    # =====================================================
    subtabs = st.tabs([

        "📐 Processamento Raman",

        "🗺️ Mapping Molecular",

        "📊 PCA Multimodal"

    ])


    # =====================================================
    # SUBABA 1 — PROCESSAMENTO
    # =====================================================
    with subtabs[0]:

        st.subheader(
            "📐 Processamento Espectral Raman"
        )

        st.caption(
            """
            Pipeline automatizado contendo:
            suavização Savitzky–Golay,
            correção ASLS,
            fitting Lorentziano,
            extração de FWHM
            e classificação molecular.
            """
        )

        files = st.file_uploader(

            "Upload espectros Raman",

            accept_multiple_files=True,

            type=[
                "xlsx",
                "csv",
                "txt",
                "log"
            ]
        )

        if files:

            for f in files:

                st.markdown(
                    f"### 📄 {f.name}"
                )

                # =========================================
                # LEITURA
                # =========================================
                try:

                    df = read_any_file(f)

                except Exception as e:

                    st.error(
                        "Erro na leitura"
                    )

                    st.exception(e)

                    continue


                # =========================================
                # PROCESSAMENTO
                # =========================================
                try:

                    result = process_raman_spectrum_with_groups(f)

                except Exception as e:

                    st.error(
                        "Erro no processamento"
                    )

                    st.exception(e)

                    continue


                # =========================================
                # FIGURAS
                # =========================================
                st.pyplot(
                    result["figures"]["raw"]
                )

                st.pyplot(
                    result["figures"]["baseline"]
                )

                st.pyplot(
                    result["figures"]["fit"]
                )


                # =========================================
                # MÉTRICAS
                # =========================================
                col1, col2 = st.columns(2)

                col1.metric(
                    "R²",
                    f"{result['r2']:.4f}"
                )

                col2.metric(
                    "Quality",
                    result["quality_flag"]
                )


                # =========================================
                # PEAKS
                # =========================================
                st.markdown(
                    "### 🔬 Picos Identificados"
                )

                st.dataframe(
                    result["peaks_df"],
                    use_container_width=True
                )


                # =========================================
                # SAVE SAMPLE-CENTRIC
                # =========================================
                samples[selected_sample]["raman"] = {

                    "filename": f.name,

                    "peaks_df": result["peaks_df"],

                    "fingerprint": result["fingerprint"],

                    "r2": result["r2"],

                    "quality_flag": result["quality_flag"]
                }


                st.success(
                    "Espectro processado e salvo."
                )


    # =====================================================
    # SUBABA 2 — MAPPING
    # =====================================================
    with subtabs[1]:

        st.subheader(
            "🗺️ Mapping Molecular Raman"
        )

        st.caption(
            """
            Reconstrução espacial
            da distribuição molecular
            obtida por espectroscopia Raman.
            """
        )

        render_mapeamento_molecular_tab()


    # =====================================================
    # SUBABA 3 — PCA
    # =====================================================
    with subtabs[2]:

        st.subheader(
            "📊 PCA Raman Multimodal"
        )

        st.caption(
            """
            PCA baseado em fingerprints
            espectrais padronizados.
            """
        )

        try:

            pca_results = run_raman_pca(samples)

        except Exception as e:

            st.warning(str(e))

            return


        scores = pca_results["scores"]

        labels = pca_results["labels"]

        explained = pca_results[
            "explained_variance"
        ]


        # ================================================
        # FIGURA PCA
        # ================================================
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=80
        )

        for i, label in enumerate(labels):

            ax.text(
                scores[i, 0],
                scores[i, 1],
                label
            )

        ax.set_xlabel(
            f"PC1 ({explained[0]*100:.2f}%)"
        )

        ax.set_ylabel(
            f"PC2 ({explained[1]*100:.2f}%)"
        )

        ax.set_title(
            "PCA — Raman Fingerprints"
        )

        ax.grid(True)

        st.pyplot(fig)


        # ================================================
        # TABELA PCA
        # ================================================
        st.markdown(
            "### 📋 Fingerprints Raman"
        )

        st.dataframe(
            pca_results["dataframe"],
            use_container_width=True
        )


        # ================================================
        # INTERPRETAÇÃO
        # ================================================
        st.info(
            """
            A análise PCA foi realizada
            utilizando fingerprints espectrais
            extraídos automaticamente
            a partir dos parâmetros Raman
            identificados no fitting Lorentziano.
            """
        )
