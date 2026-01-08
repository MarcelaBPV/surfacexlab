# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd

from raman_processing import process_raman_spectrum_with_groups


# =========================================================
# UI — ABA RAMAN
# =========================================================
def render_raman_tab(supabase):

    st.header("Análises Moleculares — Espectroscopia Raman")

    st.markdown(
        """
        Este módulo realiza o **processamento completo de espectros Raman brutos**,
        incluindo correção de baseline, suavização, normalização,
        detecção automática de picos e geração de gráfico científico.

        ⚠️ O arquivo deve conter **duas colunas numéricas**:
        - Deslocamento Raman (cm⁻¹)
        - Intensidade
        """
    )

    uploaded_file = st.file_uploader(
        "Upload do espectro Raman (.csv, .txt, .xls, .xlsx)",
        type=["csv", "txt", "xls", "xlsx"],
        key="raman_upload"
    )

    if not uploaded_file:
        st.info("Aguardando upload do espectro Raman.")
        return

    st.success(f"Arquivo carregado: {uploaded_file.name}")

    # -----------------------------------------------------
    # Processamento
    # -----------------------------------------------------
    try:
        result = process_raman_spectrum_with_groups(
            file_like=uploaded_file,
            peak_prominence=0.02
        )
    except Exception as e:
        st.error("❌ Erro ao processar o espectro Raman.")
        st.exception(e)
        return

    if not isinstance(result, dict):
        st.error("Retorno inválido do processamento Raman.")
        st.write(result)
        return

    st.caption(f"Chaves retornadas: {list(result.keys())}")

    # -----------------------------------------------------
    # Plot Raman — CORRIGIDO
    # -----------------------------------------------------
    figures = result.get("figures")

    if figures is not None:
        st.subheader("Espectro Raman Processado")

        # Caso 1: figura única
        if hasattr(figures, "savefig"):
            st.pyplot(figures)

        # Caso 2: lista de figuras
        elif isinstance(figures, (list, tuple)):
            for i, fig in enumerate(figures, start=1):
                st.markdown(f"**Figura {i}**")
                st.pyplot(fig)

        # Caso 3: dicionário de figuras
        elif isinstance(figures, dict):
            for name, fig in figures.items():
                st.markdown(f"**{name}**")
                st.pyplot(fig)

        else:
            st.warning("Formato de figura não reconhecido.")

    else:
        st.warning(
            "Nenhuma figura Raman foi gerada. "
            "O arquivo pode não representar um espectro Raman bruto."
        )

    # -----------------------------------------------------
    # Tabela de picos
    # -----------------------------------------------------
    peaks_df = result.get("peaks_df")

    st.subheader("Picos Identificados")

    if isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:
        st.dataframe(peaks_df, use_container_width=True)
    else:
        st.info("Nenhum pico Raman identificado.")
