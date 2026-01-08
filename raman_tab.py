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

        ⚠️ **O arquivo deve conter exatamente duas colunas numéricas**:
        - Deslocamento Raman (cm⁻¹)
        - Intensidade
        """
    )

    # -----------------------------------------------------
    # Upload do arquivo
    # -----------------------------------------------------
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
    # Processamento com proteção
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

    # -----------------------------------------------------
    # Validação do retorno
    # -----------------------------------------------------
    if not isinstance(result, dict):
        st.error("Retorno inválido do processamento Raman.")
        st.write(result)
        return

    st.caption(f"Chaves retornadas: {list(result.keys())}")

    # -----------------------------------------------------
    # Gráfico Raman
    # -----------------------------------------------------
    if "figure" in result and result["figure"] is not None:
        st.subheader("📈 Espectro Raman Processado")
        st.pyplot(result["figure"])
    else:
        st.warning(
            "O arquivo enviado não parece ser um espectro Raman bruto válido "
            "(duas colunas numéricas: deslocamento Raman × intensidade)."
        )

    # -----------------------------------------------------
    # Tabela de picos
    # -----------------------------------------------------
    if "peaks_df" in result and isinstance(result["peaks_df"], pd.DataFrame):
        st.subheader("Picos Identificados")

        if not result["peaks_df"].empty:
            st.dataframe(result["peaks_df"], use_container_width=True)
        else:
            st.info("Nenhum pico Raman foi identificado neste espectro.")
    else:
        st.info("Tabela de picos não disponível para este arquivo.")
