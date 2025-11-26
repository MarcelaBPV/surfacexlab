# raman_tab.py
# -*- coding: utf-8 -*-
"""
Aba 1 — Raman (Pacientes + Import Forms + Processamento Raman)

Aqui você vai basicamente pegar o bloco que já funciona no seu app antigo
(desde o "Pacientes — Cadastro e Importação do Google Forms" até o fim da
aba Raman) e colocar dentro da função render_raman_tab.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict
from datetime import datetime
import matplotlib.pyplot as plt

# Importa o pipeline que você mostrou no exemplo
from raman_processing import process_raman_pipeline

# (Se você usava annotate_molecular_groups e plot_main_and_residual,
# pode colocá-las aqui também, iguais às do código antigo.)

# --------------------------------------------------------------------
# Helpers Supabase (copie daqui ou adapte a partir do seu código antigo)
# --------------------------------------------------------------------
def safe_insert(supabase, table: str, records: List[Dict]):
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    if not records:
        return []
    batch = 800
    out: List[Dict] = []
    for i in range(0, len(records), batch):
        chunk = records[i : i + batch]
        res = supabase.table(table).insert(chunk).execute()
        data = getattr(res, "data", None)
        if not data:
            raise RuntimeError(f"Erro ao inserir lote em {table}.")
        out.extend(data)
    return out

# ... aqui você pode copiar create_patient_record, find_patient_by_email_or_cpf,
# create_sample_record, create_measurement_record, insert_raman_spectrum_df,
# insert_peaks_df, get_patients_list, get_samples_for_patient, json_safe etc.
# exatamente como no seu app original.


# --------------------------------------------------------------------
# Função principal da aba
# --------------------------------------------------------------------
def render_raman_tab(supabase):
    """
    Renderiza toda a lógica da antiga aba:
    - Cadastro manual de pacientes
    - Import XLSX/CSV do Google Forms
    - Upload de espectros (single + batch)
    - Processamento Raman (usando process_raman_pipeline)
    - Download de espectro e picos
    - Salvamento no Supabase
    - Visualização de ensaios salvos
    """
    st.header("1️⃣ Raman — Pacientes, Amostras e Espectros")

    if supabase is None:
        st.warning("Conecte o Supabase em st.secrets para habilitar cadastro e salvamento.")

    # ----------------------------------------------------------------
    # AQUI: COLE o código da sua aba 1 e 2 do app antigo,
    # adaptando apenas para usar 'supabase' recebido como parâmetro
    # (ou seja, remova a parte onde ele era criado dentro do arquivo).
    #
    # Tudo que hoje está dentro de:
    #   with tab_pat: ...  (Pacientes & Import Forms)
    #   with tab_raman: ... (Espectrometria Raman)
    # você coloca aqui, um embaixo do outro.
    #
    # Não coloque st.set_page_config aqui (já está em app.py).
    # ----------------------------------------------------------------

    st.info("Cole aqui a lógica completa da sua aba Raman atual (Pacientes + Ensaios).")
