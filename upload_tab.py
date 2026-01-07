# upload_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Upload Centralizado de Experimentos

Tipos suportados:
1) Molecular (Raman — sangue)
2) Elétrica (Resistividade — motores)
3) Físico-mecânica (Tensiometria — nanotubos)

Upload obrigatório via arquivo.
"""

import streamlit as st
from datetime import date


# =========================================================
# HELPERS — SUPABASE
# =========================================================
def get_samples(supabase):
    res = (
        supabase
        .table("samples")
        .select("id, sample_code")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data if res.data else []


def create_experiment(
    supabase,
    sample_id: str,
    experiment_type: str,
    operator: str,
    equipment: str,
    notes: str,
):
    res = (
        supabase
        .table("experiments")
        .insert({
            "sample_id": sample_id,
            "experiment_type": experiment_type,
            "operator": operator,
            "equipment": equipment,
            "notes": notes,
            "experiment_date": str(date.today()),
        })
        .execute()
    )

    if not res.data:
        raise RuntimeError("Erro ao criar experimento.")

    return res.data[0]["id"]


# =========================================================
# UI — UPLOAD TAB
# =========================================================
def render_upload_tab(supabase):
    st.header("Upload de Experimentos")

    st.markdown(
        """
        Todos os experimentos devem ser inseridos **exclusivamente via upload de arquivos**.
        
        **Tipos suportados:**
        - 1 Molecular (Raman)
        - 2 Elétrica (Resistividade)
        - 3 Físico-mecânica (Tensiometria)
        """
    )

    # -----------------------------------------------------
    # Seleção da amostra
    # -----------------------------------------------------
    samples = get_samples(supabase)

    if not samples:
        st.warning("Nenhuma amostra cadastrada.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}

    sample_code = st.selectbox(
        "Amostra",
        list(sample_map.keys()),
    )
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # Tipo de experimento
    # -----------------------------------------------------
    st.subheader("Tipo de experimento")

    experiment_type = st.radio(
        "Selecione o tipo",
        [
            "Raman (Molecular — Sangue)",
            "Elétrica (Resistividade — Motores)",
            "Físico-mecânica (Tensiometria — Nanotubos)",
        ],
    )

    if experiment_type.startswith("Raman"):
        exp_code = "Raman"
        accepted_files = ["csv", "txt", "xlsx"]
        equipment_default = "Raman Spectrometer"

    elif experiment_type.startswith("Elétrica"):
        exp_code = "Electrical"
        accepted_files = ["csv", "xlsx"]
        equipment_default = "Source Measure Unit / Multimeter"

    else:
        exp_code = "Tensiometry"
        accepted_files = ["csv", "xlsx"]
        equipment_default = "Goniometer / Tensiometer"

    # -----------------------------------------------------
    # Metadados
    # -----------------------------------------------------
    st.subheader("Metadados do experimento")

    col1, col2 = st.columns(2)
    with col1:
        operator = st.text_input("Operador")
    with col2:
        equipment = st.text_input("Equipamento", equipment_default)

    notes = st.text_area("Observações")

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------
    st.subheader("Upload do arquivo experimental")

    uploaded_file = st.file_uploader(
        "Arquivo de dados brutos",
        type=accepted_files,
    )

    # -----------------------------------------------------
    # Salvar experimento
    # -----------------------------------------------------
    if uploaded_file and st.button("💾 Registrar experimento"):
        try:
            with st.spinner("Salvando experimento..."):

                experiment_id = create_experiment(
                    supabase=supabase,
                    sample_id=sample_id,
                    experiment_type=exp_code,
                    operator=operator,
                    equipment=equipment,
                    notes=notes,
                )

                # Arquivo bruto ainda NÃO é processado aqui
                # Ele será consumido no módulo específico (Raman, Elétrica, Tensiometria)

                st.success(
                    f"✅ Experimento registrado com sucesso!\n\n"
                    f"ID do experimento: `{experiment_id}`\n\n"
                    "➡ Agora processe o arquivo no módulo correspondente."
                )

                st.info(
                    "Próximo passo:\n"
                    "- Raman → Aba Molecular\n"
                    "- Elétrica → Aba Resistividade\n"
                    "- Tensiometria → Aba Físico-mecânica"
                )

        except Exception as e:
            st.error("❌ Erro ao registrar experimento.")
            st.exception(e)
