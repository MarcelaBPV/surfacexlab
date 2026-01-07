# upload_tab.py
# -*- coding: utf-8 -*-

"""
SurfaceXLab — Upload Centralizado de Experimentos

Regra de ouro:
➡ TODO experimento entra no sistema via UPLOAD DE ARQUIVO.

Tipos suportados:
1) Molecular — Raman (sangue)
2) Elétrica — Resistividade (motores / filmes)
3) Físico-mecânica — Tensiometria (nanotubos / superfícies)

O processamento ocorre SOMENTE nos módulos específicos.
"""

import streamlit as st
from datetime import date


# =========================================================
# HELPERS — SUPABASE
# =========================================================
def get_samples(supabase):
    """Lista amostras cadastradas."""
    try:
        res = (
            supabase
            .table("samples")
            .select("id, sample_code")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data if res.data else []
    except Exception as e:
        st.error("❌ Erro ao carregar amostras.")
        st.exception(e)
        return []


def create_experiment(
    supabase,
    sample_id: str,
    experiment_type: str,
    operator: str,
    equipment: str,
    notes: str,
):
    """Cria um experimento genérico (evento CRM)."""
    res = (
        supabase
        .table("experiments")
        .insert({
            "sample_id": sample_id,
            "experiment_type": experiment_type,
            "operator": operator or None,
            "equipment": equipment or None,
            "notes": notes or None,
            "experiment_date": str(date.today()),
        })
        .execute()
    )

    if not res.data:
        raise RuntimeError("Falha ao criar experimento no banco.")

    return res.data[0]["id"]


# =========================================================
# UI — UPLOAD TAB
# =========================================================
def render_upload_tab(supabase):
    st.header("Upload de Experimentos")

    st.markdown(
        """
        Este módulo é o **ponto único de entrada de dados experimentais** no SurfaceXLab.

         **Fluxo do sistema:**
        1. Selecionar amostra
        2. Definir tipo de experimento
        3. Fazer upload do arquivo bruto
        4. Processar posteriormente no módulo correspondente
        """
    )

    # -----------------------------------------------------
    # 1️⃣ SELEÇÃO DA AMOSTRA
    # -----------------------------------------------------
    st.subheader("1️⃣ Amostra")

    samples = get_samples(supabase)

    if not samples:
        st.warning("Nenhuma amostra cadastrada. Cadastre uma amostra antes do upload.")
        return

    sample_map = {s["sample_code"]: s["id"] for s in samples}

    sample_code = st.selectbox(
        "Selecione a amostra",
        options=list(sample_map.keys()),
    )
    sample_id = sample_map[sample_code]

    # -----------------------------------------------------
    # 2️⃣ TIPO DE EXPERIMENTO
    # -----------------------------------------------------
    st.subheader("2️⃣ Tipo de experimento")

    experiment_label = st.radio(
        "Categoria",
        [
            "🧬 Molecular — Raman (sangue)",
            "⚡ Elétrica — Resistividade (motores)",
            "💧 Físico-mecânica — Tensiometria (nanotubos)",
        ],
    )

    if experiment_label.startswith("🧬"):
        experiment_type = "Raman"
        accepted_files = ["csv", "txt", "xlsx"]
        equipment_default = "Raman Spectrometer"

    elif experiment_label.startswith("⚡"):
        experiment_type = "Electrical"
        accepted_files = ["csv", "xlsx"]
        equipment_default = "Source Measure Unit / Multimeter"

    else:
        experiment_type = "Tensiometry"
        accepted_files = ["csv", "xlsx"]
        equipment_default = "Goniometer / Tensiometer"

    # -----------------------------------------------------
    # 3️⃣ METADADOS
    # -----------------------------------------------------
    st.subheader("3️⃣ Metadados do experimento")

    col1, col2 = st.columns(2)
    with col1:
        operator = st.text_input("Operador / Responsável")
    with col2:
        equipment = st.text_input("Equipamento", value=equipment_default)

    notes = st.text_area(
        "Observações",
        placeholder="Condições experimentais, observações relevantes, etc.",
    )

    # -----------------------------------------------------
    # 4️⃣ UPLOAD DO ARQUIVO
    # -----------------------------------------------------
    st.subheader("4️⃣ Upload do arquivo bruto")

    uploaded_file = st.file_uploader(
        "Arquivo experimental",
        type=accepted_files,
        help="Este arquivo será processado posteriormente no módulo específico.",
    )

    # -----------------------------------------------------
    # 5️⃣ REGISTRO DO EXPERIMENTO
    # -----------------------------------------------------
    if st.button("💾 Registrar experimento", use_container_width=True):

        if not uploaded_file:
            st.warning("Selecione um arquivo antes de continuar.")
            return

        try:
            with st.spinner("Registrando experimento no sistema..."):

                experiment_id = create_experiment(
                    supabase=supabase,
                    sample_id=sample_id,
                    experiment_type=experiment_type,
                    operator=operator,
                    equipment=equipment,
                    notes=notes,
                )

            st.success("✅ Experimento registrado com sucesso!")

            st.markdown(
                f"""
                **Resumo do registro**
                - Amostra: `{sample_code}`
                - Tipo: `{experiment_type}`
                - Arquivo: `{uploaded_file.name}`
                - ID do experimento: `{experiment_id}`
                """
            )

            st.info(
                "➡ **Próximo passo:**\n\n"
                "- Raman → Aba *Molecular*\n"
                "- Elétrica → Aba *Resistividade*\n"
                "- Tensiometria → Aba *Físico-mecânica*"
            )

        except Exception as e:
            st.error("❌ Erro ao registrar experimento.")
            st.exception(e)
