# resistividade_tab.py
# -*- coding: utf-8 -*-
"""
Aba 3 — Resistividade por método de 4 pontas
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List
from io import StringIO

from resistividade import process_resistivity  # seu módulo de cálculo


def _safe_insert(supabase, table: str, records: List[Dict]):
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    if not records:
        return []
    res = supabase.table(table).insert(records).execute()
    return getattr(res, "data", None) or []


def render_resistividade_tab(supabase):
    st.header("3️⃣ Resistividade — Método de 4 Pontas")

    st.markdown(
        """
Envie um arquivo **CSV** com colunas (ou equivalentes):

`current_a` | `voltage_v`  

O módulo `resistividade.py` irá:
- Ajustar curva I x V (regressão linear)
- Calcular resistência R, resistividade ρ, condutividade σ
- Classificar como **Condutor**, **Semicondutor** ou **Isolante**.
        """
    )

    uploaded = st.file_uploader("Arquivo CSV (corrente x tensão)", type=["csv"])
    thickness_nm = st.number_input(
        "Espessura do filme (nm)", min_value=1.0, value=200.0, step=10.0
    )
    mode = st.selectbox("Modo de cálculo", ["filme", "bulk"], index=0)

    sample_name = st.text_input("Nome da amostra", value="amostra_resist_1")
    material = st.text_input("Material / processo", value="")
    notes = st.text_area("Observações (opcional)", value="")

    if uploaded:
        try:
            # Se quiser garantir leitura, podemos passar uploaded diretamente para process_resistivity,
            # mas aqui mostramos também a tabela em Streamlit:
            df_preview = pd.read_csv(uploaded)
            st.subheader("Pré-visualização dos dados")
            st.dataframe(df_preview.head())

            # Precisamos reposicionar o cursor do arquivo para reuso
            uploaded.seek(0)
            thickness_m = thickness_nm * 1e-9
            result = process_resistivity(uploaded, thickness_m=thickness_m, mode=mode)

            df = result["df"]
            R = result["R"]
            rho = result["rho"]
            sigma = result["sigma"]
            classe = result["classe"]
            R2 = result["R2"]
            fig = result["figure"]

            st.pyplot(fig)

            st.markdown(f"**Resistência R (Ω):** `{R:.4e}`")
            st.markdown(f"**Resistividade ρ (Ω·m):** `{rho:.4e}`")
            st.markdown(f"**Condutividade σ (S/m):** `{sigma:.4e}`")
            st.markdown(f"**Classe do material:** `{classe}`")
            st.markdown(f"**R² do ajuste:** `{R2:.4f}`")

            st.subheader("Tabela completa (após leitura)")
            st.dataframe(df)

            st.download_button(
                "⬇️ Baixar tabela (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{sample_name}_IV_data.csv",
                mime="text/csv",
            )

            if supabase:
                if st.button("Salvar resultado de Resistividade no Supabase"):
                    try:
                        summary_rec = {
                            "sample_name": sample_name,
                            "material": material,
                            "mode": mode,
                            "thickness_m": float(thickness_m),
                            "R_ohm": float(R),
                            "rho_ohm_m": float(rho),
                            "sigma_S_m": float(sigma),
                            "class_label": classe,
                            "r2": float(R2),
                            "notes": notes,
                        }
                        summary_rows = _safe_insert(
                            supabase, "resistivity_summary", [summary_rec]
                        )
                        if summary_rows:
                            summary_id = summary_rows[0]["id"]
                            df_save = df.copy()
                            df_save["resistivity_id"] = summary_id
                            records = df_save.to_dict(orient="records")
                            _safe_insert(supabase, "resistivity_data", records)
                            st.success(
                                f"Dados salvos no Supabase (resistivity_summary.id={summary_id})."
                            )
                        else:
                            st.error("Não foi possível obter o ID do resumo salvo.")
                    except Exception as e:
                        st.error(f"Erro ao salvar no Supabase: {e}")
            else:
                st.info("Configure o Supabase em st.secrets para habilitar salvamento.")

        except Exception as e:
            st.error(f"Erro ao processar arquivo de resistividade: {e}")
    else:
        st.info("Envie um arquivo de I x V para começar.")
