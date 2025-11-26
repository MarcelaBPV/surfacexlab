# tensiometria_tab.py
# -*- coding: utf-8 -*-
"""
Aba 2 — Tensiometria (Ângulo de contato)
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List
from io import StringIO

from tensiometria import process_contact_angle  # seu módulo de cálculo


def _safe_insert(supabase, table: str, records: List[Dict]):
    if not supabase:
        raise RuntimeError("Supabase não configurado.")
    if not records:
        return []
    res = supabase.table(table).insert(records).execute()
    return getattr(res, "data", None) or []


def render_tensiometria_tab(supabase):
    st.header("2️⃣ Tensiometria — Ângulo de Contato")

    st.markdown(
        """
Envie o **log de ângulo de contato** (por exemplo, `.txt`) com colunas:

`Time` | `Theta(L)` | `Theta(R)` | `Mean`  

O módulo `tensiometria.py` fará o ajuste polinomial e calcula:
- Curva ajustada do ângulo
- dθ/dt (taxa de variação)
- razão média de cos(γ)
        """
    )

    uploaded = st.file_uploader("Arquivo de log (txt/csv)", type=["txt", "csv"])
    fit_order = st.number_input(
        "Ordem do polinômio para o ajuste",
        min_value=1,
        max_value=6,
        value=3,
    )

    sample_name = st.text_input("Nome da amostra / superfície", value="amostra_tensio_1")
    material = st.text_input("Material / tratamento superficial", value="")
    notes = st.text_area("Observações (opcional)", value="")

    if uploaded:
        # processar
        try:
            # Para compatibilizar com pandas, embrulhamos o conteúdo em StringIO
            content = uploaded.read().decode("utf-8", errors="ignore")
            sio = StringIO(content)

            result = process_contact_angle(sio, fit_order=fit_order)

            df = result["df"]
            gamma_ratio = result["gamma_ratio"]
            coef = result["coef"]
            dtheta_dt = result["dtheta_dt"]
            fig = result["figure"]

            st.pyplot(fig)

            st.subheader("Tabela original do log")
            st.dataframe(df)

            # acrescentar colunas calculadas
            df_out = df.copy()
            df_out["dtheta_dt"] = dtheta_dt
            st.subheader("Dados com dθ/dt")
            st.dataframe(df_out)

            st.markdown(f"**Razão média cos(γ):** `{gamma_ratio:.4f}`")
            st.markdown(f"**Coeficientes do polinômio:** `{coef}`")

            st.download_button(
                "⬇️ Baixar dados com dθ/dt (CSV)",
                df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"{sample_name}_tensiometria.csv",
                mime="text/csv",
            )

            if supabase:
                if st.button("Salvar resultado de Tensiometria no Supabase"):
                    try:
                        # Tabela de resumo
                        summary_rec = {
                            "sample_name": sample_name,
                            "material": material,
                            "gamma_ratio": float(gamma_ratio),
                            "poly_order": int(fit_order),
                            "poly_coef": coef,  # Supabase armazena como jsonb
                            "notes": notes,
                        }
                        summary_rows = _safe_insert(
                            supabase, "tensiometry_summary", [summary_rec]
                        )
                        if summary_rows:
                            summary_id = summary_rows[0]["id"]
                            df_save = df_out.copy()
                            df_save["tensiometry_id"] = summary_id
                            records = df_save.to_dict(orient="records")
                            _safe_insert(supabase, "tensiometry_data", records)
                            st.success(
                                f"Dados salvos no Supabase (tensiometry_summary.id={summary_id})."
                            )
                        else:
                            st.error("Não foi possível obter o ID de resumo salvo.")
                    except Exception as e:
                        st.error(f"Erro ao salvar no Supabase: {e}")
            else:
                st.info("Configure o Supabase em st.secrets para habilitar salvamento.")

        except Exception as e:
            st.error(f"Erro ao processar arquivo de tensiometria: {e}")
    else:
        st.info("Envie um arquivo de tensiometria para começar.")
