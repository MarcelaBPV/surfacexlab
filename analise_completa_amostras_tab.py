def build_dataset():

    if "samples_unified" not in st.session_state:
        return None

    data = []

    for sample, content in st.session_state.samples_unified.items():

        row = {"Amostra": sample}

        for key, val in content.items():

            # =========================
            # TRIPLICATA TENSIOMETRIA
            # =========================
            if key == "tensiometria" and isinstance(val, list):

                df_rep = pd.DataFrame(val)

                mean_vals = df_rep.mean(numeric_only=True)
                std_vals = df_rep.std(numeric_only=True)

                for col in mean_vals.index:
                    row[f"{col} (mean)"] = mean_vals[col]
                    row[f"{col} (std)"] = std_vals[col]

            # =========================
            # OUTROS DADOS
            # =========================
            elif isinstance(val, dict):
                row.update(val)

        data.append(row)

    return pd.DataFrame(data)
