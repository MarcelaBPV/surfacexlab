def show_sample_history(sample, supabase):

    st.subheader(f"📂 Histórico — {sample}")

    df_hist = load_sample_history(sample, supabase)

    if df_hist.empty:
        st.info("Sem histórico ainda")
        return

    st.dataframe(df_hist)

    # gráfico evolução
    if "resistividade" in df_hist.columns:

        fig = px.line(
            df_hist,
            x="created_at",
            y="resistividade",
            title="Evolução da Resistividade"
        )

        st.plotly_chart(fig, use_container_width=True)
