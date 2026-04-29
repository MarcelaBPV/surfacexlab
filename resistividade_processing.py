# =========================================================
# AJUSTE LINEAR CORRETO (V eixo X)
# =========================================================
def linear_fit(V, I):

    A = np.vstack([V, np.ones_like(V)]).T
    slope, intercept = np.linalg.lstsq(A, I, rcond=None)[0]

    I_pred = slope * V + intercept

    ss_res = np.sum((I - I_pred) ** 2)
    ss_tot = np.sum((I - np.mean(I)) ** 2)

    R2 = 1 - ss_res / ss_tot

    return slope, intercept, R2, I_pred


# =========================================================
# PIPELINE
# =========================================================
def process_resistivity(file_like, thickness_m, geometry="four_point_film"):

    df = read_iv_file(file_like)

    I = df["I"].values
    V = df["V"].values

    # 👉 AJUSTE CORRETO
    slope, offset, R2, I_pred = linear_fit(V, I)

    # 👉 Resistência = 1 / slope (pois I = V/R)
    R = 1 / slope if slope != 0 else np.nan

    rho = resistivity_4p(R, thickness_m)
    sigma = 1 / rho if rho > 0 else np.nan
    Rs = rho / thickness_m

    # diagnóstico
    regime = "Ôhmico" if R2 > 0.98 else "Não ôhmico"
    if abs(offset) > 0.01:
        regime += " + Contato não ideal"

    classe = classify(sigma)

    # =====================================================
    # GRÁFICO CORRETO (igual paper)
    # =====================================================
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)

    ax.scatter(V, I, label="Experimental")
    ax.plot(V, I_pred, label=f"Ajuste (R²={R2:.4f})")

    ax.set_xlabel("Tensão (V)")
    ax.set_ylabel("Corrente (A)")
    ax.set_title("Curva I–V")

    ax.legend()
    ax.grid(alpha=0.3)

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {
        "Resistência (Ω)": R,
        "Resistividade (Ω·m)": rho,
        "Condutividade (S/m)": sigma,
        "Sheet Resistance (Ω/sq)": Rs,
        "R²": R2,
        "Offset (A)": offset,
        "Regime": regime,
        "Classe": classe,
    }

    return {
        "df": df,
        "summary": summary,
        "figure": fig
    }
