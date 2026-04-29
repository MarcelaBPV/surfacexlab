def process_tensiometry(file_like, theta_by_liquid, ID_IG, I2D_IG):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # =====================================================
    # LEITURA INTELIGENTE
    # =====================================================
    name = file_like.name.lower()

    if name.endswith(".log"):

        df = pd.read_csv(
            file_like,
            delim_whitespace=True,
            skiprows=1
        )

        df.columns = [c.strip().lower() for c in df.columns]

        if "mean" not in df.columns:
            raise ValueError("Coluna 'Mean' não encontrada no .LOG")

        theta = pd.to_numeric(df["mean"], errors="coerce")

        time = pd.to_numeric(df["time"], errors="coerce")

    else:
        # Excel / CSV
        if name.endswith(("xlsx", "xls")):
            df = pd.read_excel(file_like)
        else:
            df = pd.read_csv(file_like)

        df.columns = [str(c).lower().strip() for c in df.columns]

        angle_col = None

        for c in df.columns:
            if "angle" in c or "theta" in c:
                angle_col = c

        if angle_col is None:
            raise ValueError("Coluna de ângulo não encontrada")

        theta = pd.to_numeric(df[angle_col], errors="coerce")

        time = np.arange(len(theta))

    # =====================================================
    # LIMPEZA (CRÍTICO)
    # =====================================================
    theta = theta.dropna()

    # remove valores absurdos (erro experimental)
    theta = theta[(theta > 5) & (theta < 150)]

    # remove outliers estatísticos
    q1, q3 = np.percentile(theta, [25, 75])
    iqr = q3 - q1

    theta = theta[
        (theta > q1 - 1.5 * iqr) &
        (theta < q3 + 1.5 * iqr)
    ]

    # =====================================================
    # ESTADO ESTÁVEL (últimos 30%)
    # =====================================================
    n = len(theta)
    stable = theta[int(0.7 * n):]

    q_star = stable.mean()
    rrms = stable.std()

    # =====================================================
    # ENERGIA SUPERFICIAL (OWRK simplificado)
    # =====================================================
    gamma_total = np.mean(list(theta_by_liquid.values()))
    gamma_d = gamma_total * 0.6
    gamma_p = gamma_total * 0.4

    # =====================================================
    # CLASSIFICAÇÃO
    # =====================================================
    if q_star < 90:
        wetting = "Hidrofílico"
    else:
        wetting = "Hidrofóbico"

    diagnostic = "Superfície estável"

    if rrms > 5:
        diagnostic += " + instabilidade de gota"

    # =====================================================
    # PLOT (NÍVEL PAPER)
    # =====================================================
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)

    ax.plot(time[:len(theta)], theta, label="Experimental")
    ax.axhline(q_star, linestyle="--", label="q*")

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo (°)")
    ax.set_title("Dinâmica do ângulo de contato")

    ax.legend()
    ax.grid(alpha=0.3)

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = {
        "q* (°)": q_star,
        "Rrms (mm)": rrms,
        "Molhabilidade": wetting,
        "Energia superficial (mJ/m²)": gamma_total,
        "Componente dispersiva": gamma_d,
        "Componente polar": gamma_p,
        "Diagnóstico": diagnostic,
        "ID/IG": ID_IG,
        "I2D/IG": I2D_IG,
    }

    return {
        "summary": summary,
        "figure": fig
    }
