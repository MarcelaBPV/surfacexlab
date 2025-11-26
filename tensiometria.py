# tensiometria.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_contact_angle(file, fit_order=3):
    """Lê arquivo de log com colunas Time, Theta(L), Theta(R), Mean.

    Faz ajuste polinomial do ângulo médio vs tempo, calcula dθ/dt e razão cos(γ).
    """
    df = pd.read_csv(file, delim_whitespace=True, comment="#")
    if "Mean" in df.columns:
        df = df.rename(columns={"Time": "t_seconds", "Mean": "angle_mean_deg"})

    t = df["t_seconds"].values
    theta = df["angle_mean_deg"].values

    coef = np.polyfit(t, theta, fit_order)
    poly = np.poly1d(coef)
    theta_fit = poly(t)
    dtheta_dt = np.gradient(theta_fit, t)
    gamma_ratio = float(np.mean(np.cos(np.radians(theta_fit))))

    fig, ax = plt.subplots()
    ax.plot(t, theta, "bo", label="Experimental")
    ax.plot(t, theta_fit, "r-", label=f"Ajuste polinomial ({fit_order}º grau)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Ângulo de contato (°)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    return {
        "df": df,
        "coef": coef.tolist(),
        "gamma_ratio": gamma_ratio,
        "dtheta_dt": dtheta_dt.tolist(),
        "figure": fig,
    }
