# raman_deconvolution_plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.special import wofz


# =========================================================
# PERFIL VOIGT
# =========================================================
def voigt(x, amp, cen, sigma, gamma):
    z = ((x-cen)+1j*gamma)/(sigma*np.sqrt(2))
    return amp*np.real(wofz(z))/(sigma*np.sqrt(2*np.pi))


def multi_voigt(x, *params):

    y = np.zeros_like(x)

    for i in range(0,len(params),4):
        amp, cen, sigma, gamma = params[i:i+4]
        y += voigt(x,amp,cen,sigma,gamma)

    return y


# =========================================================
# LEITURA ARQUIVO
# =========================================================
def read_spectrum(file):

    df = pd.read_csv(
        file,
        sep=r"\s+|\t+|;|,",
        engine="python",
        header=None,
        decimal=","
    )

    df = df.dropna()

    wave = df.iloc[:,0].astype(float).values
    intensity = df.iloc[:,1].astype(float).values

    idx = np.argsort(wave)

    return wave[idx], intensity[idx]


# =========================================================
# SCRIPT PRINCIPAL
# =========================================================
def main():

    file = "seu_arquivo.txt"

    x,y = read_spectrum(file)

    # ----------------------------------
    # recorte espectral
    # ----------------------------------
    mask = (x>=1500) & (x<=1700)

    x = x[mask]
    y = y[mask]

    # ----------------------------------
    # suavização
    # ----------------------------------
    y = savgol_filter(y,11,3)

    # ----------------------------------
    # picos iniciais estimados
    # ----------------------------------
    init = [

        400,1543,15,15,
        700,1559,15,15,
        900,1579,20,20,
        300,1600,15,15,
        400,1620,15,15,
        300,1631,15,15,
        250,1660,20,20
    ]

    popt,_ = curve_fit(
        multi_voigt,
        x,
        y,
        p0=init,
        maxfev=50000
    )

    y_fit = multi_voigt(x,*popt)

    # ----------------------------------
    # plot
    # ----------------------------------
    plt.figure(figsize=(6,4),dpi=300)

    plt.plot(x,y,'k',lw=1)

    colors = [
        "#ff6666",
        "#6699cc",
        "#ffcc66",
        "#66cc99",
        "#9999cc",
        "#cccccc",
        "#ff9999"
    ]

    for i in range(0,len(popt),4):

        amp,cen,sigma,gamma = popt[i:i+4]

        curve = voigt(x,amp,cen,sigma,gamma)

        plt.fill_between(
            x,
            curve,
            alpha=0.6,
            color=colors[i//4]
        )

        plt.text(
            cen,
            np.max(curve)*1.05,
            f"{cen:.0f}",
            ha="center",
            fontsize=9
        )

    plt.plot(x,y_fit,'r--',lw=1)

    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Intensity")

    plt.xlim(1500,1700)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
