# =========================================================
# FIT WITH PHYSICAL CONSTRAINTS
# =========================================================

lower_bounds = []
upper_bounds = []

for peak in peaks:

    lower_bounds.extend([

        0,                  # Amplitude
        wave[peak] - 15,    # Center
        0.1,                # Sigma
        0.1,                # Gamma
        0                   # Eta
    ])

    upper_bounds.extend([

        2,                  # Amplitude
        wave[peak] + 15,    # Center
        30,                 # Sigma
        30,                 # Gamma
        1                   # Eta
    ])

# =========================================================
# CURVE FIT
# =========================================================
try:

    popt, _ = curve_fit(

        multi_pseudo_voigt,

        wave,

        norm,

        p0=initial_params,

        bounds=(

            lower_bounds,

            upper_bounds
        ),

        maxfev=30000
    )

    fit = multi_pseudo_voigt(

        wave,

        *popt
    )

except:

    fit = norm
    popt = initial_params

# =========================================================
# RESIDUAL
# =========================================================
residual = norm - fit

ss_res = np.sum(residual**2)

ss_tot = np.sum(

    (norm - np.mean(norm))**2
)

r2 = 1 - (ss_res / ss_tot)

# =========================================================
# PUBLICATION STYLE FIGURE
# =========================================================
plt.rcParams.update({

    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1,
    "ytick.major.width": 1
})

fig = plt.figure(

    figsize=(10,7),

    dpi=600
)

gs = fig.add_gridspec(

    2,

    1,

    height_ratios=[4,1],

    hspace=0.08
)

ax1 = fig.add_subplot(gs[0])

ax2 = fig.add_subplot(gs[1])

# =========================================================
# EXPERIMENTAL CURVE
# =========================================================
ax1.plot(

    wave,

    norm,

    color="black",

    linewidth=2.2,

    label="Experimental"
)

# =========================================================
# FIT CURVE
# =========================================================
ax1.plot(

    wave,

    fit,

    "--",

    color="red",

    linewidth=2,

    label=f"Fit (R²={r2:.4f})"
)

# =========================================================
# COMPONENTS
# =========================================================
peak_table = []

n_peaks = len(popt)//5

colors = [

    "#f94144",
    "#f3722c",
    "#f9c74f",
    "#90be6d",
    "#43aa8b",
    "#577590",
    "#9b5de5",
    "#f15bb5"
]

for i in range(n_peaks):

    A = popt[i*5]
    x0 = popt[i*5 + 1]
    sigma = popt[i*5 + 2]
    gamma = popt[i*5 + 3]
    eta = popt[i*5 + 4]

    component = pseudo_voigt(

        wave,

        A,

        x0,

        sigma,

        gamma,

        eta
    )

    molecule = identify_peak(x0)

    # =====================================================
    # FWHM
    # =====================================================
    fwhm = 0.5346 * (2*gamma) + np.sqrt(

        0.2166*(2*gamma)**2 +

        (2.355*sigma)**2
    )

    # =====================================================
    # COMPONENT AREA
    # =====================================================
    ax1.fill_between(

        wave,

        0,

        component,

        alpha=0.35,

        color=colors[i % len(colors)]
    )

    # =====================================================
    # COMPONENT LINE
    # =====================================================
    ax1.plot(

        wave,

        component,

        linewidth=1.2,

        color=colors[i % len(colors)]
    )

    # =====================================================
    # PEAK NUMBER
    # =====================================================
    ax1.text(

        x0,

        np.max(component) + 0.02,

        f"{i+1}",

        fontsize=9,

        ha="center",

        fontweight="bold"
    )

    # =====================================================
    # TABLE
    # =====================================================
    peak_table.append({

        "Peak ID": i+1,

        "Peak (cm⁻¹)": round(x0,2),

        "Assignment": molecule,

        "Intensity": round(A,4),

        "FWHM": round(fwhm,2),

        "σ Gaussian": round(sigma,2),

        "γ Lorentzian": round(gamma,2),

        "η Mixing": round(eta,2)
    })

# =========================================================
# RESIDUAL
# =========================================================
ax2.plot(

    wave,

    residual,

    color="gray",

    linewidth=1.5
)

ax2.axhline(

    0,

    linestyle="--",

    linewidth=1,

    color="steelblue"
)

# =========================================================
# STYLE
# =========================================================
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# =========================================================
# LABELS
# =========================================================
ax1.set_ylabel(

    "Normalized Intensity",

    fontsize=15
)

ax2.set_ylabel(

    "Residual",

    fontsize=13
)

ax2.set_xlabel(

    "Raman Shift (cm⁻¹)",

    fontsize=15
)

# =========================================================
# LIMITS
# =========================================================
ax1.set_xlim(
    shift_min,
    shift_max
)

ax2.set_xlim(
    shift_min,
    shift_max
)

# =========================================================
# GRID
# =========================================================
ax1.grid(alpha=0.15)
ax2.grid(alpha=0.15)

# =========================================================
# LEGEND — RIGHT CORNER
# =========================================================
ax1.legend(

    fontsize=11,

    loc="upper right",

    frameon=True
)

# =========================================================
# TICKS
# =========================================================
ax1.tick_params(

    axis='both',

    labelsize=12
)

ax2.tick_params(

    axis='both',

    labelsize=11
)

# =========================================================
# REMOVE TITLE
# =========================================================
