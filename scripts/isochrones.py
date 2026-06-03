#!/usr/bin/env python3
"""Generate 4 publication-quality isochrone CMD plots for paper revision."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import types
import warnings
warnings.filterwarnings('ignore')

from isochrones import get_ichrone

# ── Load data ──
STAR_FILE = "/Users/carlimankowski/research/GitHub/Isochrones/final!_errors.csv"
df = pd.read_csv(STAR_FILE, low_memory=False)
cluster_col = "cluster_name"

SAVE_DIR = "/Users/carlimankowski/Downloads/revision_1/"

# ── MIST setup ──
mist = get_ichrone('mist', bands=['G', 'BP', 'RP'])

def patched_interp_value(self, pars, props):
    i0, i1, i2, i3, i4 = self.param_index_order
    pars = [pars[i0], pars[i1], pars[i2]]
    return self.model_grid.interp(pars, "all")

mist.interp_value = types.MethodType(patched_interp_value, mist)

# Gaia DR3 extinction coefficients
K_G  = 2.74
K_BP = 3.37
K_RP = 2.04

# ── Helper functions ──
def robust_stats(series):
    s = pd.to_numeric(series, errors="coerce")
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    sigma = 1.4826 * mad if mad > 0 else np.nan
    return med, sigma

def get_cluster_data(df, cluster_pattern):
    mask = df[cluster_col].astype(str).str.contains(cluster_pattern, case=False, na=False)
    dfc = df.loc[mask].copy()

    dfc["BP_RP"] = dfc["BPmag"] - dfc["RPmag"]
    color = dfc["BP_RP"].to_numpy()
    gmag = pd.to_numeric(dfc["Gmag"], errors="coerce").to_numpy()
    good = np.isfinite(color) & np.isfinite(gmag)

    # z-score outlier detection
    med_plx, sig_plx = robust_stats(dfc["Plx"])
    med_pmra, sig_pmra = robust_stats(dfc["pmRA"])
    med_pmde, sig_pmde = robust_stats(dfc["pmDE"])

    dfc["zPlx"] = (dfc["Plx"] - med_plx) / sig_plx if np.isfinite(sig_plx) and sig_plx > 0 else np.nan
    dfc["zpmRA"] = (dfc["pmRA"] - med_pmra) / sig_pmra if np.isfinite(sig_pmra) and sig_pmra > 0 else np.nan
    dfc["zpmDE"] = (dfc["pmDE"] - med_pmde) / sig_pmde if np.isfinite(sig_pmde) and sig_pmde > 0 else np.nan

    z_stack = np.vstack([
        np.abs(dfc["zPlx"].to_numpy()),
        np.abs(dfc["zpmRA"].to_numpy()),
        np.abs(dfc["zpmDE"].to_numpy())
    ])
    max_abs_z = np.nanmax(z_stack, axis=0)

    # Bad: max|z| >= 2
    bad_mask = np.isfinite(max_abs_z) & (max_abs_z >= 2.0)

    print(f"  {cluster_pattern}: {len(dfc)} stars, {good.sum()} plotted, {(good & bad_mask).sum()} non-members")
    return color, gmag, good, bad_mask

def plot_isochrones(color, gmag, good, bad_mask, cluster_label, ages_gyr, feh, ebv, dm,
                    xlim, ylim, savefile, label_text=None, iso_alpha=None, label_frac=None):
    fig, ax = plt.subplots(figsize=(9, 8))

    # Members
    ax.scatter(color[good], gmag[good], s=15, color='gray',
               edgecolors='k', linewidth=0.5, alpha=0.8, label=cluster_label)

    # Non-members
    bad_good = good & bad_mask
    if bad_good.any():
        ax.scatter(color[bad_good], gmag[bad_good],
                   marker='x', s=80, linewidth=2, color='red', label='Likely non-members')

    # Isochrones
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(ages_gyr)))
    for i, age in enumerate(ages_gyr):
        logage = np.log10(age * 1e9)
        iso = mist.isochrone(logage, feh)
        col_fit = (iso['BP_mag'] - iso['RP_mag']) + (K_BP - K_RP) * ebv
        mag_fit = iso['G_mag'] + dm + K_G * ebv

        alpha = iso_alpha.get(age, 1.0) if iso_alpha else 1.0
        legend_label = f"{age:.2f} Gyr"

        ax.plot(col_fit, mag_fit, color=colors_map[i], lw=2.5, alpha=alpha, label=legend_label)

        # Inline labels
        if label_text and age in label_text and label_text[age]:
            frac = label_frac.get(age, 0.25) if label_frac else 0.25
            frac = max(0.0, min(1.0, frac))
            idx = int(frac * (len(col_fit) - 1))
            ax.text(col_fit.iloc[idx] + 0.02, mag_fit.iloc[idx] - 0.10,
                    label_text[age], fontsize=8, color=colors_map[i], alpha=alpha)

    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$G_{\rm BP} - G_{\rm RP}$ (mag)')
    ax.set_ylabel(r'$G$ (mag)')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAVE_DIR + savefile, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {SAVE_DIR + savefile}")

# ═══════════════════════════════════════════════════════════════
# 1. NGC 752
# ═══════════════════════════════════════════════════════════════
print("\n=== NGC 752 ===")
color, gmag, good, bad = get_cluster_data(df, "NGC_752")
plot_isochrones(
    color, gmag, good, bad,
    cluster_label="NGC 752",
    ages_gyr=[1.30, 1.40, 1.50, 1.60],
    feh=0.1,
    ebv=0.035, dm=8.30,
    xlim=(0.0, 1.5), ylim=(14, 8),
    savefile="ngciso.png",
    label_text={1.30: "Literature", 1.55: "Seismic Age", 1.60: "Literature"},
    iso_alpha={1.30: 1.0, 1.40: 1.0, 1.50: 1.0, 1.60: 1.0},
    label_frac={1.30: 0.30, 1.60: 0.30}
)

# ═══════════════════════════════════════════════════════════════
# 2. Theia 6046
# ═══════════════════════════════════════════════════════════════
print("\n=== Theia 6046 ===")
color, gmag, good, bad = get_cluster_data(df, "Theia_6046")
plot_isochrones(
    color, gmag, good, bad,
    cluster_label="Theia 6046",
    ages_gyr=[0.3, 0.5, 0.7, 1.0, 1.5, 3.7],
    feh=-0.06,
    ebv=0.3, dm=9.9,
    xlim=(0.2, 2.5), ylim=(17, 9),
    savefile="theiaiso.png",
    label_text={
        0.5: "~ Seismic Range",
        0.7: "~ Seismic Range",
        3.7: "Literature Age",
    },
    iso_alpha={
        0.3: 0.20, 0.5: 1.0, 0.7: 1.0,
        1.0: 0.20, 1.5: 0.20, 3.7: 1.0,
    },
    label_frac={0.5: 0.30, 0.7: 0.30, 3.7: 0.32}
)

# ═══════════════════════════════════════════════════════════════
# 3. Casado-Alessi 1
# ═══════════════════════════════════════════════════════════════
print("\n=== Casado-Alessi 1 ===")
color, gmag, good, bad = get_cluster_data(df, "Casado-Alessi_1")
plot_isochrones(
    color, gmag, good, bad,
    cluster_label="Casado-Alessi 1",
    ages_gyr=[0.78, 0.80, 1.01, 1.45],
    feh=0.02,
    ebv=0.03, dm=9.0,
    xlim=(0.0, 1.5), ylim=(16, 8),
    savefile="casadoiso.png",
    label_text={
        0.78: "Literature",
        1.01: "Seismic Age",
        1.45: "Literature",
    },
    iso_alpha={0.78: 1.0, 0.80: 1.0, 1.01: 1.0, 1.45: 1.0},
    label_frac={0.78: 0.30, 1.01: 0.30, 1.45: 0.30}
)

# ═══════════════════════════════════════════════════════════════
# 4. Theia 844
# ═══════════════════════════════════════════════════════════════
print("\n=== Theia 844 ===")
color, gmag, good, bad = get_cluster_data(df, "Theia_844")

# Theia 844 parameters:
# Median parallax ~1.5 mas -> d ~ 667 pc -> DM = 5*log10(667)-5 = 9.12
# E(B-V) ~ 0.12 (moderate reddening in Cygnus direction)
# [Fe/H] ~ 0.0 (solar, averaging available spectroscopic values)
plot_isochrones(
    color, gmag, good, bad,
    cluster_label="Theia 844",
    ages_gyr=[0.15, 0.25, 0.35, 0.45],
    feh=0.0,
    ebv=0.12, dm=9.12,
    xlim=(0.0, 2.0), ylim=(17, 8),
    savefile="theia844iso.png",
    label_text={
        0.25: "~ Seismic Age",
        0.35: "~ Literature",
        0.45: "~ Literature",
    },
    iso_alpha={0.15: 0.5, 0.25: 1.0, 0.35: 1.0, 0.45: 1.0},
    label_frac={0.25: 0.30, 0.35: 0.30, 0.45: 0.30}
)

print("\nAll 4 plots generated!")
