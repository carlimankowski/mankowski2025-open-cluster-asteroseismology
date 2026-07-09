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

# NOTE: the previous monkeypatch of mist.interp_value scrambled the isochrone
# (out-of-order / duplicated points -> the spurious "curl" Jamie flagged).
# The stock isochrones API returns correctly EEP-ordered G/BP/RP mags, so we
# use it directly.
_ = types  # (import kept for compatibility)

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

    # Membership: keep stars consistent in parallax AND proper motion (max|z| < 2).
    # Field contaminants are dropped (not plotted) so the cluster sequence is clean,
    # as in recent Gaia cluster-CMD papers.
    member = np.isfinite(max_abs_z) & (max_abs_z < 2.0)
    good = good & member

    # distance modulus from the CLEAN members (robust to field contamination)
    plx_clean = pd.to_numeric(dfc["Plx"], errors="coerce").to_numpy()[good]
    med_plx_clean = np.nanmedian(plx_clean)
    dm_clean = 5 * np.log10(1000.0 / med_plx_clean) - 5 if med_plx_clean > 0 else np.nan
    print(f"  {cluster_pattern}: {len(dfc)} stars -> {good.sum()} clean members "
          f"| median plx={med_plx_clean:.3f} mas -> dm={dm_clean:.2f}")
    return color, gmag, good, ~member

def plot_isochrones(color, gmag, good, cluster_label, seismic_ages, lit_ages,
                    feh, ebv, dm, xlim, ylim, savefile):
    """Solid green = SEISMIC age range (this work, 2 curves); dashed orange =
    LITERATURE ages (3 curves). Distance from member parallax; reddening/metallicity
    chosen so the isochrone traces the observed cluster sequence."""
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.scatter(color[good], gmag[good], s=15, color='gray',
               edgecolors='k', linewidth=0.5, alpha=0.75, label=cluster_label, zorder=1)

    def iso_curve(age):
        iso = mist.isochrone(np.log10(age * 1e9), feh)
        mabs = iso['G_mag'].to_numpy()
        keep = (mabs > -3) & (mabs < 10)          # MS -> turnoff -> lower RGB (drop PMS/tip)
        col = ((iso['BP_mag'] - iso['RP_mag']).to_numpy()[keep]) + (K_BP - K_RP) * ebv
        mag = mabs[keep] + dm + K_G * ebv
        return col, mag

    for k, age in enumerate(seismic_ages):
        c, m = iso_curve(age)
        lab = fr'Seismic (this work): {min(seismic_ages):.2f}--{max(seismic_ages):.2f} Gyr' if k == 0 else None
        ax.plot(c, m, color='#2ca02c', lw=2.6, ls='-', zorder=4, label=lab)
    for k, age in enumerate(lit_ages):
        c, m = iso_curve(age)
        lab = fr'Literature: {min(lit_ages):.2f}--{max(lit_ages):.2f} Gyr' if k == 0 else None
        ax.plot(c, m, color='#ff7f0e', lw=2.0, ls='--', alpha=0.85, zorder=3, label=lab)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)                     # given as (faint, bright) -> inverted display
    ax.set_xlabel(r'$G_{\rm BP} - G_{\rm RP}$ (mag)')
    ax.set_ylabel(r'$G$ (mag)')
    ax.legend(fontsize=10, loc='upper left')
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
    color, gmag, good,
    cluster_label="NGC 752",
    seismic_ages=[1.45, 1.71],
    lit_ages=[1.00, 1.30, 1.60],
    feh=0.0,                          # NGC 752 ~solar (literature)
    ebv=0.04, dm=8.22,                # dm from median member parallax (2.27 mas)
    xlim=(0.0, 1.5), ylim=(14, 8),
    savefile="ngciso.png",
)

# ═══════════════════════════════════════════════════════════════
# 2. Theia 6046
# ═══════════════════════════════════════════════════════════════
print("\n=== Theia 6046 ===")
color, gmag, good, bad = get_cluster_data(df, "Theia_6046")
plot_isochrones(
    color, gmag, good,
    cluster_label="Theia 6046",
    seismic_ages=[2.59, 6.69],
    lit_ages=[2.50, 3.50, 4.50],
    feh=-0.13,                        # median GSP-Spec member metallicity
    ebv=0.30, dm=9.61,                # dm from median member parallax (1.19 mas)
    xlim=(0.2, 2.5), ylim=(17, 9),
    savefile="theiaiso.png",
)

# ═══════════════════════════════════════════════════════════════
# 3. Casado-Alessi 1
# ═══════════════════════════════════════════════════════════════
print("\n=== Casado-Alessi 1 ===")
color, gmag, good, bad = get_cluster_data(df, "Casado-Alessi_1")
plot_isochrones(
    color, gmag, good,
    cluster_label="Casado-Alessi 1",
    seismic_ages=[0.88, 1.09],
    lit_ages=[0.69, 1.07, 1.45],
    feh=0.0,                          # ~solar; reddens lower MS to trace the data
    ebv=0.12, dm=9.24,                # dm from median member parallax (1.42 mas)
    xlim=(0.0, 1.5), ylim=(16, 8),
    savefile="casadoiso.png",
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
    color, gmag, good,
    cluster_label="Theia 844",
    seismic_ages=[0.24, 0.28],
    lit_ages=[0.10, 0.27, 0.44],
    feh=0.0,                          # ~solar (GSP-Spec member median +0.09)
    ebv=0.12, dm=9.09,                # dm from median member parallax (1.52 mas)
    xlim=(0.0, 2.0), ylim=(17, 6.5),  # extended up so the young turnoff is visible
    savefile="theia844iso.png",
)

print("\nAll 4 plots generated!")
