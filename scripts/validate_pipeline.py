#!/usr/bin/env python3
"""
validate_published.py — end-to-end validation of the TESS numax pipeline on a
star with a precise, independent Kepler measurement.

Target: TIC 164557867 = KIC 10323222 = HD 175955 (K0 III red giant).
  Published (Yu et al. 2018, ApJS 236, 42; Kepler, 4 yr):
      numax = 47.02 +/- 0.79 uHz
      dnu   = 4.847 +/- 0.014 uHz
      Teff  = 4691 K
We run the IDENTICAL reprocessing + Harvey-background + Gaussian-envelope
pipeline used for the 23 cluster targets, but on the 5-sector TESS QLP data,
and check whether it recovers the Kepler numax. This directly answers the
referee's request to demonstrate the method on a known oscillator.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numax_diag import diagnostic, fit_background

REV = '/Users/carlimankowski/research/revision'
OUT = '/Users/carlimankowski/Downloads/revision_1'
os.makedirs(OUT, exist_ok=True)

TIC = 164557867
NUMAX_PUB, NUMAX_PUB_E = 47.02, 0.79      # Yu+2018 (Kepler)
DNU_PUB,   DNU_PUB_E   = 4.847, 0.014
TEFF_PUB = 4691.0

freq, power = np.loadtxt(f'{REV}/reprocessed/{TIC}_PS.txt', unpack=True)

# ── two-panel diagnostic (same as cluster targets) ───────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
nu_meas, amp = diagnostic(freq, power, NUMAX_PUB, DNU_PUB, a1, a2,
                          data_src='TESS QLP, 5 sectors')

# mark the published value on both panels for direct comparison
for ax in (a1, a2):
    ax.axvline(NUMAX_PUB, color='magenta', ls='-.', lw=1.5, zorder=7,
               label=fr'Kepler $\nu_{{\max}}$={NUMAX_PUB:.1f}')
    ax.legend(fontsize=8, loc=('lower left' if ax is a1 else 'upper right'))

dev = 100.0 * (nu_meas - NUMAX_PUB) / NUMAX_PUB
fig.suptitle(
    f"Validation — TIC {TIC} (KIC 10323222, K0 III)\n"
    f"Kepler $\\nu_{{\\max}}$ = {NUMAX_PUB:.2f} $\\mu$Hz   |   "
    f"TESS pipeline $\\nu_{{\\max}}$ = {nu_meas:.2f} $\\mu$Hz   "
    f"({dev:+.1f}%)",
    fontweight='bold', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p1 = f'{OUT}/validation_KIC10323222.jpeg'
fig.savefig(p1, dpi=180, bbox_inches='tight')
plt.close(fig)

# ── échelle diagram — same style as the per-target diagnostics (echelle pkg, BuPu) ──
from echelle import plot_echelle
bg, wn = fit_background(freq, power, NUMAX_PUB, DNU_PUB)
snr = power / np.maximum(bg, 1e-30)
band = (freq > NUMAX_PUB - 4 * DNU_PUB) & (freq < NUMAX_PUB + 4 * DNU_PUB)
f_ech, s_ech = freq[band], snr[band]

fig2, ax = plt.subplots(figsize=(5.4, 6.0))
plot_echelle(f_ech, s_ech, DNU_PUB, ax=ax, smooth=True,
             smooth_filter_width=max(1.0, DNU_PUB * 0.3), cmap='BuPu')
ax.set_title(fr'Échelle ($\Delta\nu={DNU_PUB:.3f}\,\mu$Hz)')
fig2.tight_layout()
p2 = f'{OUT}/validation_echelle_KIC10323222.jpeg'
fig2.savefig(p2, dpi=180, bbox_inches='tight')
plt.close(fig2)

print(f'Published (Kepler) numax = {NUMAX_PUB:.2f} +/- {NUMAX_PUB_E:.2f} uHz')
print(f'TESS pipeline    numax = {nu_meas:.2f} uHz   ({dev:+.1f}% deviation)')
print(f'Saved: {p1}')
print(f'Saved: {p2}')
