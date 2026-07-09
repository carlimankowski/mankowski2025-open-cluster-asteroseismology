#!/usr/bin/env python3
"""
make_appendix_all.py — Regenerate ALL 23 appendix diagnostic figures using
the current full_seismic_results.csv numax and dnu values.

Replaces both the old pipeline-generated figures AND the 3 new target figures,
ensuring all appendix figures use consistent, up-to-date measurements.

Panel (a): Log-log power spectrum + Harvey BG (log-spaced binning)
Panel (b): Background-corrected SNR around numax (with peak envelope)
Panel (c): 2D échelle diagram via the 'echelle' package (BuPu colormap)

Output: figures/appendix/TIC{tic}.jpeg  AND  Downloads/revision_1/TIC{tic}.jpeg
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, maximum_filter1d
from scipy.optimize import curve_fit
from echelle import plot_echelle
import os, warnings
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
PS_DIR  = '/Users/carlimankowski/research/revision/proper_ps'
CSV     = '/Users/carlimankowski/research/revision/full_seismic_results.csv'
OUTDIR  = '/Users/carlimankowski/research/revision/figures/appendix'
DL_DIR  = '/Users/carlimankowski/Downloads/revision_1'

df = pd.read_csv(CSV)
print(f'Loaded {len(df)} targets from CSV')

# ─── Harvey model ─────────────────────────────────────────────────────────────
def harvey1(f, a, b, wn):
    return a / (1.0 + (f / b) ** 4) + wn


def harvey2(f, a1, b1, a2, b2, wn):
    """Two-component Harvey profile + white noise (Kallinger et al. 2014 form)."""
    return a1 / (1.0 + (f / b1) ** 4) + a2 / (1.0 + (f / b2) ** 4) + wn


def fit_background(freq, power, numax, dnu_est):
    """Two-component Harvey + white-noise background, fit over the full spectrum
    with the oscillation region masked out, following standard practice
    (Kallinger et al. 2014; Mathur et al. 2011; pySYD). Returns (bg_array, wn).

    The two granulation components capture both the steep low-frequency rise and
    the granulation near numax, so the model tracks the data across the whole
    range rather than only in a narrow window around numax.
    """
    # mask the oscillation power excess so it does not bias the background
    osc_lo = max(0.1, numax - 4.0 * dnu_est)
    osc_hi = numax + 4.0 * dnu_est
    bg_mask = (freq > 0.1) & ((freq < osc_lo) | (freq > osc_hi))
    if bg_mask.sum() < 40:
        wn = np.median(power)
        return np.full_like(power, wn), wn

    f_bg, p_bg = freq[bg_mask], power[bg_mask]
    # log-spaced binning (median in each bin) for a stable fit
    nbins = min(400, max(40, len(f_bg) // 30))
    edges = np.logspace(np.log10(f_bg[0]), np.log10(f_bg[-1]), nbins + 1)
    fb, pb = [], []
    for i in range(nbins):
        m = (f_bg >= edges[i]) & (f_bg < edges[i + 1])
        if m.sum() > 0:
            fb.append(f_bg[m].mean())
            pb.append(np.median(p_bg[m]))
    fb, pb = np.array(fb), np.array(pb)
    if len(fb) < 8:
        wn = np.median(p_bg)
        return np.full_like(power, wn), wn

    ok = np.isfinite(fb) & np.isfinite(pb) & (pb > 0)
    fb, pb = fb[ok], pb[ok]
    if len(fb) < 8:
        wn = np.median(p_bg)
        return np.full_like(power, wn), wn
    logpb = np.log10(pb)

    wn_guess = np.median(pb[-max(2, len(pb) // 5):])
    a_guess = max(pb[0] - wn_guess, 1e-20)
    # characteristic frequencies bracket numax (Kallinger et al. 2014 scaling)
    b1_guess = max(0.3, 0.317 * numax ** 0.970)
    b2_guess = max(1.0, 0.948 * numax ** 0.992)

    # Fit in log space so the ~7-decade dynamic range does not let the
    # low-frequency points dominate and mis-place the background near numax.
    def logh2(f, a1, b1, a2, b2, wn):
        return np.log10(harvey2(f, a1, b1, a2, b2, wn))

    def logh1(f, a, b, wn):
        return np.log10(harvey1(f, a, b, wn))

    try:
        popt, _ = curve_fit(
            logh2, fb, logpb,
            p0=[a_guess, b1_guess, 0.4 * a_guess, b2_guess, max(wn_guess, 1e-30)],
            bounds=([0, 0.05, 0, 0.1, 0],
                    [1e14, max(numax * 2, 5), 1e14, max(numax * 5, 20), 1e10]),
            maxfev=60000,
        )
        return harvey2(freq, *popt), max(popt[4], 1e-30)
    except Exception:
        try:
            popt, _ = curve_fit(
                logh1, fb, logpb,
                p0=[a_guess, b2_guess, max(wn_guess, 1e-30)],
                bounds=([0, 0.05, 0], [1e14, max(numax * 5, 20), 1e10]),
                maxfev=20000,
            )
            return harvey1(freq, *popt), max(popt[2], 1e-30)
        except Exception:
            wn = max(wn_guess, np.median(p_bg))
            return np.full_like(power, wn), wn


plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'figure.dpi': 150,
    'font.family': 'sans-serif',
})

n_ok = 0
n_fail = 0

for _, row in df.iterrows():
    tic_id  = int(row['TICID'])
    numax   = float(row['numax_final'])
    dnu     = float(row['dnu_center'])
    cluster = str(row['cluster'])
    source  = str(row['numax_source'])
    data_src = str(row['data_source'])

    ps_file = os.path.join(PS_DIR, f'{tic_id}_PS.txt')
    if not os.path.exists(ps_file):
        print(f'  TIC {tic_id}: MISSING PS file — skipping')
        n_fail += 1
        continue

    if not (np.isfinite(numax) and np.isfinite(dnu) and numax > 0 and dnu > 0):
        print(f'  TIC {tic_id}: invalid numax/dnu — skipping')
        n_fail += 1
        continue

    freq_raw, power_raw = np.loadtxt(ps_file, unpack=True)

    # Estimated dnu for masking oscillation window
    dnu_est = 135.146 * (numax / 3076.0) ** 0.77

    # Restrict to 0.5 – 12×numax µHz
    f_max = min(freq_raw[-1], max(numax * 12, 300))
    use   = (freq_raw > 0.5) & (freq_raw <= f_max)
    freq  = freq_raw[use]
    power = power_raw[use]

    if len(freq) < 100:
        print(f'  TIC {tic_id}: too few points after masking — skipping')
        n_fail += 1
        continue

    # Harvey background
    bg, wn_level = fit_background(freq, power, numax, dnu_est)
    snr = power / np.maximum(bg, 1e-30)

    # Log-spaced display smooth
    nbins_d = min(300, len(freq) // 50)
    bin_edges_d = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), nbins_d + 1)
    fb_d, pb_d = [], []
    for i in range(nbins_d):
        m = (freq >= bin_edges_d[i]) & (freq < bin_edges_d[i + 1])
        if m.sum() > 0:
            fb_d.append(freq[m].mean()); pb_d.append(power[m].mean())
    fb_d, pb_d = np.array(fb_d), np.array(pb_d)
    smooth_disp = uniform_filter1d(pb_d, size=max(1, len(pb_d) // 20))

    # SNR window around numax
    win_snr = max(4 * dnu, numax * 0.25)
    snr_mask = (freq >= numax - win_snr) & (freq <= numax + win_snr)
    f_snr = freq[snr_mask]
    s_snr = snr[snr_mask]

    # Échelle range
    lo_ech = max(freq[0], numax - 4.5 * dnu)
    hi_ech = min(freq[-1], numax + 4.5 * dnu)
    ech_mask = (freq >= lo_ech) & (freq <= hi_ech)
    f_ech = freq[ech_mask]
    s_ech = snr[ech_mask]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.38)
    ax_ps  = fig.add_subplot(gs[0])
    ax_snr = fig.add_subplot(gs[1])
    ax_ech = fig.add_subplot(gs[2])

    # Title — show source only for non-trivial sources
    src_tag = ''
    if source in ('Hon+21',):
        src_tag = f' [Hon+21]'
    elif source in ('envelope_revised', 'reviewer'):
        src_tag = ''   # don't clutter with internal labels
    title = f'TIC {tic_id} — {cluster.replace("_", " ")}{src_tag} [{data_src}]'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)

    # Panel (a): Log-log PS
    ax_ps.loglog(freq, power, color='#aaaaaa', lw=0.4, alpha=0.55)
    if len(fb_d) > 3:
        ax_ps.loglog(fb_d, smooth_disp, color='#2166ac', lw=1.6, alpha=0.85,
                     label='Smoothed PS')
    ax_ps.loglog(freq, bg, color='red', lw=1.8, label='Harvey BG')
    ax_ps.axhline(wn_level, color='#2166ac', lw=1.0, ls='--', alpha=0.6,
                  label='WN floor')
    ax_ps.axvline(numax, color='#377eb8', lw=1.8, ls='--',
                  label=fr'$\nu_{{\max}}={numax:.1f}$')
    ax_ps.set_xlabel(r'Frequency ($\mu$Hz)')
    ax_ps.set_ylabel(r'Power (ppm$^2\,\mu$Hz$^{-1}$)')
    ax_ps.set_title(f'(a) Power spectrum [{data_src}]')
    ax_ps.set_xlim(5.0, freq[-1] * 1.1)
    ax_ps.legend(fontsize=7.5, loc='upper right')

    # Panel (b): SNR around numax
    if len(f_snr) > 10:
        df_s = np.median(np.diff(f_snr))
        max_k = max(3, int(0.5 * dnu / df_s))
        env_k = max(3, int(0.8 * dnu / df_s))
        if max_k % 2 == 0: max_k += 1
        s_env = gaussian_filter1d(maximum_filter1d(s_snr, size=max_k), env_k)
        ax_snr.plot(f_snr, s_snr, color='#888888', lw=0.5, alpha=0.5)
        ax_snr.plot(f_snr, s_env, color='#d6604d', lw=2.0, label='Peak envelope')
        ax_snr.axhline(1.0, color='#2166ac', lw=1.0, ls='--', alpha=0.7)
        ax_snr.axvline(numax, color='#377eb8', lw=1.8, ls='--',
                       label=fr'$\nu_{{\max}}={numax:.1f}\,\mu$Hz')
        ax_snr.set_ylim(bottom=0)
        ax_snr.legend(fontsize=8, loc='upper right')
    ax_snr.set_xlabel(r'Frequency ($\mu$Hz)')
    ax_snr.set_ylabel('SNR (Power / Background)')
    ax_snr.set_title(fr'(b) $\nu_{{\max}}$ = {numax:.1f}')

    # Panel (c): Échelle
    if len(f_ech) > 50:
        try:
            plot_echelle(f_ech, s_ech, dnu, ax=ax_ech, smooth=True,
                         smooth_filter_width=max(1.0, dnu * 0.3), cmap='BuPu')
            ax_ech.set_title(fr'(c) Échelle ($\Delta\nu={dnu:.3f}\,\mu$Hz)')
        except Exception as e:
            ax_ech.text(0.5, 0.5, f'Échelle error:\n{e}',
                        ha='center', va='center', transform=ax_ech.transAxes, fontsize=8)
    else:
        ax_ech.text(0.5, 0.5, 'Insufficient data for échelle',
                    ha='center', va='center', transform=ax_ech.transAxes)

    # Footer
    dnu_corr = float(row.get('dnu_corr', np.nan)) if 'dnu_corr' in row.index else np.nan
    R_seis   = float(row.get('R_seis',   np.nan)) if 'R_seis'   in row.index else np.nan
    M_seis   = float(row.get('M_seis',   np.nan)) if 'M_seis'   in row.index else np.nan
    src_disp = source if source not in ('envelope', 'envelope_revised', 'nan', '') else 'this work'
    footer = (fr'$\nu_{{\max}}$ source: {src_disp} | '
              fr'$\Delta\nu_{{corr}}={dnu_corr:.3f}\,\mu$Hz | '
              fr'$R={R_seis:.2f}\,R_\odot$ | $M={M_seis:.3f}\,M_\odot$')
    fig.text(0.5, -0.03, footer, ha='center', fontsize=9, color='#555555')

    # Save
    for out in (os.path.join(OUTDIR, f'TIC{tic_id}.jpeg'),
                os.path.join(DL_DIR,  f'TIC{tic_id}.jpeg')):
        fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    n_ok += 1
    print(f'  TIC {tic_id} ({cluster}): numax={numax:.1f}, dnu={dnu:.3f} → saved')

print(f'\nDone: {n_ok} saved, {n_fail} skipped')
