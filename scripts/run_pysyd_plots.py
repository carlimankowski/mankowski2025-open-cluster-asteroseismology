"""
Batch pySYD run + 9-panel plot generation for all viable targets.
Fixes: (1) seed=None → seed=42; (2) m>k on low-numax → estimate=False
Output plots saved to revision/results/<TIC>/global_fit.png
"""
import os, sys, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightkurve as lk
from pysyd import utils, plots
from pysyd.target import Target

warnings.filterwarnings('ignore')

WORKDIR     = '/Users/carlimankowski/research/revision'
DATA_DIR    = os.path.join(WORKDIR, 'data')
INFO_DIR    = os.path.join(WORKDIR, 'info')
RESULTS_DIR = os.path.join(WORKDIR, 'results')
OUT_CSV     = os.path.join(WORKDIR, 'pysyd_results.csv')
for d in [DATA_DIR, INFO_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

NaN = float('nan')

# Best known numax for each target (reviewer > Hon > paper > scaling-relation estimate)
# Format: tic -> (cluster, numax_seed, lower_ex, upper_ex, use_estimate)
# use_estimate=False forces pySYD to use numax_seed directly (skips auto-detect)
# Stars with numax < 2 µHz removed — TESS cannot resolve sub-2µHz oscillations
# (would need years of continuous data; 5 TESS sectors gives ~0.07µHz resolution)
# Dropped: 427648004 (1.0), 346930285 (1.37), 16482780 (1.33), 382510863 (1.72),
#          459055638 (0.57), 24661083 (0.88), 24725114 (0.50), 43901963 (1.35),
#          279706340 (0.88), 332983572 (1.22), 427725221 (1.98)
TARGETS = {
    352774198: ('COIN-Gaia_30',      2.30,   1,   6,  False),
    306552813: ('Casado_Alessi_1',  66.16,  45, 100,  True),
    306955660: ('Casado_Alessi_1',  62.90,  40,  90,  True),
    306187638: ('Casado_Alessi_1',  68.06,  45, 100,  True),
    140143506: ('HSC_1986',        130.46,  80, 200,  True),
    402808611: ('HSC_2615',          2.98,   2,   6,  False),
    414752508: ('HSC_95',           38.36,  25,  60,  True),
    354016987: ('LISC_3534',         5.66,   3,  10,  False),
    437030530: ('NGC_2682',          3.00,   2,   6,  False),
    46305204:  ('NGC_2682',          2.10,   1,   4,  False),
    67569102:  ('NGC_752',          67.76,  45, 100,  True),
    67420118:  ('NGC_752',          90.00,  65, 120,  True),
    186970424: ('NGC_752',          90.00,  65, 120,  True),
    67419922:  ('NGC_752',          33.35,  15,  55,  True),
    150753375: ('Theia_1188',        3.91,   2,   7,  False),
    321823630: ('Theia_1297',       18.96,  10,  30,  True),
    306345133: ('Theia_6046',       34.60,  15,  55,  True),
    43902016:  ('Theia_6046',       15.80,   5,  30,  True),
    24297458:  ('Theia_6046',       32.10,  15,  55,  True),
    24666306:  ('Theia_6046',       35.30,  20,  55,  True),
    24444542:  ('Theia_6046',        9.00,   3,  20,  True),
    249064439: ('Theia_6046',       62.85,  40,  90,  True),
    34472483:  ('Theia_6046',       34.76,  20,  55,  True),
    24662412:  ('Theia_6046',        2.17,   2,   4,  False),
    195860623: ('Theia_844',        36.72,  20,  60,  True),
    195862014: ('Theia_844',        20.30,  10,  35,  True),
    20869256:  ('UPK_296',           2.59,   2,   5,  False),
    459055617: ('Unknown_3',         6.63,   3,  12,  False),
    # Blind (no logg) - wide windows, let pySYD estimate
    56551765:  ('HSC_1318',         50.0,    5, 300,  True),
    98689079:  ('HSC_2919',         50.0,    5, 300,  True),
    101168806: ('IC_348',           50.0,    5, 300,  True),
    195184975: ('NGC_2632',         50.0,    5, 200,  True),
    321894003: ('Renou_23',         50.0,    5, 200,  True),
    182395456: ('Theia_12',         50.0,    5, 200,  True),
    143093935: ('Theia_169',        50.0,    5, 200,  True),
    387813185: ('Theia_5',          50.0,    5, 200,  True),
    335695889: ('UPK_189',          50.0,    5, 300,  True),
    306552754: ('Unknown_1',        50.0,    5, 300,  True),
}

# ── LC download (skip cached) ─────────────────────────────────────────────────
def lc_download(tic, window=299, sigma=4.0):
    lc_path = os.path.join(DATA_DIR, f'{tic}_LC.txt')
    if os.path.exists(lc_path):
        return 'cached'
    sr = lk.search_lightcurve(f'TIC {tic}', author='SPOC', exptime=120)
    used = 'SPOC'
    if len(sr) == 0:
        sr = lk.search_lightcurve(f'TIC {tic}', author='QLP')
        used = 'QLP'
    if len(sr) == 0:
        raise ValueError('No LC found')
    lc_list = []
    for lc_raw in sr.download_all():
        lc_list.append(lc_raw.normalize().flatten(window_length=window).remove_outliers(sigma=sigma))
    lc = lk.LightCurveCollection(lc_list).stitch().normalize()
    times, fluxes = lc['time'].value, lc['flux'].value
    idx = np.argsort(times)
    times, fluxes = times[idx], fluxes[idx]
    for i in range(1, len(times)):
        if times[i] - times[i-1] > 10:
            times[i:] -= (times[i] - times[i-1])
    with open(lc_path, 'w') as f:
        for t, fl in zip(times, fluxes):
            if np.isfinite(t) and np.isfinite(fl):
                f.write(f'\t{t}\t{fl}\n')
    ps = os.path.join(DATA_DIR, f'{tic}_PS.txt')
    if os.path.exists(ps):
        os.remove(ps)
    return used

# ── pySYD run ────────────────────────────────────────────────────────────────
def run_one(tic, cluster, numax_seed, lower_ex, upper_ex, use_estimate):
    tic_str = str(tic)

    # Fix 1: fresh star_info with header only (no trailing newline — matches working notebook)
    info_csv = os.path.join(INFO_DIR, 'star_info.csv')
    with open(info_csv, 'w') as f:
        f.write('star,seed')

    params = utils.Parameters()
    params.add_targets(stars=tic_str)
    star = Target(tic_str, params)

    # power-excess window: ±25% of seed (tight enough to prevent wandering)
    lower_ps = numax_seed * 0.75
    upper_ps = numax_seed * 1.30
    # background window: scaled to the star's frequency range
    bg_upper = min(upper_ex * 4, 525)
    bg_lower = max(lower_ex * 0.1, 0.5)

    star.params.update({
        'verbose':    False,
        'mc_iter':    1,       # Fix: mc_iter>1 causes unequal-length arrays in _save_parameters
        'smooth_ps':  0,
        'show':       False,
        'overwrite':  True,
        'fix_wn':     False,
        'save':       False,   # Fix: bypass pySYD's broken _save_parameters pandas call
        'inpdir':     DATA_DIR,
        'infdir':     INFO_DIR,
        'outdir':     RESULTS_DIR,
        'lower_ex':   lower_ex,
        'upper_ex':   upper_ex,
        'lower_ps':   lower_ps,
        'upper_ps':   upper_ps,
        'lower_bg':   bg_lower,
        'upper_bg':   bg_upper,
        'numax':      numax_seed,
        'dnu':        0.263 * numax_seed**0.772,
        'seed':       42,
        'selected':   1,
        'bg_components': 1,
    })

    # Fix 2: skip auto-estimate for low-numax stars to avoid m>k spline error
    if not use_estimate:
        star.params['estimate'] = False

    star.load_data()
    star.process_star()

    # Generate and save 9-panel plot
    star.params['show'] = False
    plots.plot_parameters(star)

    numax_out = (star.params.get('numax_smorhcv')
                 or star.params.get('numax_smoo')
                 or star.params.get('numax'))
    dnu_out   = star.params.get('obs_dnu') or star.params.get('dnu')
    snr_out   = star.params.get('snr', float('nan'))

    return {
        'tic':      tic,
        'cluster':  cluster,
        'numax':    float(numax_out) if numax_out else float('nan'),
        'dnu':      float(dnu_out)   if dnu_out   else float('nan'),
        'snr':      float(snr_out)   if snr_out   else float('nan'),
    }

# ── Main batch loop ───────────────────────────────────────────────────────────
print('=== Downloading LCs ===', flush=True)
dl_failed = []
for tic in TARGETS:
    try:
        used = lc_download(tic)
        print(f'  {tic}: {used}', flush=True)
    except Exception as e:
        print(f'  {tic}: DL FAILED — {e}', flush=True)
        dl_failed.append(tic)

print('\n=== Running pySYD ===', flush=True)
results = []
failed  = []
for tic, (cluster, numax_seed, lower_ex, upper_ex, use_est) in TARGETS.items():
    if tic in dl_failed:
        continue
    if not os.path.exists(os.path.join(DATA_DIR, f'{tic}_LC.txt')):
        print(f'  {tic}: no LC, skip', flush=True)
        continue
    print(f'  {tic} ({cluster}) numax~{numax_seed:.1f}...', end=' ', flush=True)
    try:
        res = run_one(tic, cluster, numax_seed, lower_ex, upper_ex, use_est)
        results.append(res)
        print(f'νmax={res["numax"]:.2f}  Δν={res["dnu"]:.3f}  SNR={res["snr"]:.1f}')
    except Exception as e:
        print(f'FAILED — {e}')
        traceback.print_exc()
        failed.append((tic, str(e)))

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f'\nDone: {len(results)}/{len(TARGETS)} succeeded')
if failed:
    print('Failed:')
    for tic, err in failed:
        print(f'  {tic}: {err}')
print(f'\nPlots in: {RESULTS_DIR}/<TIC>/global_fit.png')
print(f'Results:  {OUT_CSV}')
