"""
Batch pySYD run for all 52 cluster-star candidates.
Skips already-downloaded LCs and already-run pySYD results.
Outputs: pysyd_results.csv, consistency_check.csv
"""
import os, sys, warnings, traceback
import numpy as np
import pandas as pd
import lightkurve as lk
from pysyd import utils
from pysyd.target import Target

warnings.filterwarnings('ignore')

WORKDIR    = '/Users/carlimankowski/research/revision'
DATA_DIR   = os.path.join(WORKDIR, 'data')
INFO_DIR   = os.path.join(WORKDIR, 'info')
RESULTS_DIR= os.path.join(WORKDIR, 'results')
AVAIL_CSV  = os.path.join(WORKDIR, 'spoc_availability.csv')
OUT_CSV    = os.path.join(WORKDIR, 'pysyd_results.csv')
CONS_CSV   = os.path.join(WORKDIR, 'consistency_check.csv')
for d in [DATA_DIR, INFO_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

NaN = float('nan')

# ── Target list ───────────────────────────────────────────────────────────────
TARGET_DATA = [
    (352774198,  'COIN-Gaia_30',         NaN,   NaN,  NaN,   5,  30),
    (306552813,  'Casado_Alessi_1',   66.163,   NaN,  NaN,  45, 100),
    (306955660,  'Casado_Alessi_1',   62.896,   NaN,  NaN,  40,  90),
    (306187638,  'Casado_Alessi_1',   68.064,   NaN,  NaN,  45, 100),
    (395409799,  'Collinder_463',        NaN,   NaN,  NaN, 100, 400),
    (427648004,  'Collinder_463',        NaN,   NaN,  NaN,   3,  25),
    (56551765,   'HSC_1318',             NaN,   NaN,  NaN,   5, 300),
    (140143506,  'HSC_1986',             NaN,   NaN,  NaN,  80, 300),
    (402808611,  'HSC_2615',             NaN,   NaN,  NaN,   5,  40),
    (98689079,   'HSC_2919',             NaN,   NaN,  NaN,   5, 300),
    (346930285,  'HSC_924',              NaN,   NaN,  NaN,   3,  25),
    (414752508,  'HSC_95',               NaN,   NaN,  NaN,  50, 150),
    (101168806,  'IC_348',               NaN,   NaN,  NaN,   5, 300),
    (354016987,  'LISC_3534',            NaN,   NaN,  NaN,  10,  55),
    (16482780,   'NGC_2281',             NaN,   NaN,  NaN,   3,  25),
    (382510863,  'NGC_2516',             NaN,   NaN,  NaN,   5,  30),
    (195184975,  'NGC_2632',             NaN,   NaN,  NaN,   5, 200),
    (437030530,  'NGC_2682',             NaN,   NaN,  NaN,   5,  40),
    (46305204,   'NGC_2682',             NaN,   NaN,  NaN,   5,  30),
    (459055638,  'NGC_2682',             NaN,   NaN,  NaN,   2,  20),
    (67569102,   'NGC_752',           67.760,   NaN,  NaN,  45, 100),
    (67420118,   'NGC_752',           68.416,   NaN, 90.0,  65, 120),
    (186970424,  'NGC_752',           50.083,   NaN, 90.0,  65, 120),
    (67419922,   'NGC_752',           33.348,   NaN,  NaN,  15,  55),
    (321894003,  'Renou_23',             NaN,   NaN,  NaN,   5, 200),
    (150753375,  'Theia_1188',           NaN,   NaN,  NaN,   5,  45),
    (182395456,  'Theia_12',             NaN,   NaN,  NaN,   5, 200),
    (321823630,  'Theia_1297',           NaN,   NaN,  NaN,  30, 100),
    (143093935,  'Theia_169',            NaN,   NaN,  NaN,   5, 200),
    (387813185,  'Theia_5',              NaN,   NaN,  NaN,   5, 200),
    (306345133,  'Theia_6046',        22.216,  34.6,  NaN,  15,  55),
    (43902016,   'Theia_6046',        15.071,  15.8,  NaN,   5,  30),
    (24297458,   'Theia_6046',        56.009,  32.1,  NaN,  15,  55),
    (24666306,   'Theia_6046',        33.294,  35.3,  NaN,  20,  55),
    (24444542,   'Theia_6046',        16.582,   NaN,  9.0,   3,  20),
    (249064439,  'Theia_6046',        62.854,   NaN,  NaN,  40,  90),
    (34472483,   'Theia_6046',        34.764,   NaN,  NaN,  20,  55),
    (24661083,   'Theia_6046',           NaN,   NaN,  NaN,   3,  20),
    (24725114,   'Theia_6046',           NaN,   NaN,  NaN,   2,  15),
    (43901963,   'Theia_6046',           NaN,   NaN,  NaN,   3,  20),
    (24662412,   'Theia_6046',           NaN,   NaN,  NaN,   5,  30),
    (195860623,  'Theia_844',            NaN,   NaN,  NaN,  50, 130),
    (195862014,  'Theia_844',            NaN,   NaN,  NaN,  35, 100),
    (353015801,  'Theia_850',            NaN,   NaN,  NaN, 100, 500),
    (297266987,  'UPK_127',              NaN,   NaN,  NaN, 200, 600),
    (335695889,  'UPK_189',              NaN,   NaN,  NaN,   5, 300),
    (279706340,  'UPK_220',              NaN,   NaN,  NaN,   3,  20),
    (427725221,  'UPK_226',              NaN,   NaN,  NaN,   5,  30),
    (20869256,   'UPK_296',              NaN,   NaN,  NaN,   5,  35),
    (306552754,  'Unknown_1',            NaN,   NaN,  NaN,   5, 300),
    (459055617,  'Unknown_3',            NaN,   NaN,  NaN,   5, 200),
    (332983572,  'XDOCC_1',              NaN,   NaN,  NaN,   3,  20),
]
COLS = ['tic','cluster','numax_old','numax_hon','numax_reviewer','lower_ex','upper_ex']
targets = pd.DataFrame(TARGET_DATA, columns=COLS)
targets['tic'] = targets['tic'].astype(int)
STAR = {r.tic: r._asdict() for r in targets.itertuples(index=False)}

TEFF_PRIOR = {
    352774198:4601, 306552813:5035, 306955660:4920, 306187638:4500,
    395409799:8743, 427648004:4476, 56551765:7565,  140143506:5424,
    402808611:4748, 98689079:float('nan'), 346930285:4461, 414752508:4983,
    101168806:float('nan'), 354016987:4362, 16482780:4324, 382510863:4536,
    195184975:float('nan'), 437030530:4267, 46305204:4095, 459055638:3792,
    321894003:float('nan'), 150753375:4379, 182395456:float('nan'), 321823630:4893,
    143093935:float('nan'), 387813185:float('nan'),
    24661083:3973, 24725114:3759, 34472483:4922, 43901963:4023,
    306345133:4703, 24297458:4711, 43902016:4499, 24444542:4288,
    24662412:4296, 24666306:4294, 249064439:5056,
    195860623:4960, 195862014:4900, 353015801:8676,
    297266987:7990, 335695889:float('nan'), 279706340:4341, 427725221:4485,
    20869256:4559, 306552754:float('nan'), 459055617:4383, 332983572:4462,
    186970424:4859, 67569102:4862, 67419922:4825, 67420118:4956,
}

# ── Step 1: SPOC availability (extend cache for new stars) ────────────────────
if os.path.exists(AVAIL_CSV):
    avail = pd.read_csv(AVAIL_CSV)
    avail['tic'] = avail['tic'].astype(int)
    cached_tics = set(avail['tic'].tolist())
else:
    avail = pd.DataFrame()
    cached_tics = set()

new_rows = []
for tic in targets['tic']:
    if tic in cached_tics:
        continue
    print(f'  SPOC check TIC {tic}...', flush=True)
    row = {'tic': tic, 'n_spoc_120s': 0, 'spoc_sectors': [],
           'n_qlp': 0, 'qlp_sectors': [], 'data_source': 'none'}
    try:
        sr = lk.search_lightcurve(f'TIC {tic}', author='SPOC', exptime=120)
        if len(sr) > 0:
            row['n_spoc_120s'] = len(sr)
            row['data_source'] = 'SPOC'
    except: pass
    try:
        sr2 = lk.search_lightcurve(f'TIC {tic}', author='QLP')
        if len(sr2) > 0:
            row['n_qlp'] = len(sr2)
            if row['data_source'] == 'none':
                row['data_source'] = 'QLP'
    except: pass
    new_rows.append(row)

if new_rows:
    avail = pd.concat([avail, pd.DataFrame(new_rows)], ignore_index=True)
    avail.to_csv(AVAIL_CSV, index=False)
    print(f'Updated availability cache with {len(new_rows)} new stars.')

# ── Step 2: Download LCs (skip if file exists) ────────────────────────────────
download_log = {}

def lc_download(tic, overwrite=False, window=299, sigma=4.0):
    lc_path = os.path.join(DATA_DIR, f'{tic}_LC.txt')
    if os.path.exists(lc_path) and not overwrite:
        download_log[tic] = 'cached'
        return 'cached'
    sr_spoc = lk.search_lightcurve(f'TIC {tic}', author='SPOC', exptime=120)
    if len(sr_spoc) > 0:
        sr, used = sr_spoc, 'SPOC'
    else:
        sr = lk.search_lightcurve(f'TIC {tic}', author='QLP')
        used = 'QLP'
    if len(sr) == 0:
        raise ValueError(f'No LC found')
    lc_list = []
    for lc_raw in sr.download_all():
        lc_list.append(lc_raw.normalize().flatten(window_length=window).remove_outliers(sigma=sigma))
    lc = lk.LightCurveCollection(lc_list).stitch().normalize()
    times  = lc['time'].value
    fluxes = lc['flux'].value
    idx = np.argsort(times)
    times, fluxes = times[idx], fluxes[idx]
    for i in range(1, len(times)):
        if times[i] - times[i-1] > 10:
            times[i:] -= (times[i] - times[i-1])
    with open(lc_path, 'w') as f:
        for t, fl in zip(times, fluxes):
            if np.isfinite(t) and np.isfinite(fl):
                f.write(f'\t{t}\t{fl}\n')
    ps_path = os.path.join(DATA_DIR, f'{tic}_PS.txt')
    if os.path.exists(ps_path):
        os.remove(ps_path)
    download_log[tic] = used
    return used

print('\n=== Downloading LCs ===')
dl_failed = []
for tic in targets['tic']:
    try:
        used = lc_download(tic)
        print(f'  TIC {tic}: {used}', flush=True)
    except Exception as e:
        print(f'  TIC {tic}: FAILED — {e}', flush=True)
        dl_failed.append(tic)

# ── Step 3: Run pySYD ─────────────────────────────────────────────────────────
# Load existing results if any
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    prev['tic'] = prev['tic'].astype(int)
    pysyd_results = {r.tic: r._asdict() for r in prev.itertuples(index=False)}
    print(f'\nLoaded {len(pysyd_results)} existing pySYD results from cache.')
else:
    pysyd_results = {}

def run_pysyd(tic):
    sp = STAR[tic]
    tic_str = str(tic)
    info_csv = os.path.join(INFO_DIR, 'star_info.csv')
    with open(info_csv, 'w') as f:
        f.write('star,seed\n')
    params = utils.Parameters()
    params.add_targets(stars=tic_str)
    star = Target(tic_str, params)
    prior = (sp['numax_reviewer'] if np.isfinite(sp['numax_reviewer'])
             else sp['numax_hon'] if np.isfinite(sp['numax_hon'])
             else sp['numax_old'] if np.isfinite(sp['numax_old'])
             else 50.0)
    star.params.update({
        'verbose': False, 'mc_iter': 50, 'smooth_ps': 0,
        'show': False, 'overwrite': True, 'fix_wn': False,
        'inpdir': DATA_DIR, 'infdir': INFO_DIR, 'outdir': RESULTS_DIR,
        'lower_ex': sp['lower_ex'], 'upper_ex': sp['upper_ex'],
        'lower_bg': 5, 'upper_bg': 525, 'numax': prior,
    })
    star.load_data()
    star.process_star()
    numax_out = (star.params.get('numax_smorhcv')
                 or star.params.get('numax_smoo')
                 or star.params.get('numax'))
    dnu_out = star.params.get('obs_dnu') or star.params.get('dnu')
    snr_out = star.params.get('snr', float('nan'))
    return {
        'tic': tic, 'cluster': STAR[tic]['cluster'],
        'numax_new': float(numax_out) if numax_out else float('nan'),
        'dnu_new':   float(dnu_out)   if dnu_out   else float('nan'),
        'snr':       float(snr_out)   if snr_out   else float('nan'),
    }

print('\n=== Running pySYD ===')
py_failed = []
for tic in targets['tic']:
    if tic in dl_failed:
        continue
    lc_path = os.path.join(DATA_DIR, f'{tic}_LC.txt')
    if not os.path.exists(lc_path):
        continue
    print(f'  pySYD TIC {tic} ({STAR[tic]["cluster"]})...', end=' ', flush=True)
    try:
        res = run_pysyd(tic)
        pysyd_results[tic] = res
        print(f'numax={res["numax_new"]:.2f}  dnu={res["dnu_new"]:.3f}  snr={res["snr"]:.1f}')
    except Exception as e:
        print(f'FAILED — {e}')
        py_failed.append(tic)

# Save all results
out_rows = list(pysyd_results.values())
pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)
print(f'\nSaved {len(out_rows)} results to {OUT_CSV}')

# ── Step 4: Cluster consistency check ─────────────────────────────────────────
NUMAX_SUN, DNU_SUN, TEFF_SUN = 3090.0, 135.1, 5772.0

def seis_mass(numax, dnu, teff):
    if not all(np.isfinite([numax, dnu, teff])): return float('nan')
    return (numax/NUMAX_SUN)**3 * (dnu/DNU_SUN)**(-4) * (teff/TEFF_SUN)**1.5

def seis_radius(numax, dnu, teff):
    if not all(np.isfinite([numax, dnu, teff])): return float('nan')
    return (numax/NUMAX_SUN) * (dnu/DNU_SUN)**(-2) * (teff/TEFF_SUN)**0.5

seis_rows = []
for _, trow in targets.iterrows():
    tic   = trow['tic']
    res   = pysyd_results.get(tic, {})
    numax = float(res.get('numax_new', float('nan')))
    dnu   = float(res.get('dnu_new',   float('nan')))
    snr   = float(res.get('snr',       float('nan')))
    teff  = float(TEFF_PRIOR.get(tic, float('nan')))
    if not np.isfinite(teff): teff = 4500.0
    M = seis_mass(numax, dnu, teff)
    R = seis_radius(numax, dnu, teff)
    seis_rows.append({'TIC': tic, 'cluster': trow['cluster'],
                      'numax': numax, 'dnu': dnu, 'snr': snr,
                      'teff': teff, 'M_seis': M, 'R_seis': R})

seis = pd.DataFrame(seis_rows)

print('\n=== Cluster consistency (mass deviation from cluster median) ===\n')
flagged = []
cons_rows = []
for cluster, grp in seis.groupby('cluster'):
    det = grp[np.isfinite(grp['M_seis'])]
    med_M = det['M_seis'].median() if len(det) >= 2 else float('nan')
    for _, row in grp.iterrows():
        dev = abs(row['M_seis'] - med_M) / med_M if np.isfinite(med_M) and np.isfinite(row['M_seis']) else float('nan')
        flag = (np.isfinite(dev) and dev > 0.20)
        status = 'FLAG' if flag else 'ok  '
        if flag:
            flagged.append(int(row['TIC']))
        dev_str = f'{dev*100:.0f}%' if np.isfinite(dev) else '—'
        print(f"  {status}  {cluster:<22}  TIC {int(row['TIC']):<10}"
              f"  νmax={row['numax']:7.2f}  Δν={row['dnu']:7.3f}"
              f"  M={row['M_seis']:.2f}  R={row['R_seis']:.2f}"
              f"  dev={dev_str}")
        cons_rows.append({**row, 'med_M_cluster': med_M,
                          'mass_dev_pct': dev*100 if np.isfinite(dev) else float('nan'),
                          'flagged': flag})

pd.DataFrame(cons_rows).to_csv(CONS_CSV, index=False)
print(f'\nFlagged TICs (>20% mass deviation): {flagged}')
print(f'Failed downloads: {dl_failed}')
print(f'Failed pySYD:     {py_failed}')
print(f'\nResults: {OUT_CSV}')
print(f'Consistency: {CONS_CSV}')
