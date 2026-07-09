# Mankowski et al. (2025) — Open Cluster Asteroseismology with TESS

Analysis code and data for "Expanding Asteroseismic Studies in Star Clusters Using NASA's TESS and ESA's Gaia Missions"

## Structure

### `notebooks/`
1. `01_echelle_dnu.ipynb` — Interactive echelle diagrams for measuring large frequency separation with comb overlay and slider
2. `02_numax_slider.ipynb` — Interactive numax slider for verifying oscillation bump positions using SPOC/QLP power spectra
3. `03_pysyd.ipynb` — PySYD seismic analysis pipeline for numax extraction with Harvey model backgrounds
4. `04_seismic_params.ipynb` — Compute mass, radius, log g from numax + dnu + Teff using scaling relations with APOKASC 3 corrections and Sharma+2016 f_dnu
5. `05_ages.ipynb` — Stellar age estimation using MIST, Dartmouth, GARSTEC, and YREC grids following Pinsonneault et al. (2025) multi model strategy
6. `06_figures.ipynb` — Generate all paper figures: dnu vs numax, log g comparison, radius, mass, age plots
7. `07_isochrones.ipynb` — MIST isochrone CMD plots for NGC 752, Theia 6046, Casado Alessi 1, and Theia 844

### `data/`
- `seismic_results.csv` — Final asteroseismic parameters for all 23 targets (13 confirmed + 10 tentative detections)
- `age_results.csv` — Per star and cluster level age estimates from 4 grids
- `measured_dnu_values.csv` — Hand measured dnu values from echelle diagram analysis
- `target_list.csv` — Initial target list with cluster assignments
- `full_catalog.csv` — Complete input catalog (38,311 stars) with Gaia DR3 photometry

### `scripts/`
- `run_pysyd_batch.py` — Batch PySYD execution for all targets
- `run_pysyd_plots.py` — Diagnostic plot generation for PySYD results
- `diagnostic_panels.py` — Per-target diagnostic panels (power spectrum with two-component Harvey background, background-corrected SNR, echelle) used in Appendix C
- `validate_pipeline.py` — End-to-end pipeline validation on the known Kepler oscillator KIC 10323222 (recovers numax to ~1%)
- `isochrones.py` — MIST isochrone CMD plot generation

## Revision notes
- Backgrounds are fit with a two-component Harvey profile plus white noise (Kallinger et al. 2014 form), fit in log space over the full spectrum with the oscillation region masked.
- TESS-SPOC 2-minute data are used in preference to QLP where available (6 of 23 targets).
- Detections are reported as 13 confirmed + 10 tentative rather than treated equally.
- Surface-gravity validation uses the Andrae et al. (2023) machine-learning log g as the primary check, with GSP-Spec as a secondary comparison.

## Clusters with Multiple Detections
| Cluster | N | Seismic Age (Gyr) | Literature Age (Gyr) |
|---------|---|-------------------|---------------------|
| Theia 6046 | 7 | 4.52 +/- 1.81 | 0.85 (KC20) |
| NGC 752 | 4 | 1.55 +/- 0.40 | 1.30 to 1.60 |
| Casado Alessi 1 | 3 | 1.01 +/- 0.27 | 0.69 to 1.45 |
| Theia 844 | 2 | 0.26 +/- 0.37 | 0.32 to 0.44 |

## Dependencies
- Python 3.9+
- pySYD, echelle, isochrones, kiauhoku
- numpy, scipy, pandas, matplotlib, ipywidgets
- astropy, lightkurve
