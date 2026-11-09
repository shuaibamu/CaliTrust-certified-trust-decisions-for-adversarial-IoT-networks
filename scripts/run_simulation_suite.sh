#!/usr/bin/env bash
# Simulation experiments of the original study (Tables 6, 7, 9; Figs. 4-15).
# Writes PGFPlots data files (.dat/.json) consumed by figures/pgfplots/figs.tex.
set -euo pipefail
export CALITRUST_OUT=figures/pgfplots/data
python -m calitrust.experiments main valid attacks splits budget \
       temporal_sleep temporal_cam temporal_oosb ablation rho scal
python figures/pgfplots/make_figures_tex.py
