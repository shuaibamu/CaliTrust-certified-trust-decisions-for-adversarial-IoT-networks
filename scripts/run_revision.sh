#!/usr/bin/env bash
# Re-runs every revision experiment and regenerates Tables 8 and 10-14 and Fig. 16.
# Outputs go to results/ (override with CALITRUST_OUT). Run from the repository root.
set -euo pipefail
python scripts/download_data.py
python -m calitrust.rev_sim ab          # simulation: Tables 8, 10, 12 (simulation columns)
python -m calitrust.rev_sim c           # simulation: sleeper-drift diagnostics (Section 7, Remark 2)
python -m calitrust.rev_real otc 5      # real networks: Tables 8, 11, 12
python -m calitrust.rev_real alpha 5
python -m calitrust.rev_robust otc      # real networks: Tables 13, 14, Fig. 16
python -m calitrust.rev_robust alpha
python scripts/make_tables.py results/tables_rev.json
python scripts/fig16_sybil_injection.py figures/png/fig16.png
