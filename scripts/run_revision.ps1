# Windows PowerShell equivalent of run_revision.sh (run from the repository root)
$ErrorActionPreference = "Stop"
python scripts/download_data.py
python -m calitrust.rev_sim ab
python -m calitrust.rev_sim c
python -m calitrust.rev_real otc 5
python -m calitrust.rev_real alpha 5
python -m calitrust.rev_robust otc
python -m calitrust.rev_robust alpha
python scripts/make_tables.py results/tables_rev.json
python scripts/fig16_sybil_injection.py figures/png/fig16.png
