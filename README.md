# CaliTrust: certified trust decisions for adversarial IoT networks

Reference implementation of **CaliTrust**, the class-conditional conformal trust-certification method described in

> M. Shuaib, I. Aqeel, M. Y. Aalsalem, S. Alam. *Calibrated Distrust: Certified Trust Decisions for Adversarial IoT
> Networks under Post-Calibration Evidence Injection.* Journal of King Saud University – Computer and Information
> Sciences (under review).

CaliTrust turns any trust-score model into a three-way decision for each device: **trust**, **isolate** or
**attest** (send to a verification oracle). Under exchangeable calibration anchors, the probability of trusting a
malicious device is bounded by a user-chosen budget α_M. The pipeline consists of the following parts:

* an **evidence admission control** step: raters with fewer than *m* ratings are ignored, and feature
  standardisation statistics are frozen at calibration time;
* 22 label-free evidence features over a typed graph with six relations (OOR, C-LOR, C-WOR, SOR, INT+, INT−);
* a score model, which can be any encoder. The default is a relation-aware GNN (RA-GNN); GCN and MLP are included;
* **Mondrian (class-conditional) smoothed conformal prediction**, with an optional PAC mode (δ = 0.1);
* **audit-driven per-class adaptive conformal inference (ACI)**, which recalibrates online from uniform audits.

Everything is implemented from scratch with NumPy/SciPy and JAX (automatic differentiation only); scikit-learn is
used for metrics. The code runs on one CPU core.

## Installation

```bash
git clone <this repository>
cd CaliTrust
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                        # SNAP Bitcoin-OTC / Bitcoin-Alpha -> data/
```

The pinned versions are those used for the reported results (Python 3.11, JAX 0.10.2). Encoder-free baselines
(BRS, subjective logic, EigenTrust) give identical numbers across JAX releases. Encoder-based numbers can differ in
the third decimal place under other JAX/XLA versions.

## Repository layout

| path | content |
|---|---|
| `calitrust/sim.py` | Social-IoT simulator: OOR/C-LOR/C-WOR/SOR relations; BS/BM/BST/OO/SB/MIX attacks; CAM and SLEEP drift |
| `calitrust/features.py` | 22 label-free evidence features (P/R/T/S groups) and the six-relation typed graph; frozen-statistics standardisation |
| `calitrust/model.py` | RA-GNN (node-level relation attention), GCN (Guardian-style) and MLP encoders |
| `calitrust/conformal.py` | Mondrian smoothed conformal prediction, PAC quantile, split CP, DAPS / CF-GNN-style diffusion, online ACI |
| `calitrust/baselines.py` | BRS, EigenTrust (sparse), subjective logic |
| `calitrust/experiments.py` | simulation experiments of the original study (Tables 6, 7, 9; Figs. 4–15) |
| `calitrust/semisyn.py` | semi-synthetic protocol on real trust networks (attackers injected into SNAP graphs) |
| `calitrust/robust.py`, `calitrust/seqcert.py` | post-calibration injection and sequential certification modules of the original submission |
| `calitrust/realdata.py` | real-label (REV2) experiment of the original submission; superseded and not used in the revised paper |
| `calitrust/revcommon.py` | helpers for the revision: consistent-target comparison, low-cost rules, anchor-bias pools, audit replay, outcome decomposition |
| `calitrust/rev_sim.py` | revision experiments on the simulator (parts `ab` and `c`) |
| `calitrust/rev_real.py` | revision experiments on Bitcoin-OTC / Bitcoin-Alpha (baselines, marginal CP, low-cost rules, attestation load) |
| `calitrust/rev_robust.py` | post-calibration evidence injection: refit vs frozen standardisation, insiders, established sybils, *m* ∈ {3, 5, 10}, ACI after injection |
| `scripts/` | data download, table generation, figure scripts, one-shot run scripts |
| `results/` | JSON outputs of the revision runs reported in the paper, and the generated tables (`tables_rev.json`) |
| `figures/png/` | the 16 figures of the revised manuscript (`fig01.png` … `fig16.png`) |
| `figures/tikz/` | TikZ sources of Fig. 1 (SIoT structure) and Fig. 2 (architecture) |
| `figures/pgfplots/` | PGFPlots source of Figs. 4–15 (`figs.tex`) and the data files it reads |

## Reproducing the paper

All commands are run from the repository root. Results are written to `results/` (override with `CALITRUST_OUT`);
data are read from `data/` (override with `CALITRUST_DATA`).

| manuscript item | command | output |
|---|---|---|
| Fig. 1, Fig. 2 | `python scripts/fig1_fig2_render.py` (raster) or compile `figures/tikz/*.tex` (vector) | `fig1_siot_structure.png`, `fig2_architecture.png` |
| Fig. 3 | `python scripts/fig3_ragnn_mechanism.py figures/png/fig03.png` | PNG |
| Tables 6, 7, 9; Figs. 4–15 | `bash scripts/run_simulation_suite.sh`, then `pdflatex figures_main.tex` in `figures/pgfplots/` | `.dat`/`.json` in `figures/pgfplots/data/` |
| Table 8 (anchor-selection bias) | `python -m calitrust.rev_sim ab`; `python -m calitrust.rev_real {otc,alpha} 5` | `results/rev_sim_ab.json`, `results/rev_real_*.json` |
| Table 10 (attack-agnostic features, leave-one-family-out) | `python -m calitrust.rev_sim ab` | `results/rev_sim_ab.json` |
| Table 11 (semi-synthetic validation on real networks) | `python -m calitrust.rev_real {otc,alpha} 5` | `results/rev_real_*.json` |
| Table 12 (low-cost decision rules) | `rev_sim ab` and `rev_real` (as above) | as above |
| Table 13 (insiders, established sybils, admission threshold *m*) | `python -m calitrust.rev_robust {otc,alpha}` | `results/rev_robust_*.json` |
| Table 14 (audit-driven recalibration after injection) | `python -m calitrust.rev_robust {otc,alpha}` | `results/rev_robust_*.json` |
| Fig. 16 (post-calibration sybil injection) | `python scripts/fig16_sybil_injection.py figures/png/fig16.png` | PNG |
| Sleeper-drift diagnostics (Theorem 2 bound, α_M,t trajectories) | `python -m calitrust.rev_sim c` | `results/rev_sim_c.json` |
| Markdown versions of Tables 8 and 10–14 | `python scripts/make_tables.py results/tables_rev.json` | `results/tables_rev.json` |

`scripts/run_revision.sh` (or `scripts/run_revision.ps1` on Windows) runs the whole revision pipeline in order.
`python scripts/make_tables.py` on the shipped `results/*.json` reproduces `results/tables_rev.json` exactly.
On one CPU core, one `rev_robust` network (3 attack draws × all injection configurations) takes about 8 minutes.

### Protocol notes

* **Attackers on the real networks.** 10% of all users (588 on Bitcoin-OTC, 378 on Bitcoin-Alpha) are drawn from
  users with at least 3 ratings and assigned one of the five attack families. The ratings they issue and
  receive are modified according to that family. Honest users keep their real ratings, and both networks keep
  their SNAP timestamps. Genuine fraudsters in the raw data are labelled honest; this label noise affects every
  method equally.
* **Standardisation protocols.** `refit` recomputes the feature statistics after an injection, as in the original
  submission. `frozen` keeps the statistics from calibration time, which is the default for CaliTrust in the
  revised paper. Both are reported.
* **Seeds.** Simulation results use 10 networks (seeds 0–9). Leave-one-family-out runs use 5 seeds. Real-network
  baselines use 5 attack draws and injection experiments use 3.

## Citation

See `CITATION.cff`. Please cite the article above if you use this code.

## License

MIT (see `LICENSE`). The SNAP data sets are distributed by Stanford SNAP under their own terms and are not included.
