# Continuous-Surrogate HPO Study

Extends the discrete-grid DPO-AO hyperparameter search (lr, beta swept over a
fixed grid) into a continuous search space, by fitting a regression surrogate
over the grid data and letting schedulers propose any (lr, beta) in the
observed range instead of only exact grid points.

## Files

| File | What it does |
|---|---|
| `build_blackbox_surrogate.py` | Fits a surrogate model (`--surrogate knn1\|knn3\|knn5\|gp`) over log10(lr)/log10(beta) from the cleaned grid data, wraps it as a syne-tune `BlackboxSurrogate`, and pickles it. |
| `run_simulations_surrogate.py` | Runs the simulated HPO sweep (RandomSearch / TPE / CQR / ASHA / BOHB × search objectives × 25 seeds) against that surrogate. Parallelized across CPUs. |
| `analyze_results_surrogate.py` | Produces regret curves (best-value-found-so-far vs. time/evals) for one model size, one panel per search objective. |
| `plot_regret_curves_by_size.py` | Produces regret curves for one surrogate, one panel per model size instead of per objective (y-axis: Z-All or Z-Dynamic). |
| `run_surrogate.sbatch` | SLURM job: runs the three steps above for one (size, surrogate) pair. |

**Shared dependencies** (used by the study above, not surrogate-specific):
`common.py` (constants, paths, optimizer list), `schedulers.py` (builds each
optimizer's syne-tune scheduler), `build_blackbox.py` (grid-cleaning helpers
reused by `build_blackbox_surrogate.py`), `analyze_results.py` (the
`plot_regret_curves` plotting function, reused by the two analyze/plot
scripts above).

Budget regime: generous only (workers get enough simulated wallclock budget
to complete several full-fidelity trials each).

## Running it

Requires a per-size grid CSV already collected (`<size_dir>/data/*_grid.csv`)
and a Python environment with `syne-tune`, `pandas`, `numpy`, `scikit-learn`,
`matplotlib` installed.

```bash
# 1. Build the surrogate blackbox for one (size, surrogate) pair
python build_blackbox_surrogate.py --size 4b --surrogate gp

# 2. Run the simulated HPO sweep against it
python run_simulations_surrogate.py --size 4b --surrogate gp --n-procs 16

# 3. Plot regret curves for that one size
python analyze_results_surrogate.py --size 4b --surrogate gp

# 4. Plot regret curves across all sizes for that surrogate (repeat step 1-2
#    per size first)
python plot_regret_curves_by_size.py --surrogate gp
```

`--surrogate` accepts `knn1`, `knn3`, `knn5`, or `gp` (`gp` is the
recommended default — see `build_blackbox_surrogate.py`'s module docstring
for why `knn1` in particular under-serves model-based optimizers). `--family`
defaults to `qwen3`; `llama` is also supported if that data is present.

On a SLURM cluster, submit the whole per-size pipeline as one job:

```bash
sbatch --job-name=hpo_bb_surrogate_4b_gp --export=ALL,SIZE=4b,SURROGATE=gp run_surrogate.sbatch
```
