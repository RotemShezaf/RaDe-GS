# TOSCA Benchmark

Hyperparameter search and production sweep infrastructure for RaDe-GS on TOSCA shapes.

## Directory layout

All scripts live under `scripts/tosca/benchmark/`.
Training outputs land inside each shape's source directory, e.g.

```
TrainData/TOSCA/SyntheticColmapData/blue_texture/{shape}/high_res/light_0/
  ├── r4_a01/          # round-4 benchmark run
  ├── sweep_sw01/      # production sweep run
  ├── best_output/     # selected best from sweep
  └── ...
```

## Scripts

### Hyperparameter search (single shape)

| Script | Purpose |
|--------|---------|
| `benchmark_params.sh` | Round-1/2 parameter sweeps (1-D axis scans) |
| `benchmark_round3.sh` | Round-3 combo sweeps |
| `benchmark_round4.sh` | Round-4 extended sweeps (60 configs, blocks A–F) |
| `benchmark_tmux.sh` | Tmux launcher for the round scripts |
| `analyze_all_rounds.py` | Load all `benchmark_report.json` files, rank by composite score, print per-param analysis |

### Production sweep (all shapes)

| Script | Purpose |
|--------|---------|
| `benchmark_sweep.sh` | Run ~10 configurations on **all 17 TOSCA shapes** in parallel. Uses `tosca_animal_map.sh` for shape expansion. |
| `benchmark_sweep_tmux.sh` | Tmux launcher for `benchmark_sweep.sh` (default: gipdeep10, 2 GPUs) |
| `choose_sweep_results.py` | Select the best sweep result for a **single shape** using a composite score (PSNR, Chamfer, MSD, g2s\_max penalty). Copies winner to `best_output/`. |
| `choose_sweep_results_all.py` | Iterate all shapes and call `choose_sweep_results.py` logic on each. Writes `sweep_selection_summary.json`. |

### Evaluation & reporting

| Script | Purpose |
|--------|---------|
| `evaluate_benchmark.py` | Evaluate a single training output (mesh extraction + metrics). Writes `benchmark_report.json`. |
| `generate_report_pdf.py` | Generate a multi-page PDF report with tables sorted by Chamfer/PSNR/g2s and per-param scatter plots. |
| `generate_report_png.py` | Same tables/plots as the PDF version, saved as individual PNG files. |
| `generate_report_png_all.py` | Run PNG report generation across all shapes. Outputs to `<synth_data_base>/sweep_reports/`. |

## Typical workflow

```bash
# 1. Run the production sweep on all shapes (inside tmux)
bash scripts/tosca/benchmark/benchmark_sweep_tmux.sh

# 2. Generate PNG reports for every shape
python scripts/tosca/benchmark/generate_report_png_all.py

# 3. Select the best config per shape and copy to best_output/
python scripts/tosca/benchmark/choose_sweep_results_all.py
```

## Composite scoring formula

Lower is better:

```
score = psnr_err + chamfer_err + msd_err + 0.3 * g2s_max_penalty

psnr_err    = max(0, 48.5 - psnr) / 48.5
chamfer_err = max(0, chamfer - 0.101) / 0.101
msd_err     = max(0, msd - 0.2) / 0.2
g2s_max_pen = max(0, g2s_max - 5.0) / 5.0
```

## Shape list (17 shapes)

cat0, cat2, centaur0, centaur1, centaur5, david0, dog0, gorilla5, gorilla8, horse0, horse10, michael0, michael2, michael16, victoria0, victoria2, wolf0
