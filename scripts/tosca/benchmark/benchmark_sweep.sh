#!/bin/bash
#
# Benchmark Sweep: Train + Mesh Extract + Evaluate for all TOSCA shapes
# using a small set of hyperparameter configurations.
#
# Modelled after train_tosca_smart.sh but with sweep configs and unique
# wandb sweep names/IDs per shape.
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_sweep.sh [options]
#
# OPTIONS:
#   --shapes LIST              Comma-separated shape names (default: all)
#   --animals LIST             Animal names WITHOUT index; expands to all indexed shapes
#   --textures LIST            Comma-separated texture names (default: blue)
#   --colmap_resolutions LIST  Comma-separated resolution levels (default: high_res)
#   --processed_base DIR       Processed TOSCA root (default: TrainData/TOSCA/processed)
#   --synth_data_base DIR      Synthetic COLMAP data base (default: TrainData/TOSCA/SyntheticColmapData)
#   --gt_data_root DIR         GT mesh root (default: TrainData/TOSCA/processed)
#   --iterations N             Training iterations (default: 45000)
#   --max_parallel N           Maximum parallel GPU jobs (default: 2)
#   --max_parallel_cpu N       Maximum parallel CPU eval jobs (default: 4)
#   --light_id ID              Light ID for source path (default: 0)
#   --wandb_project NAME       Wandb project name (default: tosca-sweep)
#   --no_wandb                 Disable wandb logging
#   --skip_existing            Skip runs that already have benchmark_report.json
#   --evaluate_only            Skip GPU training/mesh; only run CPU evaluation on existing runs
#   --dry_run                  Print commands without executing
#
# EXAMPLES:
#   bash scripts/tosca/benchmark/benchmark_sweep.sh --dry_run
#   bash scripts/tosca/benchmark/benchmark_sweep.sh --shapes cat0,dog0
#   bash scripts/tosca/benchmark/benchmark_sweep.sh --animals cat --max_parallel 1

set -u

# Load animal→index map and expand_animals() helper
source "$(dirname "${BASH_SOURCE[0]}")/../tosca_animal_map.sh"

# ============================================================================
# ============================================================================
# R10 Per-Shape Sweep Configurations
# ============================================================================
#
# Data: R7 (sw1-sw8) + R8 (per-shape a-d) for 9 original shapes.
#       R9 was never executed (evaluate-only ran, no training).
#       8 new shapes have NO data at all.
#
# Comprehensive analysis of all available runs (166 for cat0, 10-12 for others):
#
# Universal findings:
#   - sw6 (ldn=0.01, mop=0.35, lmvg=1.0) = best PSNR on almost every shape
#   - sw7 (ldn=0.015, mop=0.35, lmvg=1.5, pd=0.15) = best g2s_max on most shapes
#   - sw8 (ldn=0.02, mop=0.40) = good count reduction with decent quality
#   - ldn=0.01→best PSNR, ldn=0.015+pd=0.15→best g2s_max (40% improvement)
#   - psa CATASTROPHIC on cat2, centaurs, gorilla5 (NEVER use)
#   - psa HELPFUL on david0, dog0 (R8 proved bpsf=0.008+dgt=0.0003+psa works)
#   - bpsf=0.005+dgt=0.0001 = only safe combination for most shapes
#   - Exception: david0/dog0 benefit from bpsf=0.008+dgt=0.0003+psa
#
# R10 design:
#   - 3 targeted configs per shape × 17 shapes = 51 total
#   - Original 9 shapes: data-informed configs targeting PSNR + g2s_max
#   - New 8 shapes: sw6/sw7/sw8 equivalents (proven universal top-3)
#
# Config format: "target_shape  name  bpsf  mop  lmvg  ldist  dgt  ldn  ldssim  psa  pd  pmst"
# All use mvncc=0.3
# ============================================================================
SWEEP_CONFIGS=(

    # ── cat0 (best: sw6=48.81/5.06, sw7=48.72/6.64, 80K GS) ─────────
    # Under 200K. Target: beat sw6's g2s_max=5.06 while maintaining PSNR≥48.5
    "cat0      cat0_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # sw6 + pd=0.10 (mild g2s push, untested midpoint)
    "cat0      cat0_l   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # ldn=0.015 + pd=0.10 (intermediate sw6↔sw7)
    "cat0      cat0_m   0.005  0.35  1.5  0.05  0.0001  0.01   0.1   0   0.15  0"     # sw7 structure but ldn=0.01 for PSNR

    # ── cat2 (best: sw6=44.15/9.82, sw4=40.82/8.95, 105K GS) ────────
    # Under 200K, NO psa! ldn=0.01 is 3+ dB better than ldn=0.04.
    "cat2      cat2_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # sw6 + pd=0.10 (mild g2s improvement)
    "cat2      cat2_l   0.005  0.35  1.5  0.05  0.0001  0.01   0.1   0   0.15  0"     # sw7 style but ldn=0.01 (keep PSNR, push g2s)
    "cat2      cat2_m   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # ldn=0.015 + pd=0.15 for max g2s push

    # ── centaur0 (best: sw6=45.83/13.32, 308K GS) ────────────────────
    # Need ≤200K → mop=0.50-0.55. NO psa. Also improve g2s_max.
    "centaur0  cen0_k   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # mop=0.45 (~245K) + pd=0.10 for g2s
    "centaur0  cen0_l   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.50 (~220K) + quality combo
    "centaur0  cen0_m   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55 (~195K, target count)

    # ── centaur1 (best: sw6=46.17/15.34, sw1=42.83/12.46, 329K GS) ──
    # Same centaur strategy. NO psa.
    "centaur1  cen1_k   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # mop=0.45 + mild pd
    "centaur1  cen1_l   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.50 + quality combo
    "centaur1  cen1_m   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55 (target count)

    # ── centaur5 (best: sw6=46.04/14.36, 334K GS) ────────────────────
    # Same centaur strategy. NO psa.
    "centaur5  cen5_k   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # mop=0.45 + mild pd
    "centaur5  cen5_l   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.50 + quality combo
    "centaur5  cen5_m   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55 (target count)

    # ── david0 (best: dav0_d=47.77/9.26@20K, sw6=47.08/11.92@481K) ──
    # dav0_d proved bpsf=0.010+dgt=0.0003+psa=10 WORKS here (best shape).
    # Try similar approach with safer parameters, plus bpsf=0.005 with count reduction.
    "david0    dav0_k   0.008  0.45  1.0  0.05  0.0002  0.01   0.1   5   0.10  0"     # mid bpsf + safe dgt + psa=5 (dav0_d inspired)
    "david0    dav0_l   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # bpsf=0.005 + count + quality combo
    "david0    dav0_m   0.010  0.50  1.0  0.05  0.0003  0.01   0.1   8   0.15  0"     # close to dav0_d recipe (proven top PSNR)

    # ── dog0 (best: dog0_a=47.14/7.33@18K, dog0_d=46.97/6.55@17K) ───
    # R8 bpsf=0.008+dgt=0.0003+psa=5-8 BEAT all sw configs (47.14 > 46.53)!
    # Lean into the R8 approach with refinements.
    "dog0      dog0_k   0.008  0.45  1.0  0.05  0.0003  0.015  0.1   5   0.15  0"     # R8 bpsf + sw7 quality params (ldn+pd)
    "dog0      dog0_l   0.008  0.40  1.0  0.05  0.0003  0.01   0.1   3   0.05  0"     # R8 bpsf + lower mop/psa for more quality
    "dog0      dog0_m   0.008  0.45  1.5  0.05  0.0003  0.01   0.1   5   0.05  0"     # R8 template + lmvg=1.5

    # ── gorilla5 (best PSNR: gor5_b=38.92/26.16, best g2s: sw6=38.06/7.17) ──
    # Huge PSNR↔g2s tension. psa HURTS g2s. NO psa. Focus on PSNR without g2s cost.
    "gorilla5  gor5_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.10  0"     # sw6 + pd=0.10 (mild g2s improvement)
    "gorilla5  gor5_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7-like quality push (best g2s combo)
    "gorilla5  gor5_m   0.005  0.30  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.30 for more GS → quality push

    # ── horse0 (hardest — best: sw7=34.82/14.49, sw8=33.96/14.16) ────
    # All g2s_max ≥ 14.16. Need both quality and count reduction.
    "horse0    hrs0_k   0.005  0.35  1.5  0.05  0.0001  0.01   0.1   0   0.10  0"     # sw7-like + ldn=0.01 for PSNR + mild pd
    "horse0    hrs0_l   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.40 + quality combo
    "horse0    hrs0_m   0.005  0.45  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # count reduction + ldn=0.02

    # ── NEW ANIMALS (no data → proven universal top-3: sw6/sw7/sw8) ───
    # sw6: best PSNR everywhere. sw7: best g2s_max. sw8: good count reduction.

    # gorilla8
    "gorilla8  gor8_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "gorilla8  gor8_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "gorilla8  gor8_m   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # horse10
    "horse10   hrs10_k  0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "horse10   hrs10_l  0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "horse10   hrs10_m  0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # michael0
    "michael0  mic0_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "michael0  mic0_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "michael0  mic0_m   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # michael2
    "michael2  mic2_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "michael2  mic2_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "michael2  mic2_m   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # michael16
    "michael16 mic16_k  0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "michael16 mic16_l  0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "michael16 mic16_m  0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # victoria0
    "victoria0 vic0_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "victoria0 vic0_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "victoria0 vic0_m   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # victoria2
    "victoria2 vic2_k   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "victoria2 vic2_l   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "victoria2 vic2_m   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # wolf0
    "wolf0     wolf0_k  0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # sw6 equivalent
    "wolf0     wolf0_l  0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # sw7 equivalent
    "wolf0     wolf0_m  0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # sw8 equivalent

    # ====================================================================
    # R9 configs (originally _e-_j / _a-_f, renamed to _n-_s to avoid
    # existing failed directories from the evaluate-only run)
    # ====================================================================

    # ── cat0 R9 (6 configs) ───────────────────────────────────────────
    "cat0      cat0_n   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # ldn=0.015 + pd=0.15
    "cat0      cat0_o   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 + lmvg=1.5
    "cat0      cat0_p   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.15  0"     # sw6 base + pd=0.15
    "cat0      cat0_q   0.005  0.30  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # lower mop=0.30 for max quality
    "cat0      cat0_r   0.005  0.35  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # ldn=0.02
    "cat0      cat0_s   0.005  0.40  1.5  0.05  0.0001  0.01   0.1   0   0.15  0"     # mop=0.40 + lmvg=1.5 + pd=0.15

    # ── cat2 R9 (6 configs) ───────────────────────────────────────────
    "cat2      cat2_n   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.15  0"     # sw6 + pd=0.15
    "cat2      cat2_o   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # ldn=0.015 + pd=0.15
    "cat2      cat2_p   0.005  0.35  1.5  0.05  0.0001  0.01   0.1   0   0.05  0"     # lmvg=1.5
    "cat2      cat2_q   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "cat2      cat2_r   0.005  0.35  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # ldn=0.02
    "cat2      cat2_s   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # lmvg=1.5+ldn=0.015+pd=0.15

    # ── centaur0 R9 (6 configs) ───────────────────────────────────────
    "centaur0  cen0_n   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "centaur0  cen0_o   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
    "centaur0  cen0_p   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55
    "centaur0  cen0_q   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality: lmvg=1.5+ldn=0.015+pd=0.15
    "centaur0  cen0_r   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.45 + ldn=0.015+pd=0.15
    "centaur0  cen0_s   0.005  0.50  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # mop=0.50 + ldn=0.02

    # ── centaur1 R9 (6 configs) ───────────────────────────────────────
    "centaur1  cen1_n   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "centaur1  cen1_o   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
    "centaur1  cen1_p   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55
    "centaur1  cen1_q   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality params
    "centaur1  cen1_r   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.45 + quality
    "centaur1  cen1_s   0.005  0.50  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # mop=0.50 + ldn=0.02

    # ── centaur5 R9 (6 configs) ───────────────────────────────────────
    "centaur5  cen5_n   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "centaur5  cen5_o   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
    "centaur5  cen5_p   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55
    "centaur5  cen5_q   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality params
    "centaur5  cen5_r   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.45 + quality
    "centaur5  cen5_s   0.005  0.50  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # mop=0.50 + ldn=0.02

    # ── david0 R9 (6 configs) ────────────────────────────────────────
    "david0    dav0_n   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "david0    dav0_o   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
    "david0    dav0_p   0.005  0.55  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.55
    "david0    dav0_q   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality focus
    "david0    dav0_r   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   3   0.05  0"     # light psa=3
    "david0    dav0_s   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   5   0.15  0"     # psa=5 + pd=0.15

    # ── dog0 R9 (6 configs) ──────────────────────────────────────────
    "dog0      dog0_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "dog0      dog0_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "dog0      dog0_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
    "dog0      dog0_q   0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality: lmvg=1.5+ldn=0.015+pd=0.15
    "dog0      dog0_r   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   3   0.05  0"     # light psa=3
    "dog0      dog0_s   0.005  0.45  1.5  0.05  0.0001  0.01   0.1   5   0.15  0"     # psa=5 + quality

    # ── gorilla5 R9 (6 configs) ──────────────────────────────────────
    "gorilla5  gor5_n   0.005  0.35  1.0  0.05  0.0001  0.01   0.1   0   0.15  0"     # sw6 + pd=0.15
    "gorilla5  gor5_o   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # ldn=0.015 + pd=0.15
    "gorilla5  gor5_p   0.005  0.35  1.5  0.05  0.0001  0.01   0.1   0   0.05  0"     # lmvg=1.5
    "gorilla5  gor5_q   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "gorilla5  gor5_r   0.005  0.35  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # ldn=0.02
    "gorilla5  gor5_s   0.005  0.35  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # combined quality

    # ── horse0 R9 (6 configs) ────────────────────────────────────────
    "horse0    hrs0_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "horse0    hrs0_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "horse0    hrs0_p   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality: lmvg=1.5+ldn=0.015+pd=0.15
    "horse0    hrs0_q   0.005  0.45  1.5  0.05  0.0001  0.01   0.1   0   0.15  0"     # mop=0.45 + quality params
    "horse0    hrs0_r   0.005  0.40  1.0  0.05  0.0001  0.02   0.1   0   0.05  0"     # ldn=0.02
    "horse0    hrs0_s   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── gorilla8 R9 (3 unique, excluding sw6/sw7/sw8 duplicates) ─────
    "gorilla8  gor8_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "gorilla8  gor8_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "gorilla8  gor8_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── horse10 R9 (3 unique) ────────────────────────────────────────
    "horse10   hrs10_n  0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "horse10   hrs10_o  0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "horse10   hrs10_p  0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── michael0 R9 (3 unique) ───────────────────────────────────────
    "michael0  mic0_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "michael0  mic0_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "michael0  mic0_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── michael2 R9 (3 unique) ───────────────────────────────────────
    "michael2  mic2_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "michael2  mic2_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "michael2  mic2_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── michael16 R9 (3 unique) ──────────────────────────────────────
    "michael16 mic16_n  0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "michael16 mic16_o  0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "michael16 mic16_p  0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── victoria0 R9 (3 unique) ──────────────────────────────────────
    "victoria0 vic0_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "victoria0 vic0_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "victoria0 vic0_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── victoria2 R9 (3 unique) ──────────────────────────────────────
    "victoria2 vic2_n   0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "victoria2 vic2_o   0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "victoria2 vic2_p   0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50

    # ── wolf0 R9 (3 unique) ──────────────────────────────────────────
    "wolf0     wolf0_n  0.005  0.40  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.40
    "wolf0     wolf0_o  0.005  0.45  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.45
    "wolf0     wolf0_p  0.005  0.50  1.0  0.05  0.0001  0.01   0.1   0   0.05  0"     # mop=0.50
)

# ============================================================================
# Default values
# ============================================================================
PROCESSED_BASE="TrainData/TOSCA/processed"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
GT_DATA_ROOT="TrainData/TOSCA/processed"
SHAPES="cat0,cat2,centaur0,centaur1,centaur5,david0,dog0,gorilla5,gorilla8,horse0,horse10,michael0,michael2,michael16,victoria0,victoria2,wolf0"
ANIMALS="" # "cat,centaur,david,dog,gorilla,horse,michael,victoria,wolf"
TEXTURES="blue"
COLMAP_RESOLUTIONS="high_res"
LIGHT_ID="0"
ITERATIONS=45000
MAX_PARALLEL=4
MAX_PARALLEL_CPU=30
WANDB_PROJECT="tosca-sweep"
USE_WANDB=true
SKIP_EXISTING=false
EVALUATE_ONLY=false
DRY_RUN=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --processed_base)       PROCESSED_BASE="$2";              shift 2 ;;
        --synth_data_base)      SYNTH_DATA_BASE="$2";             shift 2 ;;
        --gt_data_root)         GT_DATA_ROOT="$2";                shift 2 ;;
        --shapes)               SHAPES="$2";                      shift 2 ;;
        --animals)              ANIMALS="$2";                     shift 2 ;;
        --textures)             TEXTURES="$2";                    shift 2 ;;
        --colmap_resolutions)   COLMAP_RESOLUTIONS="$2";          shift 2 ;;
        --light_id)             LIGHT_ID="$2";                    shift 2 ;;
        --iterations)           ITERATIONS="$2";                  shift 2 ;;
        --max_parallel)         MAX_PARALLEL="$2";                shift 2 ;;
        --max_parallel_cpu)     MAX_PARALLEL_CPU="$2";            shift 2 ;;
        --wandb_project)        WANDB_PROJECT="$2";               shift 2 ;;
        --no_wandb)             USE_WANDB=false;                  shift   ;;
        --skip_existing)        SKIP_EXISTING=true;               shift   ;;
        --evaluate_only)        EVALUATE_ONLY=true;               shift   ;;
        --dry_run)              DRY_RUN=true;                     shift   ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Setup paths
# ============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="$PROJECT_ROOT/train.py"
MESH_EXTRACT_SCRIPT="$PROJECT_ROOT/mesh_extract_tetrahedra.py"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/tosca/benchmark/evaluate_benchmark.py"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

for f in "$TRAIN_SCRIPT" "$MESH_EXTRACT_SCRIPT" "$EVAL_SCRIPT"; do
    [ -f "$f" ] || { echo "Error: script not found: $f"; exit 1; }
done

# ============================================================================
# Expand animals → shape list
# ============================================================================
if [ -n "$ANIMALS" ] && [ -z "$SHAPES" ]; then
    SHAPES="$(expand_animals "$ANIMALS")"
    [ -n "$SHAPES" ] || { echo "Error: No shapes found for animals: $ANIMALS"; exit 1; }
fi

IFS=',' read -ra SHAPE_ARRAY        <<< "$SHAPES"
IFS=',' read -ra TEXTURE_ARRAY      <<< "$TEXTURES"
IFS=',' read -ra RESOLUTION_ARRAY   <<< "$COLMAP_RESOLUTIONS"

# ============================================================================
# GPU detection
# ============================================================================
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
if [ "$MAX_PARALLEL" -gt "$NUM_GPUS" ]; then
    echo "  [INFO] Clamping MAX_PARALLEL from $MAX_PARALLEL → $NUM_GPUS (NUM_GPUS)"
    MAX_PARALLEL=$NUM_GPUS
fi

# ============================================================================
# Print configuration
# ============================================================================
TOTAL_SHAPES=${#SHAPE_ARRAY[@]}
TOTAL_CONFIGS=${#SWEEP_CONFIGS[@]}

# Count actual jobs (per-shape configs only match their target shape)
TOTAL_JOBS=0
for shape in "${SHAPE_ARRAY[@]}"; do
    for config_line in "${SWEEP_CONFIGS[@]}"; do
        read -r target_shape _rest <<< "$config_line"
        [[ "$shape" == "$target_shape" ]] && TOTAL_JOBS=$((TOTAL_JOBS + ${#TEXTURE_ARRAY[@]} * ${#RESOLUTION_ARRAY[@]}))
    done
done

echo "============================================================"
echo "TOSCA Benchmark Sweep — ${TOTAL_CONFIGS} per-shape configs × ${TOTAL_SHAPES} shapes"
echo "============================================================"
echo "  Shapes:              ${SHAPE_ARRAY[*]}"
echo "  Textures:            ${TEXTURE_ARRAY[*]}"
echo "  Resolutions:         ${RESOLUTION_ARRAY[*]}"
echo "  Iterations:          $ITERATIONS"
echo "  Config entries:      $TOTAL_CONFIGS (per-shape)"
echo "  Total jobs:          $TOTAL_JOBS"
echo "  Max parallel GPU:    $MAX_PARALLEL"
echo "  Max parallel CPU:    $MAX_PARALLEL_CPU"
echo "  GPUs detected:       $NUM_GPUS"
echo "  Light ID:            $LIGHT_ID"
echo "  Skip existing:       $SKIP_EXISTING"
echo "  Evaluate only:       $EVALUATE_ONLY"
echo "  Dry run:             $DRY_RUN"
echo ""

# ============================================================================
# Core single-run function
# ============================================================================
run_single_sweep() {
    local shape="$1"
    local texture="$2"
    local resolution="$3"
    local config_name="$4"
    local extra_args="$5"
    local gpu_id="$6"
    local gt_mesh_path="$7"

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    # Construct source/model paths
    SOURCE_PATH="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${LIGHT_ID}"
    MODEL_PATH="$SOURCE_PATH/sweep_${config_name}"

    local JOB_NAME="$shape/$resolution/$config_name"
    echo "------------------------------------------------------------"
    echo "JOB: $JOB_NAME  [GPU $gpu_id]"

    # Check processed mesh exists
    if ! ls "$PROCESSED_BASE/$shape/mesh_${resolution}_"*.ply > /dev/null 2>&1; then
        echo "  [SKIP] No ${resolution} mesh in $PROCESSED_BASE/$shape/"
        return 0
    fi

    # Check source data exists
    local img_count
    img_count=$(ls "$SOURCE_PATH/images/" 2>/dev/null | wc -l)
    if [ ! -d "$SOURCE_PATH" ] || [ "$img_count" -eq 0 ]; then
        echo "  [SKIP] Source not rendered: $SOURCE_PATH"
        return 0
    fi

    mkdir -p "$MODEL_PATH"

    # Skip if already done
    if [ "$SKIP_EXISTING" = true ] && [ -f "$MODEL_PATH/benchmark_report.json" ]; then
        echo "  [SKIP] Already has benchmark_report.json"
        return 0
    fi

    # --- Training ---
    local train_done_marker="$MODEL_PATH/chkpnt${ITERATIONS}.txt"
    local training_ran=false

    if [ -f "$train_done_marker" ]; then
        echo "  [TRAIN] Already done — skipping"
    else
        local latest_ckpt
        latest_ckpt=$(ls -t "$MODEL_PATH"/chkpnt*.pth 2>/dev/null | head -1 || true)

        local train_cmd="$PYTHON3 $TRAIN_SCRIPT \
            -s $SOURCE_PATH \
            -m $MODEL_PATH \
            --eval \
            --iterations $ITERATIONS \
            $extra_args"

        if [ -n "$latest_ckpt" ]; then
            echo "  [TRAIN] Resuming from $(basename "$latest_ckpt")"
            train_cmd="$train_cmd --start_checkpoint $latest_ckpt"
        else
            echo "  [TRAIN] Starting fresh"
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] $train_cmd"
            training_ran=true
        else
            # Save args
            echo "$extra_args" > "$MODEL_PATH/benchmark_args.txt"
            if eval "$train_cmd" > "$MODEL_PATH/train.log" 2>&1; then
                echo "  [TRAIN] Done: $JOB_NAME"
                training_ran=true
            else
                echo "  [TRAIN FAIL] $JOB_NAME"
                return 1
            fi
        fi
    fi

    # --- Mesh extraction ---
    if [ -f "$MODEL_PATH/recon.ply" ] && [ "$training_ran" = false ]; then
        echo "  [MESH] Already done — skipping"
    else
        local mesh_cmd="$PYTHON3 $MESH_EXTRACT_SCRIPT \
            -s $SOURCE_PATH \
            -m $MODEL_PATH \
            --eval"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] $mesh_cmd"
        else
            if eval "$mesh_cmd" > "$MODEL_PATH/mesh_extract.log" 2>&1; then
                echo "  [MESH] Done: $JOB_NAME"
            else
                echo "  [MESH FAIL] $JOB_NAME"
                return 1
            fi
        fi
    fi

    # --- Evaluation deferred to CPU phase ---
    return 0
}

export -f run_single_sweep
export PROCESSED_BASE SYNTH_DATA_BASE PYTHON3 TRAIN_SCRIPT MESH_EXTRACT_SCRIPT EVAL_SCRIPT
export ITERATIONS DRY_RUN SKIP_EXISTING LIGHT_ID

# ============================================================================
# Parallel job manager (same as train_tosca_smart.sh)
# ============================================================================
RUNNING_PIDS=()
RUNNING_JOBS=()
RUNNING_GPU_SLOTS=()
RUNNING_MODEL_PATHS=()
RUNNING_GT_MESHES=()
COMPLETED=0
FAILED_JOBS=()

# Pending CPU evaluations (populated as GPU jobs complete)
PENDING_EVAL_DIRS=()
PENDING_EVAL_GT=()
PENDING_EVAL_NAMES=()

FREE_GPU_SLOTS=()
for ((g=0; g<NUM_GPUS; g++)); do FREE_GPU_SLOTS+=($g); done

_pop_gpu_slot() {
    ASSIGNED_GPU="${FREE_GPU_SLOTS[0]}"
    FREE_GPU_SLOTS=("${FREE_GPU_SLOTS[@]:1}")
}

_release_gpu_slot() {
    FREE_GPU_SLOTS+=("$1")
}

wait_for_slot() {
    while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ] || [ ${#FREE_GPU_SLOTS[@]} -eq 0 ]; do
        local NEW_PIDS=() NEW_JOBS=() NEW_GPU_SLOTS=() NEW_MDLS=() NEW_GTS=()
        for i in "${!RUNNING_PIDS[@]}"; do
            local pid="${RUNNING_PIDS[$i]}" job="${RUNNING_JOBS[$i]}" gpu_slot="${RUNNING_GPU_SLOTS[$i]}"
            local mdl="${RUNNING_MODEL_PATHS[$i]}" gt="${RUNNING_GT_MESHES[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid"); NEW_JOBS+=("$job"); NEW_GPU_SLOTS+=("$gpu_slot")
                NEW_MDLS+=("$mdl"); NEW_GTS+=("$gt")
            else
                _release_gpu_slot "$gpu_slot"
                if wait "$pid"; then
                    echo "  [GPU DONE] $job"
                    COMPLETED=$((COMPLETED + 1))
                    PENDING_EVAL_DIRS+=("$mdl")
                    PENDING_EVAL_GT+=("$gt")
                    PENDING_EVAL_NAMES+=("$job")
                else
                    echo "  [FAILED] $job"
                    FAILED_JOBS+=("$job")
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_JOBS=("${NEW_JOBS[@]}")
        RUNNING_GPU_SLOTS=("${NEW_GPU_SLOTS[@]}")
        RUNNING_MODEL_PATHS=("${NEW_MDLS[@]}")
        RUNNING_GT_MESHES=("${NEW_GTS[@]}")
        if [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ] || [ ${#FREE_GPU_SLOTS[@]} -eq 0 ]; then
            sleep 5
        fi
    done
    return 0
}

wait_for_all() {
    for i in "${!RUNNING_PIDS[@]}"; do
        local pid="${RUNNING_PIDS[$i]}" job="${RUNNING_JOBS[$i]}"
        local mdl="${RUNNING_MODEL_PATHS[$i]}" gt="${RUNNING_GT_MESHES[$i]}"
        if wait "$pid"; then
            echo "  [GPU DONE] $job"
            COMPLETED=$((COMPLETED + 1))
            PENDING_EVAL_DIRS+=("$mdl")
            PENDING_EVAL_GT+=("$gt")
            PENDING_EVAL_NAMES+=("$job")
        else
            echo "  [FAILED] $job"
            FAILED_JOBS+=("$job")
        fi
    done
    RUNNING_PIDS=(); RUNNING_JOBS=(); RUNNING_GPU_SLOTS=()
    RUNNING_MODEL_PATHS=(); RUNNING_GT_MESHES=()
}

# ============================================================================
# Main loop
# ============================================================================
START_TIME=$(date +%s)
CURRENT_JOB=0

if [ "$EVALUATE_ONLY" = true ]; then
    echo "============================================================"
    echo "EVALUATE-ONLY MODE: Scanning for runs needing evaluation..."
    echo "============================================================"
    for shape in "${SHAPE_ARRAY[@]}"; do
        for texture in "${TEXTURE_ARRAY[@]}"; do
            for resolution in "${RESOLUTION_ARRAY[@]}"; do
                GT_MESH=$(find "$GT_DATA_ROOT/$shape" -name "mesh_${resolution}_*.ply" 2>/dev/null | sort | head -1)
                for config_line in "${SWEEP_CONFIGS[@]}"; do
                    read -r target_shape cfg_name _rest <<< "$config_line"
                    [[ "$shape" != "$target_shape" ]] && continue
                    LOCAL_SOURCE="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${LIGHT_ID}"
                    LOCAL_MODEL="$LOCAL_SOURCE/sweep_${cfg_name}"
                    if [ -f "$LOCAL_MODEL/recon.ply" ] && [ ! -f "$LOCAL_MODEL/benchmark_report.json" ]; then
                        PENDING_EVAL_DIRS+=("$LOCAL_MODEL")
                        PENDING_EVAL_GT+=("$GT_MESH")
                        PENDING_EVAL_NAMES+=("$shape/$resolution/$cfg_name")
                    fi
                done
            done
        done
    done
    echo "  Found ${#PENDING_EVAL_DIRS[@]} runs to evaluate"
else
# --- Begin GPU phase ---

for shape in "${SHAPE_ARRAY[@]}"; do
    for texture in "${TEXTURE_ARRAY[@]}"; do
        for resolution in "${RESOLUTION_ARRAY[@]}"; do
            # Find GT mesh for this shape+resolution
            GT_MESH=$(find "$GT_DATA_ROOT/$shape" -name "mesh_${resolution}_*.ply" 2>/dev/null | sort | head -1)
            if [ -z "$GT_MESH" ]; then
                echo "[WARN] No GT mesh for $shape/$resolution — skipping shape"
                continue
            fi

            for config_line in "${SWEEP_CONFIGS[@]}"; do
                # Parse per-shape config line
                read -r target_shape cfg_name bpsf mop lmvg ldist dgt ldn ldssim psa pd pmst <<< "$config_line"
                [[ "$shape" != "$target_shape" ]] && continue
                CURRENT_JOB=$((CURRENT_JOB + 1))

                # Build extra args
                local_args="--big_point_scale_factor $bpsf \
                    --min_opacity_prune $mop \
                    --lambda_multi_view_geo $lmvg \
                    --lambda_distortion $ldist \
                    --densify_grad_threshold $dgt \
                    --lambda_depth_normal $ldn \
                    --lambda_dssim $ldssim \
                    --prune_scale_anisotropy $psa \
                    --percent_dense $pd \
                    --prune_min_scale_threshold ${pmst:-0} \
                    --lambda_multi_view_ncc 0.3"

                JOB_NAME="$shape/$resolution/$cfg_name"
                LOG_DIR="$PROJECT_ROOT/logs/benchmark_sweep"
                mkdir -p "$LOG_DIR"
                LOG_FILE="$LOG_DIR/${shape}_${texture}_${resolution}_${cfg_name}.log"

                # Compute model path for tracking
                LOCAL_SOURCE="$SYNTH_DATA_BASE/${texture}_texture/$shape/$resolution/light_${LIGHT_ID}"
                LOCAL_MODEL="$LOCAL_SOURCE/sweep_${cfg_name}"

                if [ "$MAX_PARALLEL" -gt 1 ]; then
                    wait_for_slot
                    _pop_gpu_slot
                    (run_single_sweep "$shape" "$texture" "$resolution" \
                        "$cfg_name" "$local_args" "$ASSIGNED_GPU" "$GT_MESH") \
                        >> "$LOG_FILE" 2>&1 &
                    RUNNING_PIDS+=($!)
                    RUNNING_JOBS+=("$JOB_NAME")
                    RUNNING_GPU_SLOTS+=("$ASSIGNED_GPU")
                    RUNNING_MODEL_PATHS+=("$LOCAL_MODEL")
                    RUNNING_GT_MESHES+=("$GT_MESH")
                    echo "[$CURRENT_JOB/$TOTAL_JOBS] Launched: $JOB_NAME  (PID $! GPU $ASSIGNED_GPU)"
                else
                    _pop_gpu_slot
                    echo "============================================================"
                    echo "[$CURRENT_JOB/$TOTAL_JOBS] $JOB_NAME  [GPU $ASSIGNED_GPU]"
                    echo "============================================================"
                    if run_single_sweep "$shape" "$texture" "$resolution" \
                            "$cfg_name" "$local_args" "$ASSIGNED_GPU" "$GT_MESH"; then
                        COMPLETED=$((COMPLETED + 1))
                        PENDING_EVAL_DIRS+=("$LOCAL_MODEL")
                        PENDING_EVAL_GT+=("$GT_MESH")
                        PENDING_EVAL_NAMES+=("$JOB_NAME")
                    else
                        FAILED_JOBS+=("$JOB_NAME")
                    fi
                    _release_gpu_slot "$ASSIGNED_GPU"
                fi
            done
        done
    done
done

if [ "$MAX_PARALLEL" -gt 1 ]; then
    wait_for_all
fi

fi  # end of GPU phase (evaluate_only check)

# ============================================================================
# Phase 2: CPU-only evaluation (no GPU required)
# ============================================================================
echo ""
echo "============================================================"
echo "Phase 2: CPU Evaluation — ${#PENDING_EVAL_DIRS[@]} jobs, max $MAX_PARALLEL_CPU parallel"
echo "============================================================"

EVAL_RUNNING_PIDS=()
EVAL_RUNNING_NAMES=()
CPU_COMPLETED=0
CPU_FAILED=0

_wait_for_cpu_slot() {
    while [ ${#EVAL_RUNNING_PIDS[@]} -ge "$MAX_PARALLEL_CPU" ]; do
        local NEW_PIDS=() NEW_NAMES=()
        for i in "${!EVAL_RUNNING_PIDS[@]}"; do
            local pid="${EVAL_RUNNING_PIDS[$i]}" name="${EVAL_RUNNING_NAMES[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid"); NEW_NAMES+=("$name")
            else
                if wait "$pid"; then
                    echo "  [EVAL DONE] $name"
                    CPU_COMPLETED=$((CPU_COMPLETED + 1))
                else
                    echo "  [EVAL FAIL] $name"
                    CPU_FAILED=$((CPU_FAILED + 1))
                fi
            fi
        done
        EVAL_RUNNING_PIDS=("${NEW_PIDS[@]}")
        EVAL_RUNNING_NAMES=("${NEW_NAMES[@]}")
        if [ ${#EVAL_RUNNING_PIDS[@]} -ge "$MAX_PARALLEL_CPU" ]; then
            sleep 2
        fi
    done
}

for i in "${!PENDING_EVAL_DIRS[@]}"; do
    local_model="${PENDING_EVAL_DIRS[$i]}"
    local_gt="${PENDING_EVAL_GT[$i]}"
    local_name="${PENDING_EVAL_NAMES[$i]}"

    # Skip if eval already done
    if [ "$SKIP_EXISTING" = true ] && [ -f "$local_model/benchmark_report.json" ]; then
        echo "  [EVAL SKIP] $local_name — already has report"
        CPU_COMPLETED=$((CPU_COMPLETED + 1))
        continue
    fi

    _wait_for_cpu_slot

    gt_arg=""
    [ -n "$local_gt" ] && gt_arg="--gt_mesh $local_gt"

    (
        CUDA_VISIBLE_DEVICES="" $PYTHON3 "$EVAL_SCRIPT" \
            --output_dir "$local_model" \
            $gt_arg \
            --iteration "$ITERATIONS" > "$local_model/eval.log" 2>&1
    ) &
    EVAL_RUNNING_PIDS+=($!)
    EVAL_RUNNING_NAMES+=("$local_name")
    echo "  [EVAL] Launched: $local_name (PID $!)"
done

# Wait for remaining CPU evals
for i in "${!EVAL_RUNNING_PIDS[@]}"; do
    pid="${EVAL_RUNNING_PIDS[$i]}" name="${EVAL_RUNNING_NAMES[$i]}"
    if wait "$pid"; then
        echo "  [EVAL DONE] $name"
        CPU_COMPLETED=$((CPU_COMPLETED + 1))
    else
        echo "  [EVAL FAIL] $name"
        CPU_FAILED=$((CPU_FAILED + 1))
    fi
done

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
echo "Benchmark Sweep Complete"
echo "============================================================"
echo "  Total time:   ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  GPU jobs:     $CURRENT_JOB  (completed: $COMPLETED, failed: ${#FAILED_JOBS[@]})"
echo "  CPU evals:    $((CPU_COMPLETED + CPU_FAILED))  (completed: $CPU_COMPLETED, failed: $CPU_FAILED)"

if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Failed jobs:"
    for j in "${FAILED_JOBS[@]}"; do echo "  - $j"; done
fi
