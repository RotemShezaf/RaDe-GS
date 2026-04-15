#!/usr/bin/env bash
# ============================================================================
# TOSCA Benchmark Sweep — Round 11
# ============================================================================
# 9 configs per animal (17 shapes × 9 = 153 configs)
# Data-driven design based on analysis of all R1-R10 results.
#
# Config format:
#   "target_shape  name  bpsf  mop  lmvg  ldist  dgt  ldn  ldssim  psa  pd  pmst"
#   (all configs implicitly use --lambda_multi_view_ncc 0.3)
#
# Naming: _t through _y (data-driven), _z/_aa/_ab (exploratory) for all shapes
#
# Key insights driving config design:
#   - ldn=0.01 → best PSNR universally; ldn=0.015 → often best g2s_max
#   - pd=0.15 helps g2s_max in most shapes (EXCEPT cat2 where it kills GS count)
#   - lmvg=1.5 often slightly helps geometry
#   - psa ONLY safe for david0/dog0 (catastrophic elsewhere)
#   - bpsf=0.008+dgt≥0.0002 only for david0/dog0
#   - Higher mop reduces GS count; sweet spot is shape-dependent
#   - Centaurs/horses need low mop (≤0.40) for adequate GS counts
#
# Usage:
#   bash scripts/tosca/benchmark/benchmark_sweep_r11.sh [options]
#   bash scripts/tosca/benchmark/benchmark_sweep_r11.sh --dry_run
#   bash scripts/tosca/benchmark/benchmark_sweep_r11.sh --shapes cat0,cat2 --dry_run
#   bash scripts/tosca/benchmark/benchmark_sweep_r11.sh --evaluate_only
# ============================================================================
set -euo pipefail

# ============================================================================
# R11 Sweep Configurations — 153 configs (9 per shape: 6 data-driven + 3 exploratory)
# ============================================================================
# Column order: target_shape name bpsf mop lmvg ldist dgt ldn ldssim psa pd pmst
SWEEP_CONFIGS=(

    # ── cat0 (21 prior runs, 77-90K GS) ──────────────────────────────
    # Best: sw6=5.10 g2s_max, cat0_p=48.85 PSNR. mop=0.35-0.40 sweet spot.
    # Key: ldn=0.015 avg 6.18 g2s_max. pd=0.15 helps. lmvg=1.5 helps.
    # Strategy: push mop=0.40 (proven 5.52 with cat0_s), try mop=0.45.
    "cat0      cat0_t   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.15  0"     # mop=0.40 + pd=0.15 (cat0_s was lmvg=1.5, test lmvg=1.0)
    "cat0      cat0_u   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.40 + lmvg=1.5 + ldn=0.015 + pd=0.15 (combine all best factors)
    "cat0      cat0_v   0.005  0.40  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # mop=0.40 + ldn=0.015 + pd=0.15 (lmvg=1.0 version)
    "cat0      cat0_w   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # test mop=0.45 (never tried, may reduce GS further)
    "cat0      cat0_x   0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # aggressive: mop=0.45 + all quality params
    "cat0      cat0_y   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.40 baseline (isolate mop effect vs sw6)

    # ── cat2 (21 prior runs, 50-156K GS) ─────────────────────────────
    # Best: sw6 dominates all metrics (44.15/10.27). pd=0.15 KILLS GS count (51K).
    # Key: Keep pd=0.05. Lower mop for more GS. sw configs (104K+) >> R9/R10 (50K).
    # Strategy: lower mop + keep pd=0.05 to maintain high GS count.
    "cat2      cat2_t   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop=0.30 for more GS + best ldn
    "cat2      cat2_u   0.005  0.25  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.25 + ldn=0.01 (sw4 was mop=0.25+ldn=0.04)
    "cat2      cat2_v   0.005  0.30  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.30 + lmvg=1.5 (quality push without pd penalty)
    "cat2      cat2_w   0.005  0.30  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # mop=0.30 + ldn=0.015
    "cat2      cat2_x   0.005  0.30  1.5  0.05  0.0001  0.015  0.1   0   0.05  0"     # mop=0.30 + lmvg=1.5 + ldn=0.015
    "cat2      cat2_y   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at mop=0.35 without pd (novel)

    # ── centaur0 (20 prior runs, 20-344K GS) ─────────────────────────
    # Best: sw6=45.83/13.32 at 308K. mop≤0.40 critical. NO psa.
    # mop=0.45+ collapses to 37 PSNR / 89K GS.
    # Strategy: explore mop=0.35-0.40 with quality params.
    "centaur0  cen0_t   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.40 + ldn=0.01 (sw8 was ldn=0.02)
    "centaur0  cen0_u   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.05  0"     # mop=0.40 + lmvg=1.5 + ldn=0.015 (no pd penalty)
    "centaur0  cen0_v   0.005  0.40  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # mop=0.40 + ldn=0.015 + mild pd=0.10
    "centaur0  cen0_w   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # mop=0.35 + ldn=0.015 + pd=0.10 (balanced)
    "centaur0  cen0_x   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mop=0.40 + pd=0.10 (between sw6 and sw8)
    "centaur0  cen0_y   0.005  0.35  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # sw6 + lmvg=1.5 (no other changes)

    # ── centaur1 (20 prior runs, 9-365K GS) ──────────────────────────
    # Best PSNR: sw6=46.17 at 329K. Best g2s_max: sw1=11.62 (ldn=0.04).
    # Strategy: mop=0.35-0.40 with low ldn.
    "centaur1  cen1_t   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.40 + ldn=0.01 (sw8 was ldn=0.02)
    "centaur1  cen1_u   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.05  0"     # quality push without pd penalty
    "centaur1  cen1_v   0.005  0.35  1.0  0.05  0.0001  0.020  0.1   0   0.05  0"     # ldn=0.02 at mop=0.35 (not tried, sw8 was mop=0.40)
    "centaur1  cen1_w   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # balanced: ldn=0.015 + pd=0.10
    "centaur1  cen1_x   0.005  0.40  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # mop=0.40 + ldn=0.015 + pd=0.10
    "centaur1  cen1_y   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mop=0.40 + mild pd

    # ── centaur5 (21 prior runs, 9-375K GS) ──────────────────────────
    # Best PSNR: sw6=46.04 at 334K. Best g2s_max: cen5_n=13.38 (mop=0.45, pd=0.05).
    # mop=0.45+pd=0.05 gave BEST g2s_max. NO psa.
    # Strategy: explore mop=0.40-0.45 with quality params.
    "centaur5  cen5_t   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # between sw6 (0.35) and cen5_n (0.45)
    "centaur5  cen5_u   0.005  0.45  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # add lmvg=1.5 to best g2s_max config
    "centaur5  cen5_v   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at best mop+pd combo
    "centaur5  cen5_w   0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.05  0"     # quality at moderate mop
    "centaur5  cen5_x   0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.10  0"     # balanced: mop=0.45 + quality + mild pd
    "centaur5  cen5_y   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   0   0.15  0"     # cen5_n + pd=0.15 (test pd effect at this mop)

    # ── david0 (20 prior runs, 7-481K GS) ────────────────────────────
    # Best: dav0_s=48.27/6.68 (mop=0.50, psa=5, pd=0.15). psa HELPS here.
    # mop=0.45-0.50 + psa=3-5 is the sweet spot.
    # Strategy: combine psa with pd and mop variations.
    "david0    dav0_t   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   5   0.15  0"     # dav0_k recipe but bpsf=0.005 + pd=0.15
    "david0    dav0_u   0.005  0.50  1.0  0.05  0.0001  0.010  0.1   3   0.15  0"     # like dav0_s but psa=3 (lighter prune)
    "david0    dav0_v   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   5   0.10  0"     # like dav0_k but standard bpsf/dgt
    "david0    dav0_w   0.005  0.50  1.5  0.05  0.0001  0.010  0.1   5   0.15  0"     # quality combo: lmvg=1.5 + psa=5
    "david0    dav0_x   0.005  0.45  1.5  0.05  0.0001  0.010  0.1   3   0.15  0"     # mop=0.45 + lmvg=1.5 + mild psa
    "david0    dav0_y   0.005  0.55  1.0  0.05  0.0001  0.010  0.1   5   0.15  0"     # higher mop=0.55 to reduce GS + psa=5

    # ── dog0 (21 prior runs, 8-316K GS) ──────────────────────────────
    # Best: dog0_r=47.60/5.47 (mop=0.40, psa=3, pd=0.05). psa=3 HELPS.
    # Strategy: explore psa=3-5 at mop=0.40-0.50.
    "dog0      dog0_t   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   3   0.15  0"     # dog0_r + pd=0.15 (test if pd helps here)
    "dog0      dog0_u   0.005  0.40  1.5  0.05  0.0001  0.010  0.1   3   0.05  0"     # add lmvg=1.5 to best config
    "dog0      dog0_v   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   3   0.05  0"     # higher mop + psa=3
    "dog0      dog0_w   0.005  0.40  1.0  0.05  0.0001  0.015  0.1   3   0.10  0"     # ldn=0.015 + psa=3 + pd=0.10
    "dog0      dog0_x   0.005  0.50  1.0  0.05  0.0001  0.010  0.1   3   0.05  0"     # mop=0.50 + psa=3
    "dog0      dog0_y   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   5   0.15  0"     # stronger psa=5 + pd=0.15

    # ── gorilla5 (21 prior runs, 64-165K GS) ─────────────────────────
    # Best g2s_max: sw6=8.49 (mop=0.35, ldn=0.01, pd=0.05). NO psa.
    # psa CATASTROPHIC. Higher mop/pd often hurts. Needs many GS.
    # Strategy: low mop (0.25-0.35) with minor param variations.
    "gorilla5  gor5_t   0.005  0.25  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lowest mop + ldn=0.01 (sw4 was ldn=0.04)
    "gorilla5  gor5_u   0.005  0.30  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.30 + lmvg=1.5
    "gorilla5  gor5_v   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without pd (test ldn alone)
    "gorilla5  gor5_w   0.005  0.30  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # lower mop + ldn=0.015
    "gorilla5  gor5_x   0.005  0.35  1.5  0.05  0.0001  0.010  0.1   0   0.10  0"     # lmvg=1.5 + moderate pd
    "gorilla5  gor5_y   0.005  0.40  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.40 + lmvg=1.5

    # ── gorilla8 (6 prior runs, 53-60K GS) ───────────────────────────
    # Best: gor8_l=17.19 (lmvg=1.5, ldn=0.015, pd=0.15). Very limited data.
    # Only tested mop=0.35-0.50. Lower mop not tried.
    # Strategy: try lower mop for more GS + various quality params.
    "gorilla8  gor8_t   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS
    "gorilla8  gor8_u   0.005  0.25  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # even lower mop
    "gorilla8  gor8_v   0.005  0.30  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # like gor8_l but lower mop
    "gorilla8  gor8_w   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without pd
    "gorilla8  gor8_x   0.005  0.35  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=1.5 without ldn/pd changes
    "gorilla8  gor8_y   0.005  0.30  1.0  0.05  0.0001  0.015  0.1   0   0.10  0"     # lower mop + balanced quality

    # ── horse0 (19 prior runs, 42-257K GS) ────────────────────────────
    # Hardest shape. Best: sw4=22.81 (mop=0.25, ldn=0.04, pd=0.10) at 257K.
    # All g2s_max ≥ 22.81. Needs lowest mop + many GS. NO psa/bpsf=0.008.
    # Strategy: low mop (0.25-0.30) with ldn=0.01-0.015 (untested combo).
    "horse0    hrs0_t   0.005  0.25  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lowest mop + best ldn (not tried!)
    "horse0    hrs0_u   0.005  0.25  1.5  0.05  0.0001  0.015  0.1   0   0.10  0"     # like sw4 but better ldn + lmvg
    "horse0    hrs0_v   0.005  0.30  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.30 + lmvg=1.5
    "horse0    hrs0_w   0.005  0.25  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # lowest mop + ldn=0.015
    "horse0    hrs0_x   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # mop=0.30 + ldn=0.01 (sw3 was ldn=0.04)
    "horse0    hrs0_y   0.005  0.25  1.5  0.05  0.0001  0.010  0.1   0   0.10  0"     # lowest mop + lmvg=1.5 + pd=0.10

    # ── horse10 (6 prior runs, 104-124K GS) ──────────────────────────
    # Best: hrs10_n=22.69 (mop=0.40, ldn=0.01, pd=0.05).
    # Only tested mop=0.35-0.50. Lower mop not tried.
    # Strategy: try lower mop + quality params.
    "horse10   hrs10_t  0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS
    "horse10   hrs10_u  0.005  0.25  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # even lower mop
    "horse10   hrs10_v  0.005  0.30  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # lower mop + quality params
    "horse10   hrs10_w  0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without pd
    "horse10   hrs10_x  0.005  0.40  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality at best mop (like hrs10_l style)
    "horse10   hrs10_y  0.005  0.35  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mild pd at mop=0.35

    # ── michael0 (6 prior runs, 110-128K GS) ─────────────────────────
    # Best: mic0_k=48.20/8.53 (mop=0.35, ldn=0.01, pd=0.05).
    # Only tested mop=0.35-0.50. ldn=0.01 best.
    # Strategy: lower mop + minor quality variations.
    "michael0  mic0_t   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS
    "michael0  mic0_u   0.005  0.35  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # add lmvg=1.5 (mic0_l was ldn=0.015+pd=0.15)
    "michael0  mic0_v   0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mild pd at mop=0.40
    "michael0  mic0_w   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without pd
    "michael0  mic0_x   0.005  0.40  1.5  0.05  0.0001  0.010  0.1   0   0.15  0"     # quality combo at mop=0.40
    "michael0  mic0_y   0.005  0.30  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # lower mop + full quality push

    # ── michael2 (6 prior runs, 70-78K GS) ───────────────────────────
    # Best: mic2_o=47.19/8.98 (mop=0.45, ldn=0.01, pd=0.05).
    # Only tested mop=0.35-0.50. Tight result range (8.98-10.41).
    # Strategy: lower mop + test ldn/lmvg variations.
    "michael2  mic2_t   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS
    "michael2  mic2_u   0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality at best mop
    "michael2  mic2_v   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at best mop
    "michael2  mic2_w   0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without pd
    "michael2  mic2_x   0.005  0.40  1.5  0.05  0.0001  0.010  0.1   0   0.15  0"     # quality at mop=0.40
    "michael2  mic2_y   0.005  0.30  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop + lmvg

    # ── michael16 (6 prior runs, 100-118K GS) ────────────────────────
    # Best: mic16_n=47.79/6.81 (mop=0.40, ldn=0.01, pd=0.05). Very good!
    # Only tested mop=0.35-0.50. ldn=0.01 dominant.
    # Strategy: try lower mop + quality params.
    "michael16 mic16_t  0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # quality at mop=0.45
    "michael16 mic16_u  0.005  0.40  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # add lmvg=1.5 to best config
    "michael16 mic16_v  0.005  0.40  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at best mop
    "michael16 mic16_w  0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # lower mop + ldn=0.015
    "michael16 mic16_x  0.005  0.40  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mild pd at best mop
    "michael16 mic16_y  0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS

    # ── victoria0 (6 prior runs, 95-108K GS) ─────────────────────────
    # Best: vic0_o=48.38/6.06 (mop=0.45, ldn=0.01, pd=0.05). Excellent!
    # Strategy: explore around mop=0.45 + add quality params.
    "victoria0 vic0_t   0.005  0.45  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # add lmvg=1.5 to best config
    "victoria0 vic0_u   0.005  0.45  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at best mop
    "victoria0 vic0_v   0.005  0.45  1.0  0.05  0.0001  0.010  0.1   0   0.10  0"     # mild pd at best mop
    "victoria0 vic0_w   0.005  0.40  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg + mop=0.40
    "victoria0 vic0_x   0.005  0.50  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # full quality at mop=0.50
    "victoria0 vic0_y   0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop for more GS

    # ── victoria2 (6 prior runs, 109-131K GS) ────────────────────────
    # Best g2s_max: vic2_m=7.29 (mop=0.40, ldn=0.02). Best PSNR: 49.06.
    # Unique: ldn=0.02 gave best g2s_max (not 0.01 or 0.015).
    # Strategy: explore ldn=0.02 variations + quality params.
    "victoria2 vic2_t   0.005  0.40  1.5  0.05  0.0001  0.020  0.1   0   0.05  0"     # add lmvg=1.5 to best config
    "victoria2 vic2_u   0.005  0.45  1.0  0.05  0.0001  0.020  0.1   0   0.05  0"     # higher mop + best ldn
    "victoria2 vic2_v   0.005  0.40  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 at mop=0.40
    "victoria2 vic2_w   0.005  0.40  1.0  0.05  0.0001  0.020  0.1   0   0.10  0"     # mild pd at best config
    "victoria2 vic2_x   0.005  0.45  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # full quality at mop=0.45
    "victoria2 vic2_y   0.005  0.50  1.0  0.05  0.0001  0.020  0.1   0   0.05  0"     # mop=0.50 + best ldn

    # ── wolf0 (6 prior runs, 69-77K GS) ──────────────────────────────
    # Best: wolf0_l=10.05 (mop=0.35, lmvg=1.5, ldn=0.015, pd=0.15).
    # Only tested mop=0.35-0.50. Lower mop not tried.
    # Strategy: try lower mop + disentangle wolf0_l's combo.
    "wolf0     wolf0_t  0.005  0.30  1.5  0.05  0.0001  0.015  0.1   0   0.15  0"     # wolf0_l but lower mop
    "wolf0     wolf0_u  0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.05  0"     # ldn=0.015 without lmvg/pd (isolate ldn)
    "wolf0     wolf0_v  0.005  0.35  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=1.5 without ldn/pd changes
    "wolf0     wolf0_w  0.005  0.30  1.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop with basic params
    "wolf0     wolf0_x  0.005  0.35  1.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # ldn=0.015 + pd=0.15 (no lmvg)
    "wolf0     wolf0_y  0.005  0.30  1.5  0.05  0.0001  0.010  0.1   0   0.05  0"     # lower mop + lmvg

    # ========================================================================
    # EXPLORATORY CONFIGS — 3 per shape (51 total)
    # Investigate parameter regions NEVER explored in R1-R11:
    #   _z:  ldist=0.10  (was always 0.05 — test 2× depth distortion)
    #   _aa: ldssim=0.2  (was always 0.1 — test 2× SSIM loss weight)
    #   _ab: lmvg=2.0    (was always ≤1.5 — test stronger multi-view geo)
    # Each config uses the shape's best-known baseline with ONE change.
    # ========================================================================

    # ── cat0 — exploratory ────────────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.015, pd=0.15, lmvg=1.0 (proven best combo)
    "cat0      cat0_z   0.005  0.40  1.0  0.10  0.0001  0.015  0.1   0   0.15  0"     # ldist=0.10 (2× depth distortion)
    "cat0      cat0_aa  0.005  0.40  1.0  0.05  0.0001  0.015  0.2   0   0.15  0"     # ldssim=0.2 (2× SSIM)
    "cat0      cat0_ab  0.005  0.40  2.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # lmvg=2.0 (stronger multi-view)

    # ── cat2 — exploratory ────────────────────────────────────────────
    # Baseline: mop=0.30, ldn=0.01, pd=0.05, lmvg=1.0 (keep pd low!)
    "cat2      cat2_z   0.005  0.30  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "cat2      cat2_aa  0.005  0.30  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "cat2      cat2_ab  0.005  0.30  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── centaur0 — exploratory ────────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.01, pd=0.05, lmvg=1.0 (sw6-like)
    "centaur0  cen0_z   0.005  0.40  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "centaur0  cen0_aa  0.005  0.40  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "centaur0  cen0_ab  0.005  0.40  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── centaur1 — exploratory ────────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.01, pd=0.05, lmvg=1.0
    "centaur1  cen1_z   0.005  0.40  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "centaur1  cen1_aa  0.005  0.40  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "centaur1  cen1_ab  0.005  0.40  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── centaur5 — exploratory ────────────────────────────────────────
    # Baseline: mop=0.45, ldn=0.01, pd=0.05, lmvg=1.0 (best g2s_max config)
    "centaur5  cen5_z   0.005  0.45  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "centaur5  cen5_aa  0.005  0.45  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "centaur5  cen5_ab  0.005  0.45  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── david0 — exploratory ─────────────────────────────────────────
    # Baseline: mop=0.50, ldn=0.01, pd=0.15, psa=5, lmvg=1.0 (dav0_s-like)
    "david0    dav0_z   0.005  0.50  1.0  0.10  0.0001  0.010  0.1   5   0.15  0"     # ldist=0.10
    "david0    dav0_aa  0.005  0.50  1.0  0.05  0.0001  0.010  0.2   5   0.15  0"     # ldssim=0.2
    "david0    dav0_ab  0.005  0.50  2.0  0.05  0.0001  0.010  0.1   5   0.15  0"     # lmvg=2.0

    # ── dog0 — exploratory ───────────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.01, pd=0.05, psa=3, lmvg=1.0 (dog0_r-like)
    "dog0      dog0_z   0.005  0.40  1.0  0.10  0.0001  0.010  0.1   3   0.05  0"     # ldist=0.10
    "dog0      dog0_aa  0.005  0.40  1.0  0.05  0.0001  0.010  0.2   3   0.05  0"     # ldssim=0.2
    "dog0      dog0_ab  0.005  0.40  2.0  0.05  0.0001  0.010  0.1   3   0.05  0"     # lmvg=2.0

    # ── gorilla5 — exploratory ───────────────────────────────────────
    # Baseline: mop=0.35, ldn=0.01, pd=0.05, lmvg=1.0 (sw6-like)
    "gorilla5  gor5_z   0.005  0.35  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "gorilla5  gor5_aa  0.005  0.35  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "gorilla5  gor5_ab  0.005  0.35  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── gorilla8 — exploratory ───────────────────────────────────────
    # Baseline: mop=0.30, ldn=0.01, pd=0.05, lmvg=1.0
    "gorilla8  gor8_z   0.005  0.30  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "gorilla8  gor8_aa  0.005  0.30  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "gorilla8  gor8_ab  0.005  0.30  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── horse0 — exploratory ─────────────────────────────────────────
    # Baseline: mop=0.25, ldn=0.01, pd=0.05, lmvg=1.0 (max GS needed)
    "horse0    hrs0_z   0.005  0.25  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "horse0    hrs0_aa  0.005  0.25  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "horse0    hrs0_ab  0.005  0.25  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── horse10 — exploratory ────────────────────────────────────────
    # Baseline: mop=0.30, ldn=0.01, pd=0.05, lmvg=1.0
    "horse10   hrs10_z  0.005  0.30  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "horse10   hrs10_aa 0.005  0.30  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "horse10   hrs10_ab 0.005  0.30  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── michael0 — exploratory ───────────────────────────────────────
    # Baseline: mop=0.35, ldn=0.01, pd=0.05, lmvg=1.0 (mic0_k-like)
    "michael0  mic0_z   0.005  0.35  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "michael0  mic0_aa  0.005  0.35  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "michael0  mic0_ab  0.005  0.35  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── michael2 — exploratory ───────────────────────────────────────
    # Baseline: mop=0.45, ldn=0.01, pd=0.05, lmvg=1.0 (mic2_o-like)
    "michael2  mic2_z   0.005  0.45  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "michael2  mic2_aa  0.005  0.45  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "michael2  mic2_ab  0.005  0.45  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── michael16 — exploratory ──────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.01, pd=0.05, lmvg=1.0 (mic16_n-like)
    "michael16 mic16_z  0.005  0.40  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "michael16 mic16_aa 0.005  0.40  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "michael16 mic16_ab 0.005  0.40  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── victoria0 — exploratory ──────────────────────────────────────
    # Baseline: mop=0.45, ldn=0.01, pd=0.05, lmvg=1.0 (vic0_o-like)
    "victoria0 vic0_z   0.005  0.45  1.0  0.10  0.0001  0.010  0.1   0   0.05  0"     # ldist=0.10
    "victoria0 vic0_aa  0.005  0.45  1.0  0.05  0.0001  0.010  0.2   0   0.05  0"     # ldssim=0.2
    "victoria0 vic0_ab  0.005  0.45  2.0  0.05  0.0001  0.010  0.1   0   0.05  0"     # lmvg=2.0

    # ── victoria2 — exploratory ──────────────────────────────────────
    # Baseline: mop=0.40, ldn=0.02, pd=0.05, lmvg=1.0 (vic2_m-like, unique ldn)
    "victoria2 vic2_z   0.005  0.40  1.0  0.10  0.0001  0.020  0.1   0   0.05  0"     # ldist=0.10
    "victoria2 vic2_aa  0.005  0.40  1.0  0.05  0.0001  0.020  0.2   0   0.05  0"     # ldssim=0.2
    "victoria2 vic2_ab  0.005  0.40  2.0  0.05  0.0001  0.020  0.1   0   0.05  0"     # lmvg=2.0

    # ── wolf0 — exploratory ──────────────────────────────────────────
    # Baseline: mop=0.35, ldn=0.015, pd=0.15, lmvg=1.5 (wolf0_l-like best)
    "wolf0     wolf0_z  0.005  0.35  1.5  0.10  0.0001  0.015  0.1   0   0.15  0"     # ldist=0.10
    "wolf0     wolf0_aa 0.005  0.35  1.5  0.05  0.0001  0.015  0.2   0   0.15  0"     # ldssim=0.2
    "wolf0     wolf0_ab 0.005  0.35  2.0  0.05  0.0001  0.015  0.1   0   0.15  0"     # lmvg=2.0
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
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) || NUM_GPUS=0
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
echo "TOSCA Benchmark Sweep R11 — ${TOTAL_CONFIGS} per-shape configs × ${TOTAL_SHAPES} shapes"
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
# Parallel job manager
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
echo "Benchmark Sweep R11 Complete"
echo "============================================================"
echo "  Total time:   ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
echo "  GPU jobs:     $CURRENT_JOB  (completed: $COMPLETED, failed: ${#FAILED_JOBS[@]})"
echo "  CPU evals:    $((CPU_COMPLETED + CPU_FAILED))  (completed: $CPU_COMPLETED, failed: $CPU_FAILED)"

if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo "Failed jobs:"
    for j in "${FAILED_JOBS[@]}"; do echo "  - $j"; done
fi
