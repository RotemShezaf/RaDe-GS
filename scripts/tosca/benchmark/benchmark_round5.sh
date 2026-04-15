#!/bin/bash
#
# benchmark_round5.sh — Round 5: Minimize g2s_max while preserving chamfer, g2s_mean, PSNR
#
# R4 best results (41 of 60 completed):
#   Chamfer:  r4_d04  = 0.1127  (ldn=0.02, 94K gauss)
#   g2s_ms:   r4_d09  = 0.1652  (mop=0.35, 77K gauss)
#   g2s_max:  r4_f06  = 7.57    (pd=0.1, 96K gauss)
#   PSNR:     r4_a09  = 48.73   (bpsf=0.008, psa=8)
#
# R4 key findings — which parameters reduce g2s_max:
#   pd ∈ {0.05-0.1}  → g2s_max 7.6-8.5  (vs 9-14 without). STRONGEST effect.
#                       Mechanism: pd>0 enables cloning (keeps Gs near parent)
#                       instead of always splitting (children can land far away).
#   lmvg ∈ {1.0-2.0} → g2s_max 8.6-9.1  (vs 10+ at 0.5). Multi-view consistency
#                       loss pushes Gs toward surface geometry.
#   mop ∈ {0.25-0.35} → g2s_ms 0.17 (best), but g2s_max actually HIGHER (10.4)
#                       because mop removes weak Gs near surface, not far outliers.
#   bpsf = 0.008      → g2s_max 9.2 (vs 10+ at 0.005). Wider scale prune.
#
# CRITICAL GAP: R4 never combined pd>0 with lmvg>0.5 or mop>0.15.
# These are the top g2s_max reducers (pd, lmvg) and both work via different
# mechanisms. Combining them should be additive.
#
# Strategy — 20 configs in 4 blocks:
#   A (6): pd × lmvg — the two strongest g2s_max reducers, never combined
#   B (5): pd × mop — pruning + clone bias, test synergy
#   C (5): Triple combo: pd + lmvg + (mop or psa) + ldn=0.02 for chamfer
#   D (4): bpsf=0.008 + best combos — wider scale threshold
#
# Gaussian budget: up to ~140K is acceptable (R4 best had 77-96K).
#
# Constraints (learned from R4):
#   - bpsf ≤ 0.004 → overflow risk (17 of 19 R4 failures)
#   - ldist ≥ 0.07 → hurts chamfer (0.14 at 0.07, 0.52 at 0.15)
#   - psa ≥ 15 → over-prunes, g2s_ms doubles
#   - pd=0.2 → untested, explore carefully
#
# Shared constants (proven R3/R4 winners):
#   ldssim=0.1, dgt=0.0001, ldist=0.05, mvncc=0.3
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_round5.sh [options]

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
SHAPE="cat0"
TEXTURE="blue"
RESOLUTION="high_res"
LIGHT_MODE="0"
ITERATIONS=45000
MAX_PARALLEL=4
MAX_PARALLEL_CPU=60
GT_DATA_ROOT="TrainData/TOSCA/processed"
SYNTH_DATA_BASE="TrainData/TOSCA/SyntheticColmapData"
BENCHMARK_OUTPUT="output/benchmarks/tosca_params"
WANDB_PROJECT="tosca-param-benchmark"
USE_WANDB=true
DRY_RUN=false
SKIP_EXISTING=false
EVALUATE_ONLY=false

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --shape)                SHAPE="$2";              shift 2 ;;
        --texture)              TEXTURE="$2";            shift 2 ;;
        --resolution)           RESOLUTION="$2";         shift 2 ;;
        --light_mode)           LIGHT_MODE="$2";         shift 2 ;;
        --iterations)           ITERATIONS="$2";         shift 2 ;;
        --max_parallel)         MAX_PARALLEL="$2";       shift 2 ;;
        --max_parallel_cpu)     MAX_PARALLEL_CPU="$2";   shift 2 ;;
        --gt_data_root)         GT_DATA_ROOT="$2";       shift 2 ;;
        --synth_data_base)      SYNTH_DATA_BASE="$2";    shift 2 ;;
        --benchmark_output)     BENCHMARK_OUTPUT="$2";   shift 2 ;;
        --wandb_project)        WANDB_PROJECT="$2";      shift 2 ;;
        --no_wandb)             USE_WANDB=false;         shift   ;;
        --dry_run)              DRY_RUN=true;            shift   ;;
        --skip_existing)        SKIP_EXISTING=true;      shift   ;;
        --evaluate_only)        EVALUATE_ONLY=true;      shift   ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
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

if [[ "$LIGHT_MODE" =~ ^[0-9]+$ ]]; then
    SOURCE_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/${SHAPE}/${RESOLUTION}/light_${LIGHT_MODE}"
else
    SOURCE_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/${SHAPE}/${RESOLUTION}/${LIGHT_MODE}"
fi

GT_MESH=$(find "$GT_DATA_ROOT/$SHAPE" -name "mesh_${RESOLUTION}_*.ply" 2>/dev/null | sort | head -1)
if [ -z "$GT_MESH" ]; then
    echo "WARNING: No ground-truth mesh found at $GT_DATA_ROOT/$SHAPE/mesh_${RESOLUTION}_*.ply"
fi

BENCHMARK_DIR="$BENCHMARK_OUTPUT/${SHAPE}_${TEXTURE}_${RESOLUTION}"
mkdir -p "$BENCHMARK_DIR"
SUMMARY_CSV="$BENCHMARK_DIR/benchmark_summary.csv"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) || true
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

# ============================================================================
# 20 configs organized in 4 blocks
#
# Shared constants (proven R3/R4 winners):
#   ldssim = 0.1,  dgt = 0.0001,  ldist = 0.05,  mvncc = 0.3
#
# Format:
#   add_combo name bpsf mop lmvg ldist dgt ldn ldssim psa pmst pd mvncc
# ============================================================================
declare -a RUN_NAMES=()
declare -a RUN_ARGS=()

add_combo() {
    local name="$1" bpsf="$2" mop="$3" lmvg="$4" ldist="$5" dgt="$6" ldn="$7" ldssim="$8" psa="$9" pmst="${10}" pd="${11}" mvncc="${12}"
    RUN_NAMES+=("$name")
    local args="--min_opacity_prune $mop --lambda_multi_view_geo $lmvg --lambda_distortion $ldist --densify_grad_threshold $dgt --big_point_scale_factor $bpsf --lambda_depth_normal $ldn --lambda_dssim $ldssim --percent_dense $pd --lambda_multi_view_ncc $mvncc"
    # Only add pruning params when non-zero (they default to 0.0 = disabled)
    if [ "$psa" != "0" ]; then
        args="$args --prune_scale_anisotropy $psa"
    fi
    if [ "$pmst" != "0" ]; then
        args="$args --prune_min_scale_threshold $pmst"
    fi
    RUN_ARGS+=("$args")
}

# ════════════════════════════════════════════════════════════════════════════
# BLOCK A (6 configs): pd × lmvg — the two strongest g2s_max reducers
#
# R4: pd=0.1 → g2s_max=7.57 (best).  lmvg=1.0 → g2s_max=8.55 (2nd best).
# NEVER tested together.  Both reduce g2s_max via different mechanisms:
#   pd>0  → enables cloning (keeps Gaussians near parent vs splitting far)
#   lmvg  → multi-view consistency loss pushes Gs toward surface geometry
# Also test lmvg=3.0 and pd=0.2 (both beyond R4 range).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd     mvncc

add_combo r5_a01  0.005  0.15  1.0   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_a02  0.005  0.15  2.0   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_a03  0.005  0.15  1.0   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r5_a04  0.005  0.15  2.0   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r5_a05  0.005  0.15  3.0   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_a06  0.005  0.15  1.0   0.05   0.0001  0.04  0.1    0   0    0.2    0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK B (5 configs): pd × mop — pruning + clone bias
#
# mop=0.25/0.35 improved g2s_ms in R4 but not g2s_max (far outliers have
# high opacity ~0.79, so mop doesn't catch them).  However, pd reduces the
# outlier creation pathway.  Together: mop cleans weak Gs, pd prevents far ones.
# Also test pd=0.15 (beyond R4 max of 0.1).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd     mvncc

add_combo r5_b01  0.005  0.25  0.5   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_b02  0.005  0.25  0.5   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r5_b03  0.005  0.35  0.5   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_b04  0.005  0.35  0.5   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r5_b05  0.005  0.25  0.5   0.05   0.0001  0.04  0.1    0   0    0.15   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK C (5 configs): Triple combo — pd + lmvg + (mop or psa) + low ldn
#
# Combine the best mechanisms: pd (prevent far Gs), lmvg (push to surface),
# mop/psa (prune outliers), ldn=0.02 (best chamfer from R4).
# These target the overall optimum across all metrics.
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd     mvncc

add_combo r5_c01  0.005  0.25  1.0   0.05   0.0001  0.02  0.1    0   0    0.05   0.3
add_combo r5_c02  0.005  0.25  1.0   0.05   0.0001  0.02  0.1    0   0    0.1    0.3
add_combo r5_c03  0.005  0.15  2.0   0.05   0.0001  0.02  0.1    0   0    0.1    0.3
add_combo r5_c04  0.005  0.25  1.0   0.05   0.0001  0.02  0.1    5   0    0.05   0.3
add_combo r5_c05  0.005  0.25  2.0   0.05   0.0001  0.02  0.1    0   0    0.05   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK D (4 configs): bpsf=0.008 + best combos
#
# bpsf=0.008 → wider scale prune threshold (2.37 vs 1.48 at 0.005).
# R4: g2s_ms=0.197, g2s_max=9.2 — better than 0.005 baseline.
# Never combined with pd or lmvg>0.5.  Test the key interactions.
# dgt stays 0.0001 (no compensation needed — 0.008 is safe).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd     mvncc

add_combo r5_d01  0.008  0.25  0.5   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r5_d02  0.008  0.25  1.0   0.05   0.0001  0.02  0.1    0   0    0.05   0.3
add_combo r5_d03  0.008  0.15  1.0   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r5_d04  0.008  0.25  1.0   0.05   0.0001  0.02  0.1    0   0    0.1    0.3

TOTAL_RUNS=${#RUN_NAMES[@]}

echo "============================================================"
echo "TOSCA Round 5 Benchmark — minimize g2s_max + chamfer, 20 configs"
echo "============================================================"
echo "  Shape:             $SHAPE"
echo "  Texture:           $TEXTURE"
echo "  Resolution:        $RESOLUTION"
echo "  Light mode:        $LIGHT_MODE"
echo "  Source path:       $SOURCE_PATH"
echo "  GT mesh:           ${GT_MESH:-<not found>}"
echo "  Iterations:        $ITERATIONS"
echo "  Max parallel GPU:  $MAX_PARALLEL"
echo "  Max parallel CPU:  $MAX_PARALLEL_CPU"
echo "  GPUs:              $NUM_GPUS"
echo "  Benchmark output:  $BENCHMARK_DIR"
echo "  Total runs:        $TOTAL_RUNS"
echo ""

if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: Source path not found: $SOURCE_PATH"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN: listing all commands ---"
    for i in "${!RUN_NAMES[@]}"; do
        echo "[$((i+1))/$TOTAL_RUNS] ${RUN_NAMES[$i]}"
        echo "  ARGS: ${RUN_ARGS[$i]}"
        echo ""
    done
    exit 0
fi

# ============================================================================
# Parallel execution infrastructure
# ============================================================================
RUNNING_PIDS=()
RUNNING_NAMES=()
RUNNING_SOURCE_PATHS=()
RUNNING_MODEL_PATHS=()
RUNNING_GPU_SLOTS=()
RUNNING_EXTRA_ARGS=()
COMPLETED=0
FAILED_RUNS=()

# Pending CPU evaluations (populated as GPU jobs complete)
PENDING_EVAL_NAMES=()
PENDING_EVAL_MODEL_PATHS=()
PENDING_EVAL_EXTRA_ARGS=()

run_eval_and_wandb() {
    local run_name="$1"
    local model_path="$2"
    local extra_args="$3"

    local gt_arg=""
    [ -n "$GT_MESH" ] && gt_arg="--gt_mesh $GT_MESH"
    CUDA_VISIBLE_DEVICES="" $PYTHON3 "$EVAL_SCRIPT" \
        --output_dir "$model_path" \
        $gt_arg \
        --iteration "$ITERATIONS" 2>&1 | tee "$model_path/eval.log"

    if [ "$USE_WANDB" = true ]; then
        CUDA_VISIBLE_DEVICES="" $PYTHON3 -c "
import json, sys, shlex
try:
    import wandb
except ImportError:
    sys.exit(0)
report_path = '${model_path}/benchmark_report.json'
try:
    with open(report_path) as f:
        report = json.load(f)
except FileNotFoundError:
    sys.exit(0)
config = {'shape':'${SHAPE}','texture':'${TEXTURE}','resolution':'${RESOLUTION}',
          'light_mode':'${LIGHT_MODE}','iterations':${ITERATIONS},'run_name':'${run_name}',
          'sweep':'round5'}
args_list = shlex.split('${extra_args}')
i = 0
while i < len(args_list):
    if args_list[i].startswith('--') and i+1 < len(args_list):
        try: config[args_list[i][2:]] = float(args_list[i+1])
        except ValueError: config[args_list[i][2:]] = args_list[i+1]
        i += 2
    else: i += 1
metrics = {}
metrics['num_gaussians'] = report.get('num_gaussians', 0)
metrics['mesh_vertices'] = report.get('mesh_vertices', 0)
g2ms = report.get('gaussian_to_recon_mesh_surface', {})
metrics['g2mesh_mean'] = g2ms.get('mean', 0)
metrics['g2mesh_p95'] = g2ms.get('p95', 0)
metrics['g2mesh_max'] = g2ms.get('max', 0)
metrics['g2s_mean_squared'] = g2ms.get('mean_squared', 0)
g2gt = report.get('gaussian_to_gt_surface', {})
metrics['g2gt_mean_squared'] = g2gt.get('mean_squared', 0)
metrics['g2gt_max'] = g2gt.get('max', 0)
if 'chamfer_distance' in report:
    metrics['chamfer_distance'] = report['chamfer_distance']
psnr = report.get('psnr', {})
if psnr:
    metrics['test_psnr'] = psnr.get('test_psnr', 0)
r2gt = report.get('recon_to_gt_accuracy', {})
if r2gt: metrics['recon_to_gt_accuracy_mean'] = r2gt.get('mean', 0)
gt2r = report.get('gt_to_recon_completeness', {})
if gt2r: metrics['gt_to_recon_completeness_mean'] = gt2r.get('mean', 0)
run = wandb.init(project='${WANDB_PROJECT}', name='${SHAPE}_${run_name}', config=config, reinit=True)
wandb.log(metrics)
wandb.finish()
" 2>&1 || echo "  WARNING: wandb logging failed for $run_name"
    fi
}

wait_for_slot() {
    while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; do
        local NEW_PIDS=() NEW_NAMES=() NEW_SRC=() NEW_MDL=() NEW_GPU=() NEW_ARGS=()
        for i in "${!RUNNING_PIDS[@]}"; do
            local pid="${RUNNING_PIDS[$i]}"
            local rname="${RUNNING_NAMES[$i]}"
            local src="${RUNNING_SOURCE_PATHS[$i]}"
            local mdl="${RUNNING_MODEL_PATHS[$i]}"
            local gpu="${RUNNING_GPU_SLOTS[$i]}"
            local eargs="${RUNNING_EXTRA_ARGS[$i]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid"); NEW_NAMES+=("$rname"); NEW_SRC+=("$src")
                NEW_MDL+=("$mdl"); NEW_GPU+=("$gpu"); NEW_ARGS+=("$eargs")
            else
                if wait "$pid"; then
                    echo "  [TRAIN OK] $rname — extracting mesh..."
                    if CUDA_VISIBLE_DEVICES=$gpu $PYTHON3 "$MESH_EXTRACT_SCRIPT" \
                            -s "$src" -m "$mdl" --eval > "$mdl/mesh_extract.log" 2>&1; then
                        echo "  [MESH OK] $rname — queued for CPU eval"
                        PENDING_EVAL_NAMES+=("$rname")
                        PENDING_EVAL_MODEL_PATHS+=("$mdl")
                        PENDING_EVAL_EXTRA_ARGS+=("$eargs")
                        COMPLETED=$((COMPLETED + 1))
                    else
                        echo "  [MESH FAIL] $rname"
                        FAILED_RUNS+=("$rname (mesh extraction)")
                    fi
                else
                    echo "  [TRAIN FAIL] $rname"
                    FAILED_RUNS+=("$rname (training)")
                fi
            fi
        done
        RUNNING_PIDS=("${NEW_PIDS[@]}")
        RUNNING_NAMES=("${NEW_NAMES[@]}")
        RUNNING_SOURCE_PATHS=("${NEW_SRC[@]}")
        RUNNING_MODEL_PATHS=("${NEW_MDL[@]}")
        RUNNING_GPU_SLOTS=("${NEW_GPU[@]}")
        RUNNING_EXTRA_ARGS=("${NEW_ARGS[@]}")
        if [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; then
            sleep 10
        fi
    done
}

wait_for_all() {
    for i in "${!RUNNING_PIDS[@]}"; do
        local pid="${RUNNING_PIDS[$i]}"
        local rname="${RUNNING_NAMES[$i]}"
        local src="${RUNNING_SOURCE_PATHS[$i]}"
        local mdl="${RUNNING_MODEL_PATHS[$i]}"
        local gpu="${RUNNING_GPU_SLOTS[$i]}"
        local eargs="${RUNNING_EXTRA_ARGS[$i]}"
        if wait "$pid"; then
            echo "  [TRAIN OK] $rname — extracting mesh..."
            if CUDA_VISIBLE_DEVICES=$gpu $PYTHON3 "$MESH_EXTRACT_SCRIPT" \
                    -s "$src" -m "$mdl" --eval > "$mdl/mesh_extract.log" 2>&1; then
                echo "  [MESH OK] $rname — queued for CPU eval"
                PENDING_EVAL_NAMES+=("$rname")
                PENDING_EVAL_MODEL_PATHS+=("$mdl")
                PENDING_EVAL_EXTRA_ARGS+=("$eargs")
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  [MESH FAIL] $rname"
                FAILED_RUNS+=("$rname (mesh extraction)")
            fi
        else
            echo "  [TRAIN FAIL] $rname"
            FAILED_RUNS+=("$rname (training)")
        fi
    done
    RUNNING_PIDS=()
    RUNNING_NAMES=()
    RUNNING_SOURCE_PATHS=()
    RUNNING_MODEL_PATHS=()
    RUNNING_GPU_SLOTS=()
    RUNNING_EXTRA_ARGS=()
}

# ============================================================================
# Main loop
# ============================================================================
START_TIME=$(date +%s)
GPU_SLOT=0

if [ "$EVALUATE_ONLY" = true ]; then
    echo "============================================================"
    echo "EVALUATE-ONLY MODE: Scanning for runs needing evaluation..."
    echo "============================================================"
    for i in "${!RUN_NAMES[@]}"; do
        run_name="${RUN_NAMES[$i]}"
        extra_args="${RUN_ARGS[$i]}"
        model_path="$BENCHMARK_DIR/$run_name"
        if [ -f "$model_path/recon.ply" ] && [ ! -f "$model_path/benchmark_report.json" ]; then
            PENDING_EVAL_NAMES+=("$run_name")
            PENDING_EVAL_MODEL_PATHS+=("$model_path")
            PENDING_EVAL_EXTRA_ARGS+=("$extra_args")
        fi
    done
    echo "  Found ${#PENDING_EVAL_NAMES[@]} runs to evaluate"
else
# --- Begin GPU phase ---

for i in "${!RUN_NAMES[@]}"; do
    run_name="${RUN_NAMES[$i]}"
    extra_args="${RUN_ARGS[$i]}"
    model_path="$BENCHMARK_DIR/$run_name"
    job_num=$((i + 1))

    echo "============================================================"
    echo "[$job_num/$TOTAL_RUNS] $run_name"
    echo "  Model path: $model_path"
    echo "  Extra args: $extra_args"
    echo "============================================================"

    if [ "$SKIP_EXISTING" = true ] && [ -f "$model_path/benchmark_report.json" ]; then
        echo "  Skipping: benchmark_report.json already exists"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    mkdir -p "$model_path"
    echo "$extra_args" > "$model_path/benchmark_args.txt"

    ASSIGNED_GPU=$((GPU_SLOT % NUM_GPUS))
    GPU_SLOT=$((GPU_SLOT + 1))

    if [ "$MAX_PARALLEL" -gt 1 ]; then
        wait_for_slot
        CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU $PYTHON3 "$TRAIN_SCRIPT" \
            -s "$SOURCE_PATH" -m "$model_path" --eval \
            --iterations "$ITERATIONS" $extra_args \
            > "$model_path/train.log" 2>&1 &

        RUNNING_PIDS+=($!)
        RUNNING_NAMES+=("$run_name")
        RUNNING_SOURCE_PATHS+=("$SOURCE_PATH")
        RUNNING_MODEL_PATHS+=("$model_path")
        RUNNING_GPU_SLOTS+=("$ASSIGNED_GPU")
        RUNNING_EXTRA_ARGS+=("$extra_args")
        echo "  Started training (PID $!, GPU $ASSIGNED_GPU)"
    else
        echo "  Training (GPU $ASSIGNED_GPU)..."
        if CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU $PYTHON3 "$TRAIN_SCRIPT" \
                -s "$SOURCE_PATH" -m "$model_path" --eval \
                --iterations "$ITERATIONS" $extra_args \
                > "$model_path/train.log" 2>&1; then
            echo "  [TRAIN OK] $run_name — extracting mesh..."
            if CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU $PYTHON3 "$MESH_EXTRACT_SCRIPT" \
                    -s "$SOURCE_PATH" -m "$model_path" --eval \
                    > "$model_path/mesh_extract.log" 2>&1; then
                echo "  [MESH OK] $run_name — queued for CPU eval"
                PENDING_EVAL_NAMES+=("$run_name")
                PENDING_EVAL_MODEL_PATHS+=("$model_path")
                PENDING_EVAL_EXTRA_ARGS+=("$extra_args")
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  [MESH FAIL] $run_name"
                FAILED_RUNS+=("$run_name (mesh extraction)")
            fi
        else
            echo "  [TRAIN FAIL] $run_name"
            FAILED_RUNS+=("$run_name (training)")
        fi
    fi
    echo ""
done

if [ "$MAX_PARALLEL" -gt 1 ]; then
    echo "Waiting for remaining GPU jobs..."
    wait_for_all
fi

fi  # end of GPU phase (evaluate_only check)

# ============================================================================
# Phase 2: CPU-only evaluation (no GPU required)
# ============================================================================
echo ""
echo "============================================================"
echo "Phase 2: CPU Evaluation — ${#PENDING_EVAL_NAMES[@]} jobs, max $MAX_PARALLEL_CPU parallel"
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

for i in "${!PENDING_EVAL_NAMES[@]}"; do
    eval_name="${PENDING_EVAL_NAMES[$i]}"
    eval_model="${PENDING_EVAL_MODEL_PATHS[$i]}"
    eval_args="${PENDING_EVAL_EXTRA_ARGS[$i]}"

    _wait_for_cpu_slot

    (
        run_eval_and_wandb "$eval_name" "$eval_model" "$eval_args"
    ) &
    EVAL_RUNNING_PIDS+=($!)
    EVAL_RUNNING_NAMES+=("$eval_name")
    echo "  [EVAL] Launched: $eval_name (PID $!)"
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
# Generate summary CSV
# ============================================================================
echo ""
echo "============================================================"
echo "Generating summary table..."
echo "============================================================"

$PYTHON3 -c "
import json, csv, sys, shlex
from pathlib import Path

benchmark_dir = Path('${BENCHMARK_DIR}')
rows = []

for run_dir in sorted(benchmark_dir.iterdir()):
    if not run_dir.name.startswith('r5_'):
        continue
    report_path = run_dir / 'benchmark_report.json'
    args_path = run_dir / 'benchmark_args.txt'
    if not report_path.exists():
        continue

    with open(report_path) as f:
        report = json.load(f)

    params = {}
    if args_path.exists():
        tokens = shlex.split(args_path.read_text().strip())
        i = 0
        while i < len(tokens):
            if tokens[i].startswith('--') and i + 1 < len(tokens):
                try: params[tokens[i][2:]] = float(tokens[i+1])
                except ValueError: params[tokens[i][2:]] = tokens[i+1]
                i += 2
            else: i += 1

    g2ms = report.get('gaussian_to_recon_mesh_surface', {})
    g2gt = report.get('gaussian_to_gt_surface', {})
    psnr_info = report.get('psnr', {})

    row = {
        'run_name': run_dir.name,
        'big_point_scale_factor': params.get('big_point_scale_factor', ''),
        'min_opacity_prune': params.get('min_opacity_prune', ''),
        'percent_dense': params.get('percent_dense', ''),
        'lambda_depth_normal': params.get('lambda_depth_normal', ''),
        'prune_scale_anisotropy': params.get('prune_scale_anisotropy', ''),
        'num_gaussians': report.get('num_gaussians', ''),
        'chamfer_distance': report.get('chamfer_distance', ''),
        'g2gt_mean_squared': g2gt.get('mean_squared', ''),
        'g2gt_max': g2gt.get('max', ''),
        'g2gt_mean': g2gt.get('mean', ''),
        'g2gt_median': g2gt.get('median', ''),
        'test_psnr': psnr_info.get('test_psnr', ''),
        'recon_to_gt_accuracy': report.get('recon_to_gt_accuracy', {}).get('mean', ''),
        'gt_to_recon_completeness': report.get('gt_to_recon_completeness', {}).get('mean', ''),
        'connected_components': report.get('connected_components', {}).get('num_components', ''),
    }
    rows.append(row)

if not rows:
    print('No R5 benchmark results found.')
    sys.exit(0)

all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
csv_path = '${BENCHMARK_DIR}/r5_summary.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)
print(f'R5 summary saved to {csv_path} ({len(rows)} runs)')
" 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "Round 5 Benchmark Complete"
echo "============================================================"
echo "  GPU completed: $COMPLETED / $TOTAL_RUNS"
echo "  GPU failed:    ${#FAILED_RUNS[@]}"
echo "  CPU evals:     $((CPU_COMPLETED + CPU_FAILED))  (ok: $CPU_COMPLETED, fail: $CPU_FAILED)"
echo "  Time:          ${HOURS}h ${MINS}m"
echo "  Summary:       ${BENCHMARK_DIR}/r5_summary.csv"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed runs:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - $f"
    done
fi
echo ""
