#!/bin/bash
#
# benchmark_round3.sh — Round 3 sweep: fix bpsf=0.005 for low G2S mean(d²),
# tune 30 multi-dimensional combos of remaining 8 hyperparams for Chamfer + PSNR.
#
# Based on Round 1+2 findings:
#   - bpsf=0.01 had lowest G2S mean(d²)=0.289, push further to 0.005
#   - ldist=0.05 best on ALL metrics (Chamfer=0.118, PSNR=48.84, G2S=0.387)
#   - ldn=0.04 best Chamfer=0.106 (but higher G2S);  0.08 balanced
#   - dgt=0.0001 best G2S+PSNR; 0.0002 best Chamfer
#   - ldssim=0.1 best Chamfer+PSNR; mop=0.15 balanced; lmvg/pd/mvncc less impactful
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_round3.sh [options]
#
# OPTIONS:  (same as benchmark_params.sh)
#   --shape SHAPE           (default: cat0)
#   --texture TEXTURE       (default: blue)
#   --resolution RES        (default: high_res)
#   --light_mode MODE       (default: 0)
#   --iterations N          (default: 45000)
#   --max_parallel N        (default: 2)
#   --benchmark_output DIR  (default: output/benchmarks/tosca_params)
#   --no_wandb              Disable wandb
#   --dry_run               Print commands without executing
#   --skip_existing         Skip runs whose output already exists

set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
SHAPE="cat0"
TEXTURE="blue"
RESOLUTION="high_res"
LIGHT_MODE="0"
ITERATIONS=45000
MAX_PARALLEL=2
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
# FIXED parameter
# ============================================================================
BPSF=0.005

# ============================================================================
# 30 multi-dimensional combos
#
# Design rationale (from Round 1+2 analysis):
#   - ldist & ldn are the most impactful params → explore interactions
#   - dgt, mop, ldssim have moderate impact → include in key combos
#   - lmvg, pd, mvncc have small impact → use best-or-default values
#
# Naming: r3_NNN
# Format: mop  lmvg  ldist  dgt      ldn   ldssim  pd    mvncc
# ============================================================================
declare -a RUN_NAMES=()
declare -a RUN_ARGS=()

add_combo() {
    local name="$1" mop="$2" lmvg="$3" ldist="$4" dgt="$5" ldn="$6" ldssim="$7" pd="$8" mvncc="$9"
    RUN_NAMES+=("$name")
    RUN_ARGS+=("--min_opacity_prune $mop --lambda_multi_view_geo $lmvg --lambda_distortion $ldist --densify_grad_threshold $dgt --big_point_scale_factor $BPSF --lambda_depth_normal $ldn --lambda_dssim $ldssim --percent_dense $pd --lambda_multi_view_ncc $mvncc")
}

# ── Anchor combos: combining best individual findings ────────────────────────
#                    mop   lmvg  ldist  dgt      ldn   ldssim pd    mvncc
add_combo r3_001     0.15  0.5   0.05   0.0002   0.04  0.1    0.05  0.3    # Best-Chamfer base
add_combo r3_002     0.2   0.2   0.05   0.0001   0.08  0.2    0.1   0.6    # Best-G2S base
add_combo r3_003     0.15  0.5   0.05   0.0001   0.04  0.1    0.05  0.1    # Best-PSNR base
add_combo r3_004     0.2   0.5   0.05   0.00015  0.04  0.1    0.05  0.3    # Balanced A
add_combo r3_005     0.15  0.2   0.05   0.00015  0.06  0.15   0.1   0.6    # Balanced B

# ── Vary ldist from best-Chamfer anchor ──────────────────────────────────────
add_combo r3_006     0.15  0.5   0.02   0.0002   0.04  0.1    0.05  0.3    # ldist lower
add_combo r3_007     0.15  0.5   0.07   0.0002   0.04  0.1    0.05  0.3    # ldist=0.07 (new)
add_combo r3_008     0.15  0.5   0.1    0.0002   0.04  0.1    0.05  0.3    # ldist=0.1  (new)

# ── Vary ldn from best-Chamfer anchor ────────────────────────────────────────
add_combo r3_009     0.15  0.5   0.05   0.0002   0.06  0.1    0.05  0.3    # ldn=0.06 (new)
add_combo r3_010     0.15  0.5   0.05   0.0002   0.08  0.1    0.05  0.3    # ldn=0.08
add_combo r3_011     0.15  0.5   0.05   0.0002   0.12  0.1    0.05  0.3    # ldn=0.12 (new)

# ── Vary dgt from anchor ────────────────────────────────────────────────────
add_combo r3_012     0.15  0.5   0.05   0.0001   0.04  0.1    0.05  0.3    # dgt lower
add_combo r3_013     0.15  0.5   0.05   0.00015  0.04  0.1    0.05  0.3    # dgt mid

# ── Vary mop from anchor ────────────────────────────────────────────────────
add_combo r3_014     0.1   0.5   0.05   0.0002   0.04  0.1    0.05  0.3    # mop=0.1 (default)
add_combo r3_015     0.2   0.5   0.05   0.0002   0.04  0.1    0.05  0.3    # mop=0.2

# ── ldist × ldn interaction ──────────────────────────────────────────────────
add_combo r3_016     0.15  0.5   0.02   0.0002   0.06  0.1    0.05  0.3    # lower ldist, mid ldn
add_combo r3_017     0.15  0.5   0.07   0.0002   0.06  0.1    0.05  0.3    # higher ldist, mid ldn
add_combo r3_018     0.15  0.5   0.02   0.0002   0.08  0.1    0.05  0.3    # lower ldist, higher ldn
add_combo r3_019     0.15  0.5   0.07   0.0002   0.08  0.1    0.05  0.3    # higher ldist, higher ldn

# ── ldist × dgt interaction ──────────────────────────────────────────────────
add_combo r3_020     0.15  0.5   0.07   0.0001   0.04  0.1    0.05  0.3    # high ldist, low dgt
add_combo r3_021     0.15  0.5   0.07   0.00015  0.04  0.1    0.05  0.3    # high ldist, mid dgt

# ── ldssim from anchor ──────────────────────────────────────────────────────
add_combo r3_022     0.15  0.5   0.05   0.0002   0.04  0.15   0.05  0.3    # ldssim=0.15 (new)
add_combo r3_023     0.15  0.5   0.05   0.0002   0.04  0.2    0.05  0.3    # ldssim=0.2

# ── lmvg from anchor ────────────────────────────────────────────────────────
add_combo r3_024     0.15  0.1   0.05   0.0002   0.04  0.1    0.05  0.3    # lmvg=0.1
add_combo r3_025     0.15  0.2   0.05   0.0002   0.04  0.1    0.05  0.3    # lmvg=0.2

# ── Diverse combos: plausible strong configs ─────────────────────────────────
add_combo r3_026     0.2   0.2   0.05   0.00015  0.04  0.1    0.1   0.6    # favor G2S + Chamfer
add_combo r3_027     0.1   0.1   0.07   0.0001   0.06  0.15   0.05  0.1    # aggressive PSNR
add_combo r3_028     0.2   0.5   0.07   0.00015  0.06  0.1    0.05  0.3    # aggressive ldist + mop
add_combo r3_029     0.15  0.1   0.05   0.0001   0.06  0.2    0.1   0.3    # low dgt, mid ldn
add_combo r3_030     0.2   0.2   0.07   0.0002   0.04  0.15   0.1   0.1    # high ldist, low ldn

TOTAL_RUNS=${#RUN_NAMES[@]}

echo "============================================================"
echo "TOSCA Round 3 Benchmark — bpsf=$BPSF fixed, 30 multi-dim combos"
echo "============================================================"
echo "  Shape:             $SHAPE"
echo "  Texture:           $TEXTURE"
echo "  Resolution:        $RESOLUTION"
echo "  Light mode:        $LIGHT_MODE"
echo "  Source path:       $SOURCE_PATH"
echo "  GT mesh:           ${GT_MESH:-<not found>}"
echo "  Iterations:        $ITERATIONS"
echo "  Max parallel:      $MAX_PARALLEL"
echo "  GPUs:              $NUM_GPUS"
echo "  Benchmark output:  $BENCHMARK_DIR"
echo "  bpsf (fixed):      $BPSF"
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
        echo "  TRAIN: $PYTHON3 $TRAIN_SCRIPT -s $SOURCE_PATH -m $BENCHMARK_DIR/${RUN_NAMES[$i]} --eval --iterations $ITERATIONS ${RUN_ARGS[$i]}"
        echo ""
    done
    exit 0
fi

# ============================================================================
# Parallel execution infrastructure (same as benchmark_params.sh)
# ============================================================================
RUNNING_PIDS=()
RUNNING_NAMES=()
RUNNING_SOURCE_PATHS=()
RUNNING_MODEL_PATHS=()
RUNNING_GPU_SLOTS=()
RUNNING_EXTRA_ARGS=()
COMPLETED=0
FAILED_RUNS=()

run_eval_and_wandb() {
    local run_name="$1"
    local model_path="$2"
    local extra_args="$3"
    local gpu_slot="$4"

    local gt_arg=""
    [ -n "$GT_MESH" ] && gt_arg="--gt_mesh $GT_MESH"
    CUDA_VISIBLE_DEVICES=$gpu_slot $PYTHON3 "$EVAL_SCRIPT" \
        --output_dir "$model_path" \
        $gt_arg \
        --iteration "$ITERATIONS" 2>&1 | tee "$model_path/eval.log"

    if [ "$USE_WANDB" = true ]; then
        CUDA_VISIBLE_DEVICES=$gpu_slot $PYTHON3 -c "
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
          'sweep':'round3','bpsf_fixed':${BPSF}}
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
                        echo "  [MESH OK] $rname — evaluating..."
                        run_eval_and_wandb "$rname" "$mdl" "$eargs" "$gpu"
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
                echo "  [MESH OK] $rname — evaluating..."
                run_eval_and_wandb "$rname" "$mdl" "$eargs" "$gpu"
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
    EVAL_JOBS=()
    for i in "${!RUN_NAMES[@]}"; do
        run_name="${RUN_NAMES[$i]}"
        extra_args="${RUN_ARGS[$i]}"
        model_path="$BENCHMARK_DIR/$run_name"
        if [ -f "$model_path/recon.ply" ] && [ ! -f "$model_path/benchmark_report.json" ]; then
            EVAL_JOBS+=("$model_path")
        fi
    done
    echo "  Found ${#EVAL_JOBS[@]} runs to evaluate"
    CPU_COMPLETED=0
    CPU_FAILED=0
    for model_path in "${EVAL_JOBS[@]}"; do
        local_gt_arg=""
        [ -n "$GT_MESH" ] && local_gt_arg="--gt_mesh $GT_MESH"
        echo "  [EVAL] $(basename $model_path)..."
        if CUDA_VISIBLE_DEVICES="" $PYTHON3 "$EVAL_SCRIPT" \
                --output_dir "$model_path" $local_gt_arg \
                --iteration "$ITERATIONS" > "$model_path/eval.log" 2>&1; then
            echo "  [EVAL DONE] $(basename $model_path)"
            CPU_COMPLETED=$((CPU_COMPLETED + 1))
        else
            echo "  [EVAL FAIL] $(basename $model_path)"
            CPU_FAILED=$((CPU_FAILED + 1))
        fi
    done
    echo "CPU evals: $CPU_COMPLETED ok, $CPU_FAILED fail"
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
                echo "  [MESH OK] $run_name — evaluating..."
                run_eval_and_wandb "$run_name" "$model_path" "$extra_args" "$ASSIGNED_GPU"
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
    echo "Waiting for remaining jobs..."
    wait_for_all
fi

fi  # end of GPU phase (evaluate_only check)

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
    psnr_info = report.get('psnr', {})

    row = {
        'run_name': run_dir.name,
        'min_opacity_prune': params.get('min_opacity_prune', ''),
        'lambda_multi_view_geo': params.get('lambda_multi_view_geo', ''),
        'lambda_distortion': params.get('lambda_distortion', ''),
        'densify_grad_threshold': params.get('densify_grad_threshold', ''),
        'big_point_scale_factor': params.get('big_point_scale_factor', ''),
        'lambda_depth_normal': params.get('lambda_depth_normal', ''),
        'lambda_dssim': params.get('lambda_dssim', ''),
        'percent_dense': params.get('percent_dense', ''),
        'lambda_multi_view_ncc': params.get('lambda_multi_view_ncc', ''),
        'num_gaussians': report.get('num_gaussians', ''),
        'mesh_vertices': report.get('mesh_vertices', ''),
        'connected_components': report.get('connected_components', {}).get('num_components', ''),
        'g2s_mean_squared': g2ms.get('mean_squared', ''),
        'g2mesh_mean': g2ms.get('mean', ''),
        'g2mesh_p95': g2ms.get('p95', ''),
        'g2mesh_max': g2ms.get('max', ''),
    }
    if 'chamfer_distance' in report:
        row['chamfer_distance'] = report['chamfer_distance']
    if 'recon_to_gt_accuracy' in report:
        row['recon_to_gt_accuracy'] = report['recon_to_gt_accuracy'].get('mean', '')
    if 'gt_to_recon_completeness' in report:
        row['gt_to_recon_completeness'] = report['gt_to_recon_completeness'].get('mean', '')
    if psnr_info:
        row['test_psnr'] = psnr_info.get('test_psnr', '')
        row['train_psnr'] = psnr_info.get('train_psnr', '')
    rows.append(row)

if not rows:
    print('No benchmark results found.')
    sys.exit(0)

all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
csv_path = '${SUMMARY_CSV}'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)
print(f'Summary saved to {csv_path} ({len(rows)} runs)')
" 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "Round 3 Benchmark Complete"
echo "============================================================"
echo "  Completed: $COMPLETED / $TOTAL_RUNS"
echo "  Failed:    ${#FAILED_RUNS[@]}"
echo "  Time:      ${HOURS}h ${MINS}m"
echo "  Summary:   $SUMMARY_CSV"
echo "  Reports:   $BENCHMARK_DIR/*/benchmark_report.json"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed runs:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - $f"
    done
fi
echo ""
