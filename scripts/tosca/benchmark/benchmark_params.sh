#!/bin/bash
#
# benchmark_params.sh — Benchmark different training parameters for TOSCA Gaussian splatting.
#
# Trains + extracts mesh for multiple parameter combinations, evaluates mesh quality
# against ground-truth, and produces a CSV summary table and per-run JSON reports.
# Optionally logs results to Weights & Biases.
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_params.sh [options]
#
# OPTIONS:
#   --shape SHAPE              TOSCA shape to benchmark (default: cat0)
#   --texture TEXTURE          Texture name (default: blue)
#   --resolution RES           Resolution level (default: high_res)
#   --light_mode MODE          "decoupled_appearance" or "light_0" etc. (default: decoupled_appearance)
#   --iterations N             Training iterations (default: 30000)
#   --max_parallel N           Max parallel training jobs (default: 2)
#   --gt_data_root DIR         TOSCA processed mesh root (default: TrainData/TOSCA/processed)
#   --synth_data_base DIR      Synthetic COLMAP data root (default: TrainData/TOSCA/SyntheticColmapData)
#   --benchmark_output DIR     Where to store benchmark results (default: output/benchmarks/tosca_params)
#   --wandb_project NAME       Wandb project name (default: tosca-param-benchmark)
#   --no_wandb                 Disable wandb logging
#   --dry_run                  Print commands without executing
#   --skip_existing            Skip runs whose output already exists
#
# Benchmarked Parameters (edit arrays below):
#   - min_opacity_prune
#   - lambda_multi_view_geo
#   - lambda_distortion
#   - densify_grad_threshold
#   - big_point_scale_factor
#
# EXAMPLES:
#   # Quick benchmark on cat0
#   bash scripts/tosca/benchmark/benchmark_params.sh --shape cat0
#
#   # Dry run to see all combinations
#   bash scripts/tosca/benchmark/benchmark_params.sh --dry_run
#
#   # Multiple shapes sequentially
#   for s in cat0 dog0 horse0; do
#       bash scripts/tosca/benchmark/benchmark_params.sh --shape "$s"
#   done

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

# ============================================================================
# Parameter grids — edit these to control the sweep
# ============================================================================
MIN_OPACITY_PRUNE_VALUES=(0.1 0.15 0.2)
LAMBDA_MULTI_VIEW_GEO_VALUES=(0.1 0.2 0.5)
LAMBDA_DISTORTION_VALUES=(0.01 0.02 0.05)
DENSIFY_GRAD_THRESHOLD_VALUES=(0.0001 0.00015 0.0002 0.00025)
BIG_POINT_SCALE_FACTOR_VALUES=(0.002 0.005 0.01 0.02 0.04 0.06)
LAMBDA_DEPTH_NORMAL_VALUES=(0.04 0.08 0.16)
LAMBDA_DSSIM_VALUES=(0.1 0.2 0.4)
PERCENT_DENSE_VALUES=(0.05 0.1 0.2)
LAMBDA_MULTI_VIEW_NCC_VALUES=(0.1 0.3 0.6)

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

# Resolve python interpreter
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
PYTHON3="$CONDA_BASE/envs/geo_splat/bin/python3"
if [ ! -x "$PYTHON3" ]; then PYTHON3=python3; fi

# Source path for the dataset
# LIGHT_MODE can be "decoupled_appearance" or a numeric light ID like "0" -> "light_0"
if [[ "$LIGHT_MODE" =~ ^[0-9]+$ ]]; then
    SOURCE_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/${SHAPE}/${RESOLUTION}/light_${LIGHT_MODE}"
else
    SOURCE_PATH="$SYNTH_DATA_BASE/${TEXTURE}_texture/${SHAPE}/${RESOLUTION}/${LIGHT_MODE}"
fi

# Ground-truth mesh (first match for the resolution)
GT_MESH=$(find "$GT_DATA_ROOT/$SHAPE" -name "mesh_${RESOLUTION}_*.ply" 2>/dev/null | sort | head -1)
if [ -z "$GT_MESH" ]; then
    echo "WARNING: No ground-truth mesh found at $GT_DATA_ROOT/$SHAPE/mesh_${RESOLUTION}_*.ply"
    echo "         Evaluation will skip GT mesh comparison."
fi

# Benchmark output directory
BENCHMARK_DIR="$BENCHMARK_OUTPUT/${SHAPE}_${TEXTURE}_${RESOLUTION}"
mkdir -p "$BENCHMARK_DIR"

# Summary CSV
SUMMARY_CSV="$BENCHMARK_DIR/benchmark_summary.csv"

# Detect GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) || true
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

echo "============================================================"
echo "TOSCA Parameter Benchmark"
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
echo "  Wandb:             $USE_WANDB (project: $WANDB_PROJECT)"
echo ""
echo "  Parameter grid:"
echo "    min_opacity_prune:        ${MIN_OPACITY_PRUNE_VALUES[*]}"
echo "    lambda_multi_view_geo:    ${LAMBDA_MULTI_VIEW_GEO_VALUES[*]}"
echo "    lambda_distortion:        ${LAMBDA_DISTORTION_VALUES[*]}"
echo "    densify_grad_threshold:   ${DENSIFY_GRAD_THRESHOLD_VALUES[*]}"
echo "    big_point_scale_factor:   ${BIG_POINT_SCALE_FACTOR_VALUES[*]}"
echo "    lambda_depth_normal:      ${LAMBDA_DEPTH_NORMAL_VALUES[*]}"
echo "    lambda_dssim:             ${LAMBDA_DSSIM_VALUES[*]}"
echo "    percent_dense:            ${PERCENT_DENSE_VALUES[*]}"
echo "    lambda_multi_view_ncc:    ${LAMBDA_MULTI_VIEW_NCC_VALUES[*]}"
echo ""

# Validate source
if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: Source path not found: $SOURCE_PATH"
    exit 1
fi

# ============================================================================
# Build combinations
# Each combination varies ONE parameter from its baseline while keeping others
# at their default values. This gives a 1D sweep per parameter.
# ============================================================================
# Defaults (from arguments/__init__.py)
DEFAULT_MIN_OPACITY=0.1
DEFAULT_LAMBDA_MVG=0.1
DEFAULT_LAMBDA_DIST=0.01
DEFAULT_DENSIFY_GRAD=0.0002
DEFAULT_BIG_POINT=0.06
DEFAULT_LAMBDA_DN=0.08
DEFAULT_LAMBDA_DSSIM=0.2
DEFAULT_PERCENT_DENSE=0.1
DEFAULT_LAMBDA_MVNCC=0.3

declare -a RUN_NAMES=()
declare -a RUN_ARGS=()

# Helper: builds the full args string for a single 1D sweep point.
# Usage: build_args <param_to_vary> <value> → sets BUILT_ARGS
build_args() {
    local vary_param="$1" vary_val="$2"
    local mop=$DEFAULT_MIN_OPACITY lmvg=$DEFAULT_LAMBDA_MVG ldist=$DEFAULT_LAMBDA_DIST
    local dgt=$DEFAULT_DENSIFY_GRAD bpsf=$DEFAULT_BIG_POINT ldn=$DEFAULT_LAMBDA_DN
    local ldssim=$DEFAULT_LAMBDA_DSSIM pd=$DEFAULT_PERCENT_DENSE mvncc=$DEFAULT_LAMBDA_MVNCC
    case "$vary_param" in
        min_opacity_prune)      mop=$vary_val   ;;
        lambda_multi_view_geo)  lmvg=$vary_val  ;;
        lambda_distortion)      ldist=$vary_val ;;
        densify_grad_threshold) dgt=$vary_val   ;;
        big_point_scale_factor) bpsf=$vary_val  ;;
        lambda_depth_normal)    ldn=$vary_val   ;;
        lambda_dssim)           ldssim=$vary_val ;;
        percent_dense)          pd=$vary_val    ;;
        lambda_multi_view_ncc)  mvncc=$vary_val ;;
    esac
    BUILT_ARGS="--min_opacity_prune $mop --lambda_multi_view_geo $lmvg --lambda_distortion $ldist --densify_grad_threshold $dgt --big_point_scale_factor $bpsf --lambda_depth_normal $ldn --lambda_dssim $ldssim --percent_dense $pd --lambda_multi_view_ncc $mvncc"
}

# Sweep min_opacity_prune
for val in "${MIN_OPACITY_PRUNE_VALUES[@]}"; do
    build_args min_opacity_prune "$val"
    RUN_NAMES+=("mop_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep lambda_multi_view_geo
for val in "${LAMBDA_MULTI_VIEW_GEO_VALUES[@]}"; do
    build_args lambda_multi_view_geo "$val"
    RUN_NAMES+=("lmvg_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep lambda_distortion
for val in "${LAMBDA_DISTORTION_VALUES[@]}"; do
    build_args lambda_distortion "$val"
    RUN_NAMES+=("ldist_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep densify_grad_threshold
for val in "${DENSIFY_GRAD_THRESHOLD_VALUES[@]}"; do
    build_args densify_grad_threshold "$val"
    RUN_NAMES+=("dgt_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep big_point_scale_factor
for val in "${BIG_POINT_SCALE_FACTOR_VALUES[@]}"; do
    build_args big_point_scale_factor "$val"
    RUN_NAMES+=("bpsf_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep lambda_depth_normal
for val in "${LAMBDA_DEPTH_NORMAL_VALUES[@]}"; do
    build_args lambda_depth_normal "$val"
    RUN_NAMES+=("ldn_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep lambda_dssim
for val in "${LAMBDA_DSSIM_VALUES[@]}"; do
    build_args lambda_dssim "$val"
    RUN_NAMES+=("ldssim_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep percent_dense
for val in "${PERCENT_DENSE_VALUES[@]}"; do
    build_args percent_dense "$val"
    RUN_NAMES+=("pd_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Sweep lambda_multi_view_ncc
for val in "${LAMBDA_MULTI_VIEW_NCC_VALUES[@]}"; do
    build_args lambda_multi_view_ncc "$val"
    RUN_NAMES+=("mvncc_${val}"); RUN_ARGS+=("$BUILT_ARGS")
done

# Remove duplicate default run (the default combo appears once per sweep)
# We keep all runs — the default row will appear once per parameter but the
# output folder is unique per name so no collision occurs.

TOTAL_RUNS=${#RUN_NAMES[@]}
echo "Total benchmark runs: $TOTAL_RUNS"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "--- DRY RUN: listing all commands ---"
    for i in "${!RUN_NAMES[@]}"; do
        echo "[$((i+1))/$TOTAL_RUNS] ${RUN_NAMES[$i]}"
        echo "  TRAIN: $PYTHON3 $TRAIN_SCRIPT -s $SOURCE_PATH -m $BENCHMARK_DIR/${RUN_NAMES[$i]} --eval --iterations $ITERATIONS ${RUN_ARGS[$i]}"
        echo "  MESH:  $PYTHON3 $MESH_EXTRACT_SCRIPT -s $SOURCE_PATH -m $BENCHMARK_DIR/${RUN_NAMES[$i]} --eval"
        echo "  EVAL:  $PYTHON3 $EVAL_SCRIPT --output_dir $BENCHMARK_DIR/${RUN_NAMES[$i]} --gt_mesh $GT_MESH --iteration $ITERATIONS"
        echo ""
    done
    exit 0
fi

# ============================================================================
# Parallel execution infrastructure (adapted from train_tosca_all.sh)
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

    # Evaluate
    local gt_arg=""
    [ -n "$GT_MESH" ] && gt_arg="--gt_mesh $GT_MESH"
    CUDA_VISIBLE_DEVICES=$gpu_slot $PYTHON3 "$EVAL_SCRIPT" \
        --output_dir "$model_path" \
        $gt_arg \
        --iteration "$ITERATIONS" 2>&1 | tee "$model_path/eval.log"

    # Log to wandb
    if [ "$USE_WANDB" = true ]; then
        CUDA_VISIBLE_DEVICES=$gpu_slot $PYTHON3 -c "
import json, sys
try:
    import wandb
except ImportError:
    print('WARNING: wandb not installed, skipping logging')
    sys.exit(0)

report_path = '${model_path}/benchmark_report.json'
try:
    with open(report_path) as f:
        report = json.load(f)
except FileNotFoundError:
    print(f'WARNING: {report_path} not found, skipping wandb')
    sys.exit(0)

# Parse the extra args into a config dict
config = {
    'shape': '${SHAPE}',
    'texture': '${TEXTURE}',
    'resolution': '${RESOLUTION}',
    'light_mode': '${LIGHT_MODE}',
    'iterations': ${ITERATIONS},
    'run_name': '${run_name}',
}
# Parse key=value from extra_args
import shlex
args_list = shlex.split('${extra_args}')
i = 0
while i < len(args_list):
    if args_list[i].startswith('--'):
        key = args_list[i][2:]
        if i + 1 < len(args_list) and not args_list[i+1].startswith('--'):
            try:
                config[key] = float(args_list[i+1])
            except ValueError:
                config[key] = args_list[i+1]
            i += 2
        else:
            config[key] = True
            i += 1
    else:
        i += 1

# Metrics to log
metrics = {}
metrics['num_gaussians'] = report.get('num_gaussians', 0)
metrics['mesh_vertices'] = report.get('mesh_vertices', 0)
metrics['connected_components'] = report.get('connected_components', {}).get('num_components', 0)

# Gaussian -> recon mesh surface
g2ms = report.get('gaussian_to_recon_mesh_surface', {})
metrics['g2mesh_mean'] = g2ms.get('mean', 0)
metrics['g2mesh_p95'] = g2ms.get('p95', 0)
metrics['g2mesh_max'] = g2ms.get('max', 0)

# GT mesh metrics
if 'chamfer_distance' in report:
    metrics['chamfer_distance'] = report['chamfer_distance']
    metrics['chamfer_pct_bbox'] = report.get('chamfer_pct_bbox', 0)

r2gt = report.get('recon_to_gt_accuracy', {})
if r2gt:
    metrics['recon_to_gt_accuracy_mean'] = r2gt.get('mean', 0)

gt2r = report.get('gt_to_recon_completeness', {})
if gt2r:
    metrics['gt_to_recon_completeness_mean'] = gt2r.get('mean', 0)

run = wandb.init(
    project='${WANDB_PROJECT}',
    name='${SHAPE}_${run_name}',
    config=config,
    reinit=True,
)
wandb.log(metrics)
wandb.finish()
print(f'Logged to wandb: ${WANDB_PROJECT}/${SHAPE}_${run_name}')
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

    # Skip if already completed
    if [ "$SKIP_EXISTING" = true ] && [ -f "$model_path/benchmark_report.json" ]; then
        echo "  Skipping: benchmark_report.json already exists"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    mkdir -p "$model_path"

    # Save run config
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
        # Sequential mode
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

# Wait for remaining parallel jobs
if [ "$MAX_PARALLEL" -gt 1 ]; then
    echo "Waiting for remaining jobs..."
    wait_for_all
fi

# ============================================================================
# Generate summary CSV from all benchmark_report.json files
# ============================================================================
echo ""
echo "============================================================"
echo "Generating summary table..."
echo "============================================================"

$PYTHON3 -c "
import json, csv, sys
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

    # Parse benchmark args
    params = {}
    if args_path.exists():
        import shlex
        tokens = shlex.split(args_path.read_text().strip())
        i = 0
        while i < len(tokens):
            if tokens[i].startswith('--') and i + 1 < len(tokens):
                try:
                    params[tokens[i][2:]] = float(tokens[i+1])
                except ValueError:
                    params[tokens[i][2:]] = tokens[i+1]
                i += 2
            else:
                i += 1

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
        'g2mesh_mean': report.get('gaussian_to_recon_mesh_surface', {}).get('mean', ''),
        'g2mesh_p95': report.get('gaussian_to_recon_mesh_surface', {}).get('p95', ''),
        'g2mesh_max': report.get('gaussian_to_recon_mesh_surface', {}).get('max', ''),
    }
    # GT metrics
    if 'chamfer_distance' in report:
        row['chamfer_distance'] = report['chamfer_distance']
        row['chamfer_pct_bbox'] = report.get('chamfer_pct_bbox', '')
    if 'recon_to_gt_accuracy' in report:
        row['recon_to_gt_accuracy'] = report['recon_to_gt_accuracy'].get('mean', '')
    if 'gt_to_recon_completeness' in report:
        row['gt_to_recon_completeness'] = report['gt_to_recon_completeness'].get('mean', '')
    # PSNR
    psnr_info = report.get('psnr', {})
    if psnr_info:
        row['test_psnr'] = psnr_info.get('test_psnr', '')
        row['train_psnr'] = psnr_info.get('train_psnr', '')

    rows.append(row)

if not rows:
    print('No benchmark results found.')
    sys.exit(0)

# Collect all keys
all_keys = list(dict.fromkeys(k for row in rows for k in row.keys()))

csv_path = '${SUMMARY_CSV}'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(rows)

print(f'Summary saved to {csv_path} ({len(rows)} runs)')

# Print table
print()
header = '  '.join(f'{k:>25s}' for k in all_keys)
print(header)
print('-' * len(header))
for row in rows:
    vals = []
    for k in all_keys:
        v = row.get(k, '')
        if isinstance(v, float):
            vals.append(f'{v:>25.6f}')
        else:
            vals.append(f'{str(v):>25s}')
    print('  '.join(vals))
" 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo "Benchmark Complete"
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
