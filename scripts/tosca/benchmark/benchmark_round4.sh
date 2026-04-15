#!/bin/bash
#
# benchmark_round4.sh — Round 4: Reduce g2s_max while preserving R3 advantages
#
# R3 best (r3_003): Chamfer=0.106, PSNR=48.6, g2s_msq=0.116, g2s_max=9.83
#
# Root cause of high g2s_max:
#   Scene extent = 296.82.  With bpsf=0.005 the prune threshold = 1.484.
#   But far Gaussians (d>1 from mesh) have max_scale ~0.88 → they SLIP THROUGH.
#   These are small, somewhat round floaters (anisotropy ~4x) with high opacity (0.79).
#   Close Gaussians are thin surface discs (anisotropy ~52x, min_scale 0.02).
#
# Strategy:
#   A) prune_scale_anisotropy (psa): prune round floaters (ratio<thresh), never tested before
#   B) Ultra-low bpsf (0.002-0.004): tighten scale threshold to catch floaters
#   C) prune_min_scale_threshold (pmst): prune fat Gaussians (min_scale>thresh)
#   D) Soft regularization: higher ldist/lmvg + lower ldn push Gs to surface
#   E) Joint: combine best strategies
#
# All configs share proven winners from R3:
#   ldssim=0.1, dgt=0.0001, mvncc=0.3
#
# 50 total configs.
#
# USAGE:
#   bash scripts/tosca/benchmark/benchmark_round4.sh [options]

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
# 60 configs organized in 6 blocks
#
# Shared constants (proven R3 winners):
#   ldssim = 0.1
#   dgt    = 0.0001
#   mvncc  = 0.3
#
# Format:
#   add_combo name bpsf mop lmvg ldist dgt ldn ldssim psa pmst pd mvncc
#
# NOTE: percent_dense was BUGGED before Round 4 (never assigned to GaussianModel).
#       Now fixed in train.py.  Default=0.1, original dev intended 0.01 (.bak).
#       pd controls split/clone threshold: max_scale > pd*extent → split, else clone.
#       With extent=296.82: pd=0.1→29.7 (always clone), pd=0.01→2.97 (mostly clone),
#       pd=0.001→0.30 (mostly split). Old runs had effective pd=0.0 → always split.
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
# BLOCK A (12 configs): prune_scale_anisotropy — targeted floater removal
#
# Far Gaussians: anisotropy ratio ~4.  Close Gaussians: ratio ~52.
# psa=5 → prune if ratio<5 (catches most floaters, keeps surface discs)
# psa=8 → moderate, psa=15 → conservative (only catches very round ones)
# Cross with bpsf ∈ {0.003, 0.005, 0.008} to see interaction.
# Low bpsf uses higher dgt to prevent densify-prune runaway:
#   bpsf=0.003 → dgt=0.0003 (3× base), bpsf=0.005/0.008 → dgt=0.0001 (1×)
# All use pd=0.0 (same as previous rounds — always split, no clone).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd    mvncc

add_combo r4_a01  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    5   0    0.0   0.3
add_combo r4_a02  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    8   0    0.0   0.3
add_combo r4_a03  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    15  0    0.0   0.3
add_combo r4_a04  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    3   0    0.0   0.3
add_combo r4_a05  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    5   0    0.0   0.3
add_combo r4_a06  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    8   0    0.0   0.3
add_combo r4_a07  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    15  0    0.0   0.3
add_combo r4_a08  0.008  0.15  0.5   0.05   0.0001  0.04  0.1    5   0    0.0   0.3
add_combo r4_a09  0.008  0.15  0.5   0.05   0.0001  0.04  0.1    8   0    0.0   0.3
add_combo r4_a10  0.008  0.15  0.5   0.05   0.0001  0.04  0.1    15  0    0.0   0.3
add_combo r4_a11  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    20  0    0.0   0.3
add_combo r4_a12  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    3   0    0.0   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK B (8 configs): bpsf exploration (0.002-0.005)
#
# Low bpsf tightens scale prune threshold; paired with higher dgt to prevent
# densify-prune runaway.  Compensation: bpsf*dgt ≈ constant.
#   bpsf=0.002 → dgt=0.0004, bpsf=0.003 → dgt=0.0003, bpsf=0.004 → dgt=0.0002
# Varying ldn/ldist/mop to explore interaction with tighter scale pruning.
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd    mvncc
add_combo r4_b01  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0    0.0   0.3
add_combo r4_b02  0.003  0.15  0.5   0.05   0.0003  0.03  0.1    0   0    0.0   0.3
add_combo r4_b03  0.004  0.15  0.5   0.07   0.0002  0.04  0.1    0   0    0.0   0.3
add_combo r4_b04  0.004  0.2   0.5   0.05   0.0002  0.04  0.1    0   0    0.0   0.3
add_combo r4_b05  0.002  0.15  0.5   0.05   0.0004  0.04  0.1    0   0    0.0   0.3
add_combo r4_b06  0.002  0.15  0.5   0.05   0.0004  0.03  0.1    0   0    0.0   0.3
add_combo r4_b07  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_b08  0.005  0.15  0.5   0.05   0.0001  0.03  0.1    0   0    0.0   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK C (8 configs): prune_min_scale_threshold (pmst)
#
# Far Gaussians: min_scale ~0.22.  Close Gaussians: min_scale ~0.02.
# pmst=0.1 → prune if min_scale>0.1 (kills fat Gaussians, keeps thin ones)
# Combined with bpsf ∈ {0.005, 0.003} (0.003 uses dgt=0.0003 for compensation).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst  pd    mvncc
add_combo r4_c01  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0.05  0.0   0.3
add_combo r4_c02  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0.1   0.0   0.3
add_combo r4_c03  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0.15  0.0   0.3
add_combo r4_c04  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0.25  0.0   0.3
add_combo r4_c05  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0.05  0.0   0.3
add_combo r4_c06  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0.1   0.0   0.3
add_combo r4_c07  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0.15  0.0   0.3
add_combo r4_c08  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0.25  0.0   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK D (10 configs): Soft regularization — push Gaussians to surface
#
# Keep bpsf=0.005, no extra pruning params.
# Vary {ldist, lmvg, ldn, mop} to reduce floaters via loss landscape.
# Higher ldist/lmvg push Gs to surface.  Lower ldn reduces round floaters.
# Higher mop removes low-opacity drifters (far Gs have opacity=0.79).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd    mvncc
add_combo r4_d01  0.005  0.15  0.5   0.07   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d02  0.005  0.15  0.5   0.10   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d03  0.005  0.15  0.5   0.15   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d04  0.005  0.15  0.5   0.05   0.0001  0.02  0.1    0   0    0.0   0.3
add_combo r4_d05  0.005  0.15  0.5   0.05   0.0001  0.03  0.1    0   0    0.0   0.3
add_combo r4_d06  0.005  0.15  1.0   0.05   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d07  0.005  0.15  2.0   0.05   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d08  0.005  0.25  0.5   0.05   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d09  0.005  0.35  0.5   0.05   0.0001  0.04  0.1    0   0    0.0   0.3
add_combo r4_d10  0.005  0.2   1.0   0.07   0.0001  0.03  0.1    0   0    0.0   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK E (12 configs): Joint optimization — combine best strategies
#
# Mix psa + low bpsf (with dgt compensation) + stronger regularization + higher mop.
# These target the known interactions:
#   - low bpsf (0.003) + dgt=0.0003 = tight scale pruning without runaway
#   - high ldist + low ldn = soft push to surface + allow flat discs
#   - high mop = remove lower-opacity drifters
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd    mvncc
add_combo r4_e01  0.003  0.2   0.5   0.05   0.0003  0.04  0.1    5   0    0.0   0.3
add_combo r4_e02  0.003  0.2   1.0   0.07   0.0003  0.03  0.1    5   0    0.0   0.3
add_combo r4_e03  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    8   0.1  0.0   0.3
add_combo r4_e04  0.005  0.2   0.5   0.05   0.0001  0.04  0.1    5   0    0.0   0.3
add_combo r4_e05  0.005  0.2   1.0   0.07   0.0001  0.03  0.1    5   0    0.0   0.3
add_combo r4_e06  0.005  0.2   1.0   0.10   0.0001  0.03  0.1    5   0    0.0   0.3
add_combo r4_e07  0.005  0.25  1.0   0.07   0.0001  0.02  0.1    8   0    0.0   0.3
add_combo r4_e08  0.005  0.2   0.5   0.05   0.0001  0.04  0.1    8   0.1  0.0   0.3
add_combo r4_e09  0.003  0.2   0.5   0.05   0.0003  0.04  0.1    5   0.1  0.0   0.3
add_combo r4_e10  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    5   0.1  0.0   0.3
add_combo r4_e11  0.005  0.25  1.0   0.05   0.0001  0.04  0.1    10  0    0.0   0.3
add_combo r4_e12  0.003  0.2   1.0   0.07   0.0003  0.03  0.1    0   0.15 0.0   0.3

# ════════════════════════════════════════════════════════════════════════════
# BLOCK F (10 configs): percent_dense exploration
#
# With the percent_dense bug fixed (was always 0.0 → always split, never clone),
# we now explore the correct split/clone threshold.
# Scene extent = 296.82.  pd × extent gives the split-vs-clone boundary:
#   pd=0.001 → 0.30   (mostly split — similar to old buggy behavior)
#   pd=0.005 → 1.48   (split if max_scale > 1.48, else clone)
#   pd=0.01  → 2.97   (original developer intent from .bak file)
#   pd=0.02  → 5.94
#   pd=0.05  → 14.84  (mostly clone)
#   pd=0.1   → 29.68  (default — almost always clone, never split)
#
# Use proven-good base params from r3_003: bpsf=0.005, ldist=0.05, ldn=0.04
# Cross with bpsf ∈ {0.005, 0.003} for interaction (0.003 uses dgt=0.0003).
# ════════════════════════════════════════════════════════════════════════════
#               name     bpsf   mop   lmvg  ldist  dgt     ldn   ldssim psa  pmst pd     mvncc
add_combo r4_f01  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.001  0.3
add_combo r4_f02  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.005  0.3
add_combo r4_f03  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.01   0.3
add_combo r4_f04  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.02   0.3
add_combo r4_f05  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.05   0.3
add_combo r4_f06  0.005  0.15  0.5   0.05   0.0001  0.04  0.1    0   0    0.1    0.3
add_combo r4_f07  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0    0.005  0.3
add_combo r4_f08  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0    0.01   0.3
add_combo r4_f09  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    0   0    0.02   0.3
add_combo r4_f10  0.003  0.15  0.5   0.05   0.0003  0.04  0.1    5   0    0.01   0.3

TOTAL_RUNS=${#RUN_NAMES[@]}

echo "============================================================"
echo "TOSCA Round 4 Benchmark — g2s_max reduction + percent_dense, 60 configs"
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
          'sweep':'round4'}
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
        'prune_scale_anisotropy': params.get('prune_scale_anisotropy', ''),
        'prune_min_scale_threshold': params.get('prune_min_scale_threshold', ''),
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
echo "Round 4 Benchmark Complete"
echo "============================================================"
echo "  GPU completed: $COMPLETED / $TOTAL_RUNS"
echo "  GPU failed:    ${#FAILED_RUNS[@]}"
echo "  CPU evals:     $((CPU_COMPLETED + CPU_FAILED))  (ok: $CPU_COMPLETED, fail: $CPU_FAILED)"
echo "  Time:          ${HOURS}h ${MINS}m"
echo "  Summary:       $SUMMARY_CSV"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
    echo "  Failed runs:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - $f"
    done
fi
echo ""
