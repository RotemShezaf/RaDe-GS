#!/bin/bash
#SBATCH --job-name=gaussian_tests
#SBATCH --output=test_output_%j.log
#SBATCH --error=test_error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#
# SLURM batch script to run tests with GPU allocation
#
# Usage:
#   sbatch run_tests_batch.sh            # Submit as batch job
#

set -e

# Configuration
ENV_NAME="geo_splat"
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$TEST_DIR")"

echo "========================================"
echo "Running Tests (SLURM Batch Job)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Test directory: $TEST_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Environment: $ENV_NAME"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Set TORCH_CUDA_ARCH_LIST to suppress compilation warning
export TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6'

# Change to project root
cd $PROJECT_ROOT

# Print environment info
echo "Python version:"
python --version
echo ""

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

echo "Installed packages (key ones):"
python -c "import numpy; print(f'numpy: {numpy.__version__}')" || echo "numpy not found"
python -c "import scipy; print(f'scipy: {scipy.__version__}')" || echo "scipy not found"
python -c "import pytest; print(f'pytest: {pytest.__version__}')" || echo "pytest not found"
echo ""

# Run tests
echo "Running pytest..."
echo "========================================"
python -m pytest $TEST_DIR -v --tb=short --color=yes

# Exit with pytest's exit code
exit $?
