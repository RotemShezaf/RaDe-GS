#!/bin/bash
#
# Script to run tests with SLURM (srun) using GPU and geo_splat environment
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh test_load_utils.py # Run specific test file
#   ./run_tests.sh -k test_name       # Run specific test by name

set -e  # Exit on error

# Configuration
ENV_NAME="geo_splat"
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$TEST_DIR")"

echo "========================================"
echo "Running Tests with SLURM (srun)"
echo "========================================"
echo "Test directory: $TEST_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Environment: $ENV_NAME"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found in PATH"
    exit 1
fi

# Determine test arguments
if [ $# -eq 0 ]; then
    TEST_ARGS="$TEST_DIR"
else
    TEST_ARGS="$@"
fi

echo "Test arguments: $TEST_ARGS"
echo ""

# Run tests with srun
# Request 1 GPU, 8 CPUs, 32GB RAM, 1 hour time limit
echo "Submitting job with srun..."
echo "Command: srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00"
echo ""

srun --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=1:00:00 \
     --pty \
     bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        cd $PROJECT_ROOT
        
        # Set TORCH_CUDA_ARCH_LIST to suppress compilation warning
        export TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6'
        
        echo 'Python version:'
        python --version
        echo ''
        echo 'Running pytest...'
        python -m pytest $TEST_ARGS -v --tb=short --color=yes
    "

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"
