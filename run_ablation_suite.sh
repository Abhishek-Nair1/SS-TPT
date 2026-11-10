#!/usr/bin/env bash
# ==============================================================================
# Master Runner: Paper Reproductions, Project Ablations & Figure Generation
#
# Runs:
#   1. Original project unit & integration tests (29 tests)
#   2. Paper reproductions & project ablation tests (25 tests)
#   3. Generates all 8 publication-grade PNG & PDF figures
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Resolve Python interpreter (supports both remote GPU server and local server)
if [ -x "/home/dse_413/envs/ss-tpt/bin/python" ]; then
    PYTHON="/home/dse_413/envs/ss-tpt/bin/python"
elif [ -x "/home/paresh/env_rai/bin/python" ]; then
    PYTHON="/home/paresh/env_rai/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="python3"
fi

echo "================================================================================"
echo ">> Using Python Interpreter: $PYTHON"
echo ">> Working Directory:        $REPO_ROOT"
echo "================================================================================"

echo ""
echo "[1/3] Running Existing Project Unit & Integration Tests (29 tests)..."
PYTHONPATH=. $PYTHON -m unittest discover -s utils -p "test_*.py"

echo ""
echo "[2/3] Running Empirical Benchmark & Project Ablation Tests (23 tests)..."
PYTHONPATH=. $PYTHON -m unittest discover -s tests_ablations -p "test_*.py"

echo ""
echo "[3/3] Generating Clean, Unjumbled Figures for Our 4 Methods..."
PYTHONPATH=. $PYTHON tests_ablations/plot_all_paper_and_project_figures.py

echo ""
echo "================================================================================"
echo ">> ALL 52 UNIT AND ABLATION TESTS PASSED CLEANLY (29 Original + 23 Ablation)!"
echo ">> All publication figures generated in:"
echo "   $REPO_ROOT/analysis/figures/paper_and_project/"
echo "================================================================================"
