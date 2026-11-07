#!/bin/bash
# Push missing files to bhaskargurram-ai/agenthallu-bench
#
# Prerequisites:
#   1. Clone the repo:  git clone git@github.com:bhaskargurram-ai/agenthallu-bench.git /tmp/agenthallu-push
#   2. Run this script from the agenthallu_bench/ directory
#
# This copies all missing source files and pushes them.

set -e

REPO_DIR="/tmp/agenthallu-push"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Source: $SRC_DIR"
echo "Target: $REPO_DIR"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning repo..."
    git clone git@github.com:bhaskargurram-ai/agenthallu-bench.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull origin main

# Copy missing directories
for dir in metrics agent injector tracer tests; do
    echo "Copying $dir/..."
    mkdir -p "$REPO_DIR/$dir"
    cp "$SRC_DIR/$dir/"*.py "$REPO_DIR/$dir/" 2>/dev/null || true
done

# Copy config
cp "$SRC_DIR/config.py" "$REPO_DIR/config.py"

# Copy new experiment scripts
cp "$SRC_DIR/experiments/rejudge_p2_ensemble.py" "$REPO_DIR/experiments/"
cp "$SRC_DIR/experiments/run_interceptor_control.py" "$REPO_DIR/experiments/"

# Copy corrected analysis
cp "$SRC_DIR/analysis/hop_analysis.py" "$REPO_DIR/analysis/hop_analysis.py"

# Copy updated data files
cp "$SRC_DIR/data/results/hop_rates.json" "$REPO_DIR/data/results/hop_rates.json"

# Stage and commit
git add -A
git status

echo ""
echo "Review the changes above, then run:"
echo "  cd $REPO_DIR"
echo "  git commit -m 'fix: add missing source code + corrected hop_analysis'"
echo "  git push origin main"
