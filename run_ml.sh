#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this compute job on a Linux fleet host; do not run the experiment on the Mac.' >&2
  exit 2
fi
python_exec="${LAB_PYTHON:-$repo_root/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LAB_C=100
export LAB_DATA="${LAB_DATA:-$repo_root/ml-loop-lab/data}"
mkdir -p "$LAB_DATA" "$repo_root/ml-loop-lab/results"
cd "$repo_root/ml-loop-lab"
if [[ ! -f "$LAB_DATA/splits.npz" ]]; then "$python_exec" src/prep_data.py; fi
if [[ ! -f "$LAB_DATA/emb.npz" ]]; then "$python_exec" src/embed.py; fi
if [[ ! -f "$LAB_DATA/drop.json" ]]; then "$python_exec" src/dedupe.py; fi
if [[ ! -f "$LAB_DATA/drift_monitor.json" || ! -f "$LAB_DATA/hardness.npy" ]]; then
  "$python_exec" src/prepare_run.py
fi
run_dir="$(mktemp -d "$repo_root/ml-loop-lab/results/reproduction-XXXXXXXX")"
echo "Writing a separate reproduction to $run_dir"
"$python_exec" -m unittest test_method -v
"$python_exec" -u src/loop.py --suite all --seeds final --final --procs 6 --out "$run_dir/final.jsonl"
"$python_exec" src/analysis.py "$run_dir" "$run_dir/final.jsonl"
"$python_exec" src/signal_thresholds.py "$run_dir/final.jsonl" "$run_dir/signals.json"
"$python_exec" src/tune_on_test.py "$run_dir/peek-demo.json"
echo 'Completed a separate repeat of the fixed experiment. It is not new independent test data.'
