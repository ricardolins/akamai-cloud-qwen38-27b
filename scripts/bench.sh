#!/bin/bash
# Latency benchmark for Qwen3.8-27B on vLLM. Runs INSIDE the vLLM pod (bench client = same image).
# Fixed-length random prompts + fixed output (--ignore-eos) so TTFT/ITL/TPOT/E2E are comparable across sizes.
# Usage: bench.sh <tag> [output_len]
set -euo pipefail
TAG=${1:-baseline}; OUT=${2:-512}
DIR=/models/results/$TAG; mkdir -p "$DIR"
run() {  # run <input_len> <concurrency> <num_prompts>
  local IN=$1 C=$2 N=$3 NAME="in${1}_out${OUT}_c${2}"
  echo "=== $(date -u +%T) $NAME"
  vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
    --model qwen3.8-27b --tokenizer Qwen/Qwen3.8-27B \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 0 \
    --ignore-eos --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$DIR" --result-filename "$NAME.json" 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL|Median E2EL" 
}
# single user (latency): the four prompt sizes requested
run 8000   1 10
run 10000  1 10
run 60000  1 5
run 120000 1 5
# 10k prompt under concurrency: per-user generation speed as load grows
run 10000  2 10
run 10000  4 16
run 10000  8 24
echo "=== done $(date -u +%T)"
