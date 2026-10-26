#!/bin/bash
# Latency benchmark for Qwen3.8-27B on vLLM. Runs INSIDE the vLLM pod (bench client = same image).
# Fixed-length random prompts + fixed output (--ignore-eos) so TTFT/ITL/TPOT/E2E are comparable across sizes.
# Usage: bench.sh <tag> [output_len] [set]   set = main (default) | 240k | all | sweep | quant
#   quant: 10k at 1/2/4/8 users + 122k at 1/2 users (FP8 / NVFP4 / MTP comparison)
#   MODEL / TOKENIZER env vars override the served model name and tokenizer
#   sweep: 10k prompt at concurrency 16..128 to find the throughput plateau
#   240k needs the server started with --max-model-len >= 262144
set -euo pipefail
TAG=${1:-baseline}; OUT=${2:-512}; SET=${3:-main}
DIR=/models/results/$TAG; mkdir -p "$DIR"
run() {  # run <input_len> <concurrency> <num_prompts>
  local IN=$1 C=$2 N=$3 NAME="in${1}_out${OUT}_c${2}"
  echo "=== $(date -u +%T) $NAME"
  vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
    --model ${MODEL:-qwen3.8-27b} --tokenizer ${TOKENIZER:-Qwen/Qwen3.8-27B} \
    --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 0 \
    --ignore-eos --num-prompts "$N" --max-concurrency "$C" --request-rate inf \
    --num-warmups 1 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$DIR" --result-filename "$NAME.json" 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL|Median E2EL" 
}
if [[ $SET == main || $SET == all ]]; then
# single user (latency): the four prompt sizes requested
run 8000   1 10
run 10000  1 10
run 60000  1 5
run 120000 1 5
# 10k prompt under concurrency: per-user generation speed as load grows
run 10000  2 10
run 10000  4 16
run 10000  8 24
fi
if [[ $SET == 240k || $SET == all ]]; then
# very long context: 240k prompt (close to the 262,144 native limit)
run 240000 1 3
fi
if [[ $SET == sweep ]]; then
# throughput plateau: keep doubling concurrency; 2x requests per level so it reaches steady state
for C in ${SWEEP:-16 32 64 128}; do run 10000 "$C" $((C * 2)); done
fi
if [[ $SET == quant ]]; then
run 10000 1 10; run 10000 2 10; run 10000 4 16; run 10000 8 24
run 122000 1 5; run 122000 2 4
fi
echo "=== done $(date -u +%T)"
