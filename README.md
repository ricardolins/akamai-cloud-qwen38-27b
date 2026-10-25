# Qwen3.8-27B (BF16) on Akamai Cloud — 4× RTX PRO 6000 Blackwell

Serving **Qwen/Qwen3.8-27B** in 16-bit (BF16) with **vLLM** on Akamai Cloud (LKE, plan `g3-gpu-rtxpro6000-blackwell-4`), and measuring **TTFT, ITL, TPOT and end-to-end latency** for prompts of 8k, 10k, 60k and 120k tokens.

> Status: in progress. Results and the step-by-step guide will be added as the runs complete.

## Layout

```
k8s/        Kubernetes manifests (vLLM TP=4, TP=1 comparison)
scripts/    bench.sh (runs inside the vLLM pod), summarize.py (JSON -> table)
results/    raw vllm bench JSON per configuration + hardware/startup logs
docs/       didactic guide and results analysis
00-logs/    commands applied to the cluster
```
