# Qwen3.8-27B (BF16) on Akamai Cloud — 4× RTX PRO 6000 Blackwell

Serving **Qwen/Qwen3.8-27B** in 16-bit (BF16) with **vLLM** on Akamai Cloud (LKE, plan `g3-gpu-rtxpro6000-blackwell-4`), and measuring **TTFT, ITL, TPOT and end-to-end latency** for prompts of 8k, 10k, 60k and 120k tokens.

## Key results (10k-token prompt)

| | TP=4 (4 GPUs) | TP=1 (1 GPU) | Target |
|---|---:|---:|---:|
| TTFT | 1.57 s | 1.56 s | 16 ms — not achievable (prefill compute floor) |
| Generation speed, 1 user | **71.6 tok/s** | 26.1 tok/s | 35–45 tok/s — **met with TP=4** |
| Users kept at ≥35 tok/s each | 4 | 0 | — |
| Peak total output | 185 tok/s | 210 tok/s (one GPU!) | — |

Long context works up to 240k tokens (TTFT 46.7 s, 61 tok/s on TP=4). Under load, the PCIe-only interconnect makes 4× single-GPU replicas the throughput-optimal layout.

**Native FP8 / NVFP4 on Blackwell, one GPU ($3/h):**

| ~122k prompt, 1 user | TTFT | Generation | GSM8K vs BF16 |
|---|---:|---:|---|
| BF16 (TP=1) | 27.0 s | 23 tok/s | 92.8% |
| FP8 | 21.0 s | 37 tok/s | 92.8% (identical) |
| NVFP4 | **18.0 s** | **41 tok/s** | 95.6% (no measurable loss) |

At 10k, one FP8/NVFP4 GPU keeps ≥25 tok/s for at least 8 concurrent users (≥4× fewer GPUs than BF16 TP=4). Against a 2× H100 setup on Oracle Cloud ($10/GPU-h list) for the ~122k workload, 2 RTX PRO 6000 with NVFP4 cost **70% less** for 18 s / 41 tok/s.

- [docs/GUIDE.md](docs/GUIDE.md): step-by-step guide (concepts, setup, methodology, pitfalls)
- [docs/RESULTS.md](docs/RESULTS.md): full results and analysis
- [reports/Qwen3.8-27B-benchmark-results.docx](reports/Qwen3.8-27B-benchmark-results.docx): Word report with all tables
- [results/node-incidents.md](results/node-incidents.md): analysis of the two node outages during testing

## Layout

```
k8s/        Kubernetes manifests (vLLM TP=4, TP=1 comparison)
scripts/    bench.sh (inside the vLLM pod), summarize.py, gpu_stats.py, make_report_docx.js
results/    raw vllm bench JSON per configuration + hardware/startup logs
docs/       GUIDE.md (step by step) and RESULTS.md (analysis)
reports/    Word report generated from results/
00-logs/    commands applied to the cluster
```
