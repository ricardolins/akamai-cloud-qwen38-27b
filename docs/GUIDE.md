# Step-by-step guide — Qwen3.8-27B (BF16) with vLLM on 4× RTX PRO 6000 Blackwell on Akamai Cloud

This guide explains **what was done, why, and how to reproduce it**. Final numbers are in [RESULTS.md](RESULTS.md).

---

## 1. Concepts in 2 minutes

Serving one LLM request happens in **two phases**, and each metric measures one of them:

```
 request arrives
      │
      ▼
 ┌────────────┐   the model reads the whole prompt at once
 │  PREFILL   │   (bound by COMPUTE — FLOPs)
 └─────┬──────┘
       │  ◄── TTFT: time to first token
       ▼
 ┌────────────┐   generates one token at a time, re-reading the weights each time
 │   DECODE   │   (bound by MEMORY BANDWIDTH)
 └─────┬──────┘
       │  ◄── ITL: inter-token latency
       ▼
 full response  ◄── E2E: end-to-end latency
```

| Metric | What it is | Driven mainly by |
|---|---|---|
| **TTFT** | Time until the first output token | Prompt length × compute |
| **ITL** | Gap between two consecutive output tokens | GPU memory bandwidth (reading ~54 GB of weights per token) |
| **TPOT** | Average time per output token **excluding** the first: `(E2E − TTFT) / (tokens − 1)` | Same as ITL, but per request |
| **E2E** | Total latency: `TTFT + TPOT × (tokens − 1)` | Both phases |
| **tok/s per user** | `1000 / TPOT`, the "typing speed" a user sees | Decode |
| **tok/s total** | Output tokens per second summed over all users | Decode + concurrency |

> **Why does TTFT grow with the prompt while ITL barely moves?** Prefill must process every prompt token: 10k tokens is about 10× the work of 1k. In decode, each new token costs roughly the same regardless of prompt length, because the dominant cost is reading the weights from memory. This model accentuates that: 75% of its layers use *linear attention*, whose per-token cost does not grow with context (section 3).

---

## 2. Infrastructure

| Item | Value |
|---|---|
| Platform | Akamai Cloud — LKE (managed Kubernetes) |
| Node plan | `g3-gpu-rtxpro6000-blackwell-4` |
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition, **96 GB** each (97,887 MiB), 600 W, compute capability 12.0 |
| GPU interconnect | **PCIe Gen5 x16 through the host bridge (`PHB`). No NVLink** |
| CPU / RAM | 64 vCPUs / 692 GB |
| Local disk | ~4.2 TB NVMe |
| Driver / CUDA | 615.71.10 / CUDA 13.4 (driver) |
| GPU Operator | pre-installed (driver and toolkit come with the LKE image) |

Raw details: [`results/hardware.txt`](../results/hardware.txt).

> **Why the missing NVLink matters:** with *tensor parallelism*, the GPUs exchange partial results (all-reduce) **at every layer**. Without NVLink that exchange goes over PCIe. The vLLM log shows `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs` and `SymmMemCommunicator: Device capability 12.0 not supported`, which means only generic NCCL is available.

---

## 3. The model

| Item | Value |
|---|---|
| Model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0, no token required |
| Precision | **BF16** (the requested "FP16": the official checkpoint is BF16; an FP8 variant also exists) |
| Parameters | 27.78 B → **~51.7 GiB** of weights |
| Architecture | Hybrid: 64 layers, of which **48 are linear attention** (Gated DeltaNet) and **16 are full attention** (1 in every 4) |
| Native context | **262,144 tokens** |

**Practical consequence of the hybrid design:** only the 16 full-attention layers keep a per-token KV cache (4 KV heads × 256 dims). That is about **64 KB per token**, so a 120k prompt needs only ~8 GB of KV. The linear layers keep a fixed-size state per sequence (vLLM calls it the "Mamba cache"). Therefore:
- huge prompts fit comfortably;
- ITL barely degrades with context (14 ms at 8k → 16.6 ms at 240k).

---

## 4. Step by step

### 4.1 Check the cluster

```bash
export KUBECONFIG=~/Downloads/<cluster-kubeconfig>.yaml
kubectl get nodes -o wide
kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # → 4
```

**Why:** confirm that the node advertises its 4 GPUs to Kubernetes (`nvidia.com/gpu: 4`). That means the driver, the container toolkit and the device plugin all work.

### 4.2 Pick the vLLM version

We used the official image **`vllm/vllm-openai:v0.32.0`**, the latest stable release at the time. The model's official recipe ([recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)) is validated from 0.26 onward. We avoided `nightly` so results are reproducible.

### 4.3 The manifest (TP=4)

File: [`k8s/10-vllm-qwen38-27b.yaml`](../k8s/10-vllm-qwen38-27b.yaml). Key points:

| Flag / field | Value | Why |
|---|---|---|
| model (positional) | `Qwen/Qwen3.8-27B` | On vLLM ≥ 0.30 the model is a positional argument (`--model` is deprecated) |
| `--tensor-parallel-size` | `4` | Splits **every layer** across the 4 GPUs, so prefill and decode are ~4× "wider" |
| `--max-model-len` | `262144` | The model's native limit, needed for the 240k test. The tests up to 120k ran with 131072, which does not change latency |
| `--language-model-only` | — | The checkpoint includes a vision tower. The test is text-only, so this saves memory |
| `--no-enable-prefix-caching` | — | **Essential for benchmarking.** With prefix caching, repeated prompts would not be re-processed and TTFT would look artificially low |
| `--gpu-memory-utilization` | `0.90` | Use 90% of VRAM for weights + KV cache |
| `/dev/shm` | in-memory `emptyDir`, 32 Gi | NCCL uses shared memory between the per-GPU processes |
| `models` volume | `hostPath` on local NVMe | The ~56 GB model stays on the node's disk and is not downloaded again if the pod restarts |
| `startupProbe` | up to 40 min | First start: pull the image (9.2 GB) + download the model (~56 GB) + compile CUDA graphs |

Apply:

```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/10-vllm-qwen38-27b.yaml
kubectl -n llm-bench get pods -w
```

**What to expect in the log** (`kubectl -n llm-bench logs deploy/qwen38-27b`):

```
Loading safetensors checkpoint shards: 100% | 18/18
GPU KV cache size: 4,504,533 tokens, Maximum concurrency for 262,144 tokens per request: 17.18x
init engine (profile, create kv cache, warmup model) took ~190 s (compilation: ~46 s)
Application startup complete.
```

Observed times: image pull 1 min 50 s, model download < 90 s, engine start ~3–4 min.

### 4.4 Smoke test

```bash
kubectl -n llm-bench exec deploy/qwen38-27b -- curl -s localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","prompt":"The capital of Brazil is","max_tokens":12,"temperature":0}'
# → " Brasília, a city that was purpose-built to serve as the"
```

### 4.5 The benchmark

Script: [`scripts/bench.sh`](../scripts/bench.sh). It runs **inside the vLLM pod**, using the `vllm bench serve` client shipped in the same image, so no client-server network sits between them (that is not what we want to measure).

Methodology choices:

| Choice | Why |
|---|---|
| `--dataset-name random` with `--random-range-ratio 0` | Prompts of **exactly** 8,000 / 10,000 / 60,000 / 120,000 / 240,000 tokens |
| `--random-output-len 512` + `--ignore-eos` | Every response is exactly 512 tokens, so TPOT and E2E are comparable |
| `--max-concurrency 1` | Measures **single-user** latency, which is what the TTFT and tok/s targets describe |
| `--num-warmups 1` | The first request pays warm-up costs and is excluded |
| 10 requests (8k, 10k), 5 (60k, 120k), 3 (240k) | Larger prompts take longer. Spread was < 1%, so this is enough |
| Concurrency 2, 4, 8 with a 10k prompt | Shows how much **per-user** speed drops when several users share the server |
| `--percentile-metrics ttft,tpot,itl,e2el` | p50, p90 and p99 of each metric |

Run:

```bash
POD=$(kubectl -n llm-bench get pod -l app=qwen38-27b -o jsonpath='{.items[0].metadata.name}')
kubectl -n llm-bench cp scripts/bench.sh $POD:/tmp/bench.sh
kubectl -n llm-bench exec $POD -- bash -c 'nohup /tmp/bench.sh tp4 512 main > /models/results-tp4.log 2>&1 &'
# 240k (server started with --max-model-len 262144):
kubectl -n llm-bench exec $POD -- bash -c 'nohup /tmp/bench.sh tp4-262k 512 240k > /models/results-tp4-262k.log 2>&1 &'
```

GPU telemetry is recorded every 5 s during each run:

```bash
nvidia-smi dmon -s pucm -d 5 -o T > /models/results/<tag>/gpu-dmon.log
```

Then copy and summarize:

```bash
kubectl -n llm-bench cp $POD:/models/results/tp4 results/tp4
python3 scripts/summarize.py results/tp4               # latency table
python3 scripts/gpu_stats.py results/tp4/gpu-dmon.log  # power / temperature / clock
```

### 4.6 Comparison with TP=1 (one GPU)

The model **fits on one GPU**: 51.7 GiB of weights in 96 GB. To decide with data, everything was repeated with [`k8s/11-vllm-qwen38-27b-tp1.yaml`](../k8s/11-vllm-qwen38-27b-tp1.yaml):

```bash
kubectl -n llm-bench scale deploy/qwen38-27b --replicas=0
kubectl apply -f k8s/11-vllm-qwen38-27b-tp1.yaml
```

> **Pitfall:** on the first attempt vLLM refused to start with TP=1:
> ```
> ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (642).
> ```
> The linear-attention layers need one state block per active sequence. One GPU only has room for ~640 blocks, while vLLM reserves 1024 sequences by default. Fix: `--max-num-seqs=256`. TP=4 does not hit this because it has 4× the memory.

With TP=1, **31.6 GiB** is left for KV → **511,103 tokens** of cache (versus 4.5 million with TP=4). A 240k prompt still fits, but only ~2 of that size at the same time.

### 4.7 Quantized weights on one GPU: FP8, NVFP4 and MTP

**Why:** decode is limited by the bytes read per token, and prefill by math throughput. Blackwell tensor cores run **FP8 and FP4 (NVFP4) natively**. The H100 has FP8 but no FP4. Smaller weights mean faster decode, and lower-precision math means faster prefill, on a single $3/h GPU.

Manifest: [`k8s/20-vllm-tp1-quantized.yaml`](../k8s/20-vllm-tp1-quantized.yaml). Four independent TP=1 servers, one per GPU, run at the same time:

| Server | Checkpoint | Weights in GPU memory | Extra flag |
|---|---|---|---|
| `qwen-fp8` | `Qwen/Qwen3.8-27B-FP8` (official) | 28.0 GiB | — |
| `qwen-fp8-mtp` | same | 28.0 GiB | `--speculative-config={"method":"mtp","num_speculative_tokens":3}` |
| `qwen-nvfp4` | `Inferact/Qwen3.8-27B-NVFP4` (from the vLLM recipe) | 23.2 GiB | — |
| `qwen-nvfp4-mtp` | same | 23.2 GiB | MTP as above |

How to confirm NVFP4 really runs on the FP4 tensor cores (and is not just dequantized to BF16):
```
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
```

Run the comparison set (10k prompt at 1/2/4/8 users + 122k at 1/2 users) in each pod:
```bash
MODEL=qwen-fp8 /tmp/bench.sh tp1-fp8 512 quant
```

MTP acceptance comes from the server's Prometheus metrics:
```bash
curl -s localhost:8000/metrics | grep -E 'spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total'
```

Quality check ([`scripts/gsm8k_eval.py`](../scripts/gsm8k_eval.py)): the first 250 GSM8K test questions, temperature 0, thinking disabled, exact numeric answer. It runs against BF16, FP8 and NVFP4 the same way. It uses only the Python standard library (the dataset comes from the Hugging Face datasets-server JSON API), because the vLLM image does not ship `pandas`.

> **Pitfalls**
> - With MTP, **ITL** is measured per accepted *chunk* of tokens, so it looks worse (~35 ms) even when generation is much faster. Use **TPOT** (and tok/s per user = 1000 / TPOT) to compare speed.
> - MTP helps at 10k but **hurts at ~122k**: the draft head's attention over the long context costs more than the accepted tokens save. Enable it per workload, not by default.
> - Random-token prompts are fine for TTFT/decode speed, but MTP acceptance on real text can differ. Re-measure acceptance on production prompts.
> - The four quantized servers shared one host while running. That is realistic for one-replica-per-GPU serving, but slightly pessimistic compared with a single server alone.

---

## 5. Incident during the test

During the first run the **node went down** at 20:47:04 UTC, about 3 minutes into the benchmark (10k scenario, single user). The VM stopped responding (no ping, kubelet gone). Kubernetes marked the node `NotReady` and evicted the pod. **The cause was an Akamai host issue**, confirmed by the account owner. After the node rebooted, vLLM came back on its own (the model was still on local disk), and the benchmark was re-run from scratch.

Telemetry from the following runs: with **TP=4** the GPUs peaked at **381 W** (of a 600 W limit) and **62 °C**, clock stable at ~2.4 GHz. With **TP=1** the single GPU runs **at the 600 W limit** (peaks of 609 W) and up to **78 °C**, with a slightly lower clock (~2.35 GHz), i.e. mild power capping. Neither explains the outages (see `results/node-incidents.md`).

Lessons:
- On a **single-node** cluster, a host failure takes the whole service down. For production, use at least 2 GPU nodes.
- Record GPU telemetry **during** benchmarks. Without it, power and temperature cannot be ruled out afterwards.
- Keeping the model on `hostPath`/local NVMe cut recovery to ~4 min, with no new download.

---

## 6. Cleanup

```bash
kubectl delete namespace llm-bench
# The model stays in /var/lib/llm-models on the node. To delete it, run a pod with that hostPath
# and remove the directory, or recycle the node through LKE.
```
