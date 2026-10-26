# Results — Qwen3.8-27B (BF16) on 4× RTX PRO 6000 Blackwell

All numbers are p50 (median) unless stated. Output is fixed at 512 tokens. Raw data: [`results/`](../results). The same tables are in the Word report: [`reports/Qwen3.8-27B-benchmark-results.docx`](../reports/Qwen3.8-27B-benchmark-results.docx).

## 1. Targets vs measured (10k-token prompt)

| Target | Requirement | TP=4 (4 GPUs) | TP=1 (1 GPU) | Verdict |
|---|---|---|---|---|
| TTFT | 16 ms | 1,570 ms | 1,562 ms | **Not met.** ~1.6 s is the compute floor for prefilling 10k tokens on a 27B model |
| Generation speed | 35–45 tok/s per user | **71.6 tok/s** | 26.1 tok/s | **Met with TP=4**, single user and up to 4 concurrent users |

## 2. Single user, by prompt length

| Prompt | TTFT TP=4 | TTFT TP=1 | ITL TP=4 | ITL TP=1 | tok/s per user TP=4 | tok/s per user TP=1 | E2E TP=4 | E2E TP=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 1.25 s | 1.22 s | 14.0 ms | 38.2 ms | **71.7** | 26.2 | 8.4 s | 20.7 s |
| 10k | 1.57 s | 1.56 s | 14.0 ms | 38.3 ms | **71.6** | 26.1 | 8.7 s | 21.1 s |
| 60k | 9.83 s | 11.16 s | 14.6 ms | 40.5 ms | **69.0** | 24.7 | 17.2 s | 31.8 s |
| 120k | 20.95 s | 27.00 s | 15.3 ms | 43.5 ms | **66.3** | 23.1 | 28.7 s | 49.1 s |
| 240k | 46.72 s | 72.80 s | 16.6 ms | 48.6 ms | **61.5** | 20.7 | 55.0 s | 97.5 s |

**Reading it:**
- **TTFT does not improve with 4 GPUs for short prompts** (8k–10k: same as one GPU). The GPUs have no NVLink, so the all-reduce between them goes over PCIe and eats the gain. TP=4 helps only on long prompts: −22% at 120k, −36% at 240k.
- **Decode is where TP=4 shines**: 2.7× faster per user. Each token reads ¼ of the weights from each GPU.
- **Decode barely slows with context** (14 → 16.6 ms from 8k to 240k), because 48 of the 64 layers use linear attention.

## 3. Concurrency with a 10k prompt — the throughput plateau

| Users | TTFT TP=4 | TTFT TP=1 | tok/s per user TP=4 | tok/s per user TP=1 | **tok/s total TP=4** | **tok/s total TP=1** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.6 s | 1.6 s | **71.6** | 26.1 | 59 | 24 |
| 2 | 2.3 s | 2.8 s | **57.0** | 24.8 | 91 | 44 |
| 4 | 4.6 s | 4.3 s | **50.6** | 22.5 | 139 | 76 |
| 8 | 6.2 s | 5.1 s | 32.1 | 18.0 | **185** | 122 |
| 16 | 13.5 s | 6.1 s | 9.8 | 12.2 | 124 | 170 |
| 32 | 13.5 s | 6.3 s | 4.6 | 7.2 | 132 | **210** |
| 64 | 13.6 s | 53.7 s | 2.2 | 5.7 | 135 | 202 |
| 128 | 16.5 s | — | 1.1 | — | 141 | — |

**Reading it:**
- **The bottleneck under load is prefill, not decode.** Each request brings 10k tokens to process and produces only 512. Beyond the plateau, extra users only lengthen the queue: TTFT p99 reaches 7 minutes at 128 users on TP=4.
- **TP=4 peaks at 185 tok/s with 8 users** and settles at ~125–140. Up to **4 users** it keeps ≥35 tok/s each.
- **TP=1 on one GPU reaches 210 tok/s**, more than all four GPUs in TP=4. At 64 users its TTFT jumps (54 s) because one GPU's KV cache (511k tokens) holds only ~48 requests of 10k.
- **TP=4 telemetry at saturation**: ~190 W average per GPU while "100% busy". The GPUs spend most of their time waiting on the PCIe all-reduce. A single GPU under the same load runs at ~500 W.

## 4. What this means for deployment

| Layout on one 4-GPU server | 1-user speed | Users at ≥35 tok/s | Peak total output | Basis |
|---|---:|---:|---:|---|
| 1× TP=4 | 72 tok/s | 4 | 185 tok/s | measured |
| 4× TP=1 (one replica per GPU) | 26 tok/s | 0 | ~840 tok/s | **estimate** (4 × measured single-GPU plateau) |
| 2× TP=2 | — | — | — | **not measured yet**: the most promising balance, because vLLM's custom all-reduce works on 2 PCIe GPUs but is disabled for 4 |

- **Latency target (35–45 tok/s per user)** → TP=4 meets it, for up to ~4 concurrent users with 10k prompts.
- **Throughput target (many users)** → independent single-GPU replicas behind a load balancer, accepting ~26 tok/s per user.
- **Both** → test 2× TP=2 next.
- The TTFT target of 16 ms is not achievable for 10k prompts on any of these layouts. Prefill needs ~540 TFLOPs, so TTFT will always be on the order of a second. For sub-second perceived latency, use prefix caching (shared system prompts / documents) or a smaller or quantized model (FP8/NVFP4).

## 5. Sizing for TTFT ≤ 16 s and ≥ 25 tok/s per user (10k prompt)

A concurrency level passes when **TTFT p99 ≤ 16 s** (worst case, not just the median) **and** per-user speed **≥ 25 tok/s**.

| Users | TP=4 TTFT p50 / p99 | TP=4 tok/s per user | TP=4 | TP=1 TTFT p50 / p99 | TP=1 tok/s per user | TP=1 |
|---:|---:|---:|:---:|---:|---:|:---:|
| 1 | 1.6 / 1.6 s | 71.6 | ✅ | 1.6 / 1.6 s | 26.1 | ✅ |
| 2 | 2.3 / 3.1 s | 57.0 | ✅ | 2.8 / 3.0 s | 24.8 | ⚠️ just below |
| 4 | 4.6 / 6.1 s | 50.6 | ✅ | 4.3 / 5.9 s | 22.5 | ❌ |
| 8 | 6.2 / 12.0 s | 32.1 | ✅ | 5.1 / 11.8 s | 18.0 | ❌ |
| 16 | 13.5 / 52.5 s | 9.8 | ❌ | 6.1 / 23.5 s | 12.2 | ❌ |

**Capacity per 4-GPU server:** TP=4 → **8 concurrent users** (0.5 GPU per user). TP=1 → 1 user per GPU, so 4 per server (1 GPU per user), or 8 if 24.8 tok/s is accepted as 25.

| Concurrent users | **TP=4** | TP=1 (strict ≥25 tok/s) |
|---:|---:|---:|
| 8 | **4 GPUs** (1 server) | 8 GPUs (2 servers) |
| 40 | **20 GPUs** (5 servers) | 40 GPUs (10 servers) |
| 100 | **52 GPUs** (13 servers) | 100 GPUs (25 servers) |

**Recommendation for this target: TP=4, one 4-GPU server per 8 concurrent users.** It meets both limits with margin (32 tok/s, worst-case TTFT 12 s). TP=1 sits right at the 25 tok/s edge.

Notes:
- "Concurrent users" means requests in flight at the same time. At 8 concurrent requests each one takes ~22 s, so a TP=4 server completes ~0.36 requests/s (**~1,300 requests of 10k tokens per hour**).
- Levels between 8 and 16 were not measured. TP=4 capacity is between 8 and ~12; 8 is the safe measured value.
- Valid for 10k prompts and 512-token answers. Longer prompts or answers reduce capacity.
- FP8 weights would likely let TP=1 keep ≥25 tok/s with several users per GPU (estimate, not measured).

## 6. Infrastructure observations

- **Two node outages** during testing, both platform-side (host stall, then an ACPI power-off from the platform). Analysis: [`results/node-incidents.md`](../results/node-incidents.md). Production should use ≥2 GPU nodes.
- **TP=1 runs one GPU at its 600 W limit** (~78 °C) with a slightly lower clock: mild power capping. TP=4 stays far below the limit.
- The node's SSH port is open to the Internet with password authentication (brute-force attempts in the journal). Apply a Cloud Firewall and keep TCP 10250 from `192.168.128.0/17` open for `kubectl exec/logs`.
