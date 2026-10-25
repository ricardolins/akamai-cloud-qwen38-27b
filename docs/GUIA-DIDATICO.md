# Guia didático — Qwen3.8-27B (BF16) com vLLM em 4× RTX PRO 6000 Blackwell na Akamai Cloud

Este guia explica, passo a passo, **o que foi feito, por que, e como reproduzir**. Os números finais estão em [RESULTADOS.md](RESULTADOS.md).

---

## 1. Conceitos em 2 minutos

Quando um LLM responde a uma requisição, o trabalho acontece em **duas fases**, e cada métrica mede uma delas:

```
 requisição chega
      │
      ▼
 ┌────────────┐   o modelo lê o prompt inteiro de uma vez
 │  PREFILL   │   (limitado por COMPUTAÇÃO — FLOPs)
 └─────┬──────┘
       │  ◄── TTFT: "time to first token" = tempo até o 1º token sair
       ▼
 ┌────────────┐   gera 1 token por vez, cada um relendo os pesos
 │   DECODE   │   (limitado por BANDA DE MEMÓRIA)
 └─────┬──────┘
       │  ◄── ITL: "inter-token latency" = tempo entre dois tokens
       ▼
 resposta completa  ◄── E2E: tempo total da requisição
```

| Métrica | O que é | Depende principalmente de |
|---|---|---|
| **TTFT** | Tempo até o primeiro token | Tamanho do prompt × capacidade de cálculo |
| **ITL** | Intervalo entre tokens consecutivos | Banda de memória da GPU (ler 54 GB de pesos a cada token) |
| **TPOT** | Tempo médio por token de saída, **excluindo** o primeiro: `(E2E − TTFT) / (tokens − 1)` | Igual ao ITL, mas por requisição |
| **E2E** | Latência total: `TTFT + TPOT × (tokens − 1)` | As duas fases |
| **tok/s por usuário** | `1000 / TPOT` — a "velocidade de digitação" que o usuário vê | Decode |
| **tok/s total** | Tokens gerados por segundo somando todos os usuários | Decode + concorrência |

> **Por que o TTFT cresce com o prompt e o ITL quase não?** No prefill o modelo precisa processar todos os tokens do prompt (10k tokens ≈ 10× o trabalho de 1k). No decode, cada token novo custa quase o mesmo, não importa o tamanho do prompt — o custo dominante é ler os pesos da memória. Neste modelo isso é ainda mais acentuado, porque 75% das camadas usam *linear attention*, cujo custo por token não cresce com o contexto (seção 3).

---

## 2. Infraestrutura

| Item | Valor |
|---|---|
| Plataforma | Akamai Cloud — LKE (Kubernetes gerenciado) |
| Plano do node | `g3-gpu-rtxpro6000-blackwell-4` |
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Server Edition, **96 GB** cada (97.887 MiB), 600 W, compute capability 12.0 |
| Interconexão entre GPUs | **PCIe Gen5 x16 via host bridge (`PHB`) — sem NVLink** |
| CPU / RAM | 64 vCPUs / 692 GB |
| Disco local | ~4,2 TB NVMe |
| Driver / CUDA | 615.71.10 / CUDA 13.4 (driver) |
| GPU Operator | já instalado (driver e toolkit vêm da imagem do LKE) |

Detalhes brutos: [`results/hardware.txt`](../results/hardware.txt).

> **Por que a falta de NVLink importa:** com *tensor parallelism*, as GPUs trocam resultados parciais (all-reduce) **a cada camada**. Sem NVLink essa troca vai pelo PCIe. No log do vLLM aparecem os avisos: `Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs` e `SymmMemCommunicator: Device capability 12.0 not supported` — ou seja, só o NCCL "genérico" fica disponível.

---

## 3. O modelo

| Item | Valor |
|---|---|
| Modelo | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — Apache 2.0, não exige token |
| Precisão | **BF16** (o "FP16" pedido — o checkpoint oficial é BF16; existe também uma versão FP8) |
| Parâmetros | 27,78 B → **~51,7 GiB** de pesos |
| Arquitetura | Híbrida: 64 camadas, das quais **48 são *linear attention*** (Gated DeltaNet) e **16 são *full attention*** (1 a cada 4) |
| Contexto nativo | **262.144 tokens** |

**Consequência prática da arquitetura híbrida:** só as 16 camadas de *full attention* guardam KV cache por token (4 cabeças KV × 256 dim). Isso dá ~**64 KB por token** — um prompt de 120k ocupa só ~8 GB de KV. As camadas *linear* guardam um estado de tamanho fixo por sequência (o vLLM chama de "Mamba cache"). Por isso:
- prompts enormes cabem com folga;
- o ITL quase não piora com o contexto (14 ms em 8k → 16,6 ms em 240k).

---

## 4. Passo a passo

### 4.1 Conferir o cluster

```bash
export KUBECONFIG=~/Downloads/<kubeconfig-do-cluster>.yaml
kubectl get nodes -o wide
kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # → 4
```

**Por quê:** confirmar que o node tem as 4 GPUs anunciadas ao Kubernetes (`nvidia.com/gpu: 4`), o que significa que driver, container toolkit e device plugin estão funcionando.

### 4.2 Escolher a versão do vLLM

Usamos a imagem oficial **`vllm/vllm-openai:v0.32.0`** (última versão estável na data). A receita oficial do modelo ([recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)) foi validada a partir da 0.26. Evitamos `nightly` para que o resultado seja reprodutível.

### 4.3 O manifesto (TP=4)

Arquivo: [`k8s/10-vllm-qwen38-27b.yaml`](../k8s/10-vllm-qwen38-27b.yaml). Os pontos importantes:

| Flag / campo | Valor | Por quê |
|---|---|---|
| modelo (posicional) | `Qwen/Qwen3.8-27B` | No vLLM ≥ 0.30 o modelo vai como argumento posicional (`--model` está depreciado) |
| `--tensor-parallel-size` | `4` | Divide **cada camada** entre as 4 GPUs: prefill e decode ficam ~4× mais "largos" |
| `--max-model-len` | `262144` | Limite nativo do modelo; necessário para o teste de 240k (os testes até 120k rodaram com 131072 — isso não altera a latência) |
| `--language-model-only` | — | O checkpoint tem uma torre de visão; o teste é só texto, então economizamos memória |
| `--no-enable-prefix-caching` | — | **Essencial para benchmark**: com cache de prefixo, prompts repetidos não seriam reprocessados e o TTFT ficaria artificialmente baixo |
| `--gpu-memory-utilization` | `0.90` | Reserva 90% da VRAM para pesos + KV cache |
| `/dev/shm` | `emptyDir` em memória, 32 Gi | O NCCL usa memória compartilhada entre os processos de cada GPU |
| volume `models` | `hostPath` no NVMe local | Os ~56 GB do modelo ficam no disco do node e não são baixados de novo se o pod reiniciar |
| `startupProbe` | até 40 min | 1ª subida: baixar a imagem (9,2 GB) + modelo (~56 GB) + compilar CUDA graphs |

Aplicar:

```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/10-vllm-qwen38-27b.yaml
kubectl -n llm-bench get pods -w
```

**O que esperar no log** (`kubectl -n llm-bench logs deploy/qwen38-27b`):

```
Loading safetensors checkpoint shards: 100% | 18/18
GPU KV cache size: 4,504,533 tokens, Maximum concurrency for 262,144 tokens per request: 17.18x
init engine (profile, create kv cache, warmup model) took ~190 s (compilation: ~46 s)
Application startup complete.
```

Tempos observados: imagem 1 min 50 s; download do modelo < 90 s; subida do engine ~3–4 min.

### 4.4 Teste rápido

```bash
kubectl -n llm-bench exec deploy/qwen38-27b -- curl -s localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","prompt":"The capital of Brazil is","max_tokens":12,"temperature":0}'
# → " Brasília, a city that was purpose-built to serve as the"
```

### 4.5 O benchmark

Script: [`scripts/bench.sh`](../scripts/bench.sh). Ele roda **dentro do pod do vLLM**, usando o cliente `vllm bench serve` que já vem na mesma imagem (sem rede entre cliente e servidor, que não é o que queremos medir).

Decisões de metodologia:

| Decisão | Por quê |
|---|---|
| `--dataset-name random` com `--random-range-ratio 0` | Prompts com **exatamente** 8.000 / 10.000 / 60.000 / 120.000 / 240.000 tokens |
| `--random-output-len 512` + `--ignore-eos` | Toda resposta tem exatamente 512 tokens → TPOT e E2E comparáveis |
| `--max-concurrency 1` | Mede a latência de **um usuário** — é o que os alvos de TTFT e tok/s descrevem |
| `--num-warmups 1` | A primeira requisição paga custos de aquecimento; não entra na conta |
| 10 requisições (8k, 10k), 5 (60k, 120k), 3 (240k) | Prompts maiores demoram mais; os desvios foram < 1%, então isso basta |
| Concorrência 2, 4, 8 com prompt de 10k | Mostra quanto a velocidade **por usuário** cai quando vários usam ao mesmo tempo |
| `--percentile-metrics ttft,tpot,itl,e2el` | p50, p90, p99 de cada métrica |

Rodar:

```bash
POD=$(kubectl -n llm-bench get pod -l app=qwen38-27b -o jsonpath='{.items[0].metadata.name}')
kubectl -n llm-bench cp scripts/bench.sh $POD:/tmp/bench.sh
kubectl -n llm-bench exec $POD -- bash -c 'nohup /tmp/bench.sh tp4 512 main > /models/results-tp4.log 2>&1 &'
# 240k (servidor com --max-model-len 262144):
kubectl -n llm-bench exec $POD -- bash -c 'nohup /tmp/bench.sh tp4-262k 512 240k > /models/results-tp4-262k.log 2>&1 &'
```

Junto com o benchmark gravamos a telemetria das GPUs a cada 5 s:

```bash
nvidia-smi dmon -s pucm -d 5 -o T > /models/results/<tag>/gpu-dmon.log
```

Depois, copiar e resumir:

```bash
kubectl -n llm-bench cp $POD:/models/results/tp4 results/tp4
python3 scripts/summarize.py results/tp4          # tabela de latências
python3 scripts/gpu_stats.py results/tp4/gpu-dmon.log   # potência / temperatura / clock
```

### 4.6 Comparação com TP=1 (uma GPU)

O modelo **cabe em uma GPU**: 51,7 GiB de pesos em 96 GB. Para decidir com dados, repetimos tudo com [`k8s/11-vllm-qwen38-27b-tp1.yaml`](../k8s/11-vllm-qwen38-27b-tp1.yaml):

```bash
kubectl -n llm-bench scale deploy/qwen38-27b --replicas=0
kubectl apply -f k8s/11-vllm-qwen38-27b-tp1.yaml
```

> **Armadilha encontrada:** na 1ª tentativa o vLLM recusou subir com TP=1:
> ```
> ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (642).
> ```
> As camadas *linear attention* precisam de um bloco de estado por sequência ativa. Em uma GPU sobram só ~640 blocos, e o padrão do vLLM reserva 1024 sequências. Correção: `--max-num-seqs=256`. Com TP=4 o problema não aparece porque a memória é 4× maior.

Com TP=1 sobram **31,6 GiB** para KV → **511.103 tokens** de cache (contra 4,5 milhões com TP=4). Ainda cabe um prompt de 240k, mas só ~2 desse tamanho ao mesmo tempo.

---

## 5. Incidente durante o teste

Na 1ª execução, o **node caiu** às 20:47:04 UTC, cerca de 3 min depois do início do benchmark (cenário de 10k, 1 usuário). A VM parou de responder (sem ping, kubelet fora). O Kubernetes marcou o node como `NotReady` e despejou o pod. **A causa foi um problema no host da Akamai**, confirmado pelo dono da conta. Depois do reboot do node, o vLLM subiu sozinho (o modelo continuava no disco local) e o benchmark foi refeito do zero.

A telemetria das execuções seguintes também descarta energia e temperatura como causa: pico de **381 W** (de 600 W de limite), no máximo **62 °C**, clock estável em ~2,4 GHz e sem *throttling*.

Lições:
- Em cluster de **um único node**, uma falha do host derruba o serviço inteiro. Para produção, use pelo menos 2 nodes de GPU.
- Grave telemetria de GPU **durante** o benchmark — sem ela não há como descartar energia ou temperatura depois.
- Deixar o modelo em `hostPath`/NVMe local reduziu a recuperação a ~4 min (sem novo download).

---

## 6. Limpeza

```bash
kubectl delete namespace llm-bench
# o modelo fica em /var/lib/llm-models no node; para apagar, rode um pod com hostPath e remova o diretório,
# ou recicle o node pelo LKE.
```
