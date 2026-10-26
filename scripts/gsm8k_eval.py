#!/usr/bin/env python3
"""Quick quality check: GSM8K exact-match accuracy against an OpenAI-compatible vLLM server.

Same N questions (first N of the GSM8K test split), temperature 0, thinking disabled, so the
BF16 / FP8 / NVFP4 runs are directly comparable. This is a sanity check of quantization loss,
not a full evaluation.

Usage (inside the vLLM pod): python3 gsm8k_eval.py --model <served-name> --n 250 --out result.json
"""
import argparse
import concurrent.futures as cf
import json
import re
import time
import urllib.request

# Hugging Face datasets-server JSON API: standard library only, nothing to install in the vLLM image
ROWS = "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset={o}&length={n}"
PROMPT = ("Solve the following math problem. Think step by step, then give the final answer "
          "on the last line in the form 'Answer: <number>'.\n\n{q}")


def number(text):
    text = text.replace(",", "")
    m = re.findall(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)", text)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(m[-1]) if m else None


def ask(base, model, q):
    body = {"model": model, "temperature": 0, "max_tokens": 768,
            "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    while len(rows) < a.n:
        n = min(100, a.n - len(rows))
        with urllib.request.urlopen(ROWS.format(o=len(rows), n=n), timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    questions = [r["question"] for r in rows]
    gold = [float(r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(lambda q: ask(a.base, a.model, q), questions))
    pred = [number(o) for o in outs]
    correct = sum(p is not None and abs(p - g) < 1e-6 for p, g in zip(pred, gold))
    res = {"model": a.model, "n": a.n, "correct": correct, "accuracy": correct / a.n,
           "seconds": round(time.time() - t0, 1)}
    json.dump({**res, "samples": [{"q": q, "gold": g, "pred": p, "out": o}
                                  for q, g, p, o in zip(questions, gold, pred, outs)]},
              open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
