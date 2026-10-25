#!/usr/bin/env python3
"""Turn `vllm bench serve` JSON results into a Markdown table.
Usage: summarize.py results/<tag> [results/<tag2> ...]"""
import json, pathlib, re, sys

def g(d, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None

rows = []
for base in sys.argv[1:]:
    for f in sorted(pathlib.Path(base).glob("*.json")):
        d = json.loads(f.read_text())
        m = re.match(r"in(\d+)_out(\d+)_c(\d+)", f.stem)
        inp, out, conc = map(int, m.groups())
        tpot = g(d, "median_tpot_ms")
        rows.append(dict(
            cfg=pathlib.Path(base).name, inp=inp, out=out, conc=conc, n=d.get("completed"),
            ttft50=g(d, "median_ttft_ms"), ttft99=g(d, "p99_ttft_ms"),
            itl50=g(d, "median_itl_ms"), itl99=g(d, "p99_itl_ms"),
            tpot50=tpot, e2e50=g(d, "median_e2el_ms"),
            tps_user=(1000.0 / tpot) if tpot else None,
            tps_total=g(d, "output_throughput"),
        ))

rows.sort(key=lambda r: (r["cfg"], r["conc"], r["inp"]))
fmt = lambda v, p=0: "—" if v is None else f"{v:,.{p}f}"
print("| Config | Prompt | Conc. | n | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | tok/s per user | tok/s total |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['cfg']} | {r['inp']:,} | {r['conc']} | {r['n']} | {fmt(r['ttft50'])} | {fmt(r['ttft99'])} | "
          f"{fmt(r['itl50'],1)} | {fmt(r['itl99'],1)} | {fmt(r['tpot50'],1)} | {fmt((r['e2e50'] or 0)/1000,1)} | "
          f"{fmt(r['tps_user'],1)} | {fmt(r['tps_total'],1)} |")
