#!/usr/bin/env python3
"""Summarize `nvidia-smi dmon -s pucm -o T` logs: per-GPU power/temperature/SM/clock, idle vs busy.
Usage: gpu_stats.py results/<tag>/gpu-dmon.log"""
import sys, collections
cols = None; per = collections.defaultdict(list)
for line in open(sys.argv[1]):
    if line.startswith("#Time"):
        cols = line[1:].split(); continue
    if line.startswith("#") or not line.strip() or cols is None:
        continue
    v = line.split()
    r = dict(zip(cols, v))
    try:
        per[int(r["gpu"])].append({k: float(r[k]) for k in ("pwr", "gtemp", "sm", "pclk") if r.get(k, "-") != "-"})
    except ValueError:
        pass
print("| GPU | samples | busy samples (SM>20%) | power avg busy (W) | power max (W) | temp max (C) | SM avg busy (%) | clock avg busy (MHz) |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for g in sorted(per):
    s = per[g]; b = [x for x in s if x.get("sm", 0) > 20] or s
    avg = lambda k, xs: sum(x[k] for x in xs) / len(xs)
    print(f"| {g} | {len(s)} | {len(b)} | {avg('pwr', b):.0f} | {max(x['pwr'] for x in s):.0f} | {max(x['gtemp'] for x in s):.0f} | {avg('sm', b):.0f} | {avg('pclk', b):.0f} |")
