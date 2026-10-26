#!/usr/bin/env node
// Build the Word report (tables) straight from the vllm bench JSON files in results/.
// Usage: node scripts/make_report_docx.js [repo_root] [output.docx]
// Re-run after new results land (e.g. the concurrency sweeps) — sections fill in automatically.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, PageOrientation, BorderStyle, LevelFormat, Footer, PageNumber,
} = require("docx");

const ROOT = process.argv[2] || path.join(__dirname, "..");
const OUT = process.argv[3] || path.join(ROOT, "reports", "Qwen3.8-27B-benchmark-results.docx");

// ---------- data ----------
function load(tag) {
  const dir = path.join(ROOT, "results", tag);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter((f) => /^in\d+_out\d+_c\d+\.json$/.test(f)).map((f) => {
    const d = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    const [, inp, out, c] = f.match(/in(\d+)_out(\d+)_c(\d+)/).map(Number);
    const tpot = d.median_tpot_ms;
    return {
      tag, inp, out, c, n: d.completed,
      ttft50: d.median_ttft_ms, ttft99: d.p99_ttft_ms,
      itl50: d.median_itl_ms, itl99: d.p99_itl_ms,
      tpot50: tpot, e2e50: d.median_e2el_ms / 1000,
      tpsUser: tpot ? 1000 / tpot : null, tpsTotal: d.output_throughput,
    };
  });
}
const tp4 = [...load("tp4"), ...load("tp4-262k"), ...load("tp4-sweep")];
const tp1 = [...load("tp1"), ...load("tp1-sweep")];
const swept = (rows) => rows.some((r) => r.inp === 10000 && r.c >= 16);
// Largest measured concurrency (10k prompt) meeting TTFT p99 <= ttftMs and per-user speed >= tps
const SLO = { ttftMs: 16000, tps: 25 };
const capacity = (rows) => conc(rows).filter((r) => r.ttft99 <= SLO.ttftMs && r.tpsUser >= SLO.tps).reduce((m, r) => Math.max(m, r.c), 0);
const single = (rows) => rows.filter((r) => r.c === 1).sort((a, b) => a.inp - b.inp);
const conc = (rows) => rows.filter((r) => r.inp === 10000).sort((a, b) => a.c - b.c);
const find = (rows, inp, c) => rows.find((r) => r.inp === inp && r.c === c);

function gpuStats(tag) {
  const f = path.join(ROOT, "results", tag, "gpu-dmon.log");
  if (!fs.existsSync(f)) return null;
  let cols = null; const per = {};
  for (const line of fs.readFileSync(f, "utf8").split("\n")) {
    if (line.startsWith("#Time")) { cols = line.slice(1).trim().split(/\s+/); continue; }
    if (!cols || line.startsWith("#") || !line.trim()) continue;
    const v = line.trim().split(/\s+/); const r = Object.fromEntries(cols.map((c, i) => [c, v[i]]));
    const g = +r.gpu; if (Number.isNaN(g)) continue;
    (per[g] = per[g] || []).push({ pwr: +r.pwr, t: +r.gtemp, sm: +r.sm, clk: +r.pclk });
  }
  const all = Object.values(per).flat(); if (!all.length) return null;
  const busy = all.filter((x) => x.sm > 20);
  const avg = (xs, k) => xs.reduce((s, x) => s + x[k], 0) / xs.length;
  return { gpus: Object.keys(per).length, pwrBusy: avg(busy, "pwr"), pwrMax: Math.max(...all.map((x) => x.pwr)),
           tMax: Math.max(...all.map((x) => x.t)), smBusy: avg(busy, "sm"), clkBusy: avg(busy, "clk") };
}

// ---------- formatting helpers ----------
const FONT = "Arial";
const PAGE_W = 16838, PAGE_H = 11906, MARGIN = 1080;        // A4 landscape, 0.75" margins
const CONTENT_W = PAGE_W - 2 * MARGIN;                        // 14678 DXA
const C = { head: "1F3864", headText: "FFFFFF", band: "F2F5FA", good: "E2F0D9", bad: "FBE5D6", grid: "BFC7D5" };
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "—"
  : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const k = (n) => (n >= 1000 ? `${n / 1000}k` : `${n}`);

const border = { style: BorderStyle.SINGLE, size: 4, color: C.grid };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, { header = false, fill, bold = false, align = AlignmentType.RIGHT } = {}) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: C.head, type: ShadingType.CLEAR, color: "auto" }
      : fill ? { fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: header ? AlignmentType.CENTER : align,
      children: [new TextRun({ text: String(text), font: FONT, size: header ? 18 : 18,
        bold: header || bold, color: header ? C.headText : undefined })] })],
  });
}

// cols: [{h, w (relative), align?}], rows: [[...values]], fills?: fn(rowIdx,colIdx)->color
function table(cols, rows, { fills, firstColLeft = true } = {}) {
  const total = cols.reduce((s, c) => s + c.w, 0);
  const widths = cols.map((c) => Math.floor((c.w / total) * CONTENT_W));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((s, w) => s + w, 0);
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: cols.map((c, i) => cell(c.h, widths[i], { header: true })) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((v, ci) => cell(v, widths[ci], {
        fill: (fills && fills(ri, ci)) || (ri % 2 ? C.band : undefined),
        align: ci === 0 && firstColLeft ? AlignmentType.LEFT : (cols[ci].align || AlignmentType.RIGHT),
        bold: ci === 0,
      })) })),
    ],
  });
}

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [R_(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [R_(t)] });
const R_ = (o) => new TextRun(typeof o === "string" ? { text: o, font: FONT } : { font: FONT, ...o });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: [R_({ text: t, ...opts })] });
const runs = (...parts) => new Paragraph({ spacing: { after: 120 },
  children: parts.map(R_) });
const B = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
  children: (Array.isArray(t) ? t : [t]).map(R_) });
const note = (t) => new Paragraph({ spacing: { before: 80, after: 200 },
  children: [R_({ text: t, italics: true, size: 18, color: "595959" })] });
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------- content ----------
const t4_10 = find(tp4, 10000, 1), t1_10 = find(tp1, 10000, 1);
const body = [];

body.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [R_("Qwen3.8-27B (BF16) on Akamai Cloud")] }));
body.push(P("4× NVIDIA RTX PRO 6000 Blackwell — vLLM latency and throughput benchmark", { size: 28, color: "404040" }));
body.push(P(`Test date: 2026-10-25 (UTC) · vLLM 0.32.0 · Generated ${new Date().toISOString().slice(0, 10)}`, { size: 18, color: "7F7F7F" }));

// 1. Executive summary
body.push(H1("1. Executive summary"));
body.push(table(
  [{ h: "Target", w: 3 }, { h: "Requirement", w: 3 }, { h: "TP=4 (4 GPUs) measured", w: 3 }, { h: "TP=1 (1 GPU) measured", w: 3 }, { h: "Verdict", w: 4, align: AlignmentType.LEFT }],
  [
    ["TTFT, 10k prompt", "16 ms", `${fmt(t4_10 && t4_10.ttft50)} ms`, `${fmt(t1_10 && t1_10.ttft50)} ms`,
      "Not met — ~1.6 s is the compute floor for prefilling 10k tokens on a 27B model"],
    ["Generation speed, 10k prompt", "35–45 tok/s per user", `${fmt(t4_10 && t4_10.tpsUser, 1)} tok/s`, `${fmt(t1_10 && t1_10.tpsUser, 1)} tok/s`,
      "Met with TP=4 (single user and up to 4 concurrent users); not met with TP=1"],
  ],
  { fills: (r, c) => (c === 2 ? (r === 0 ? C.bad : C.good) : c === 3 ? C.bad : undefined) },
));
body.push(gap());
body.push(B([{ text: "TP=4 is the recommended configuration for latency: ", bold: true },
  `${fmt(t4_10 && t4_10.tpsUser)} tok/s per user at 10k (ITL ${fmt(t4_10 && t4_10.itl50, 1)} ms), 2.7× faster generation than one GPU.`]));
body.push(B([{ text: "TTFT is bound by prefill compute, not by the number of GPUs for short prompts: ", bold: true },
  "at 8k–10k TP=1 and TP=4 have the same TTFT (~1.2–1.6 s). The GPUs have no NVLink, so the all-reduce between them runs over PCIe and absorbs the gain. TP=4 only helps on long prompts (−22% at 120k, −36% at 240k)."]));
body.push(B([{ text: "Long context works up to the native 262k limit: ", bold: true },
  `240k-token prompt: TTFT ${fmt(find(tp4, 240000, 1) && find(tp4, 240000, 1).ttft50 / 1000, 1)} s and ${fmt(find(tp4, 240000, 1) && find(tp4, 240000, 1).tpsUser)} tok/s per user on TP=4. The hybrid architecture (48 of 64 layers use linear attention) keeps decode speed almost flat with context.`]));
const c4 = find(tp4, 10000, 4), c8 = find(tp4, 10000, 8);
const peak = (rows) => { const r = conc(rows); return r.length ? r.reduce((a, b) => (b.tpsTotal > a.tpsTotal ? b : a)) : null; };
const p4 = peak(tp4), p1 = peak(tp1);
if (swept(tp4) && swept(tp1)) body.push(B([{ text: "Under load, one GPU outserves four: ", bold: true },
  `TP=4 peaks at ${fmt(p4.tpsTotal)} output tok/s (${p4.c} users) and settles at ~${fmt(Math.min(...conc(tp4).filter((x) => x.c >= 16).map((x) => x.tpsTotal)))}–${fmt(Math.max(...conc(tp4).filter((x) => x.c >= 16).map((x) => x.tpsTotal)))} tok/s beyond that, while a single GPU with TP=1 reaches ~${fmt(p1.tpsTotal)} tok/s (${p1.c} users). For many concurrent users, four independent TP=1 replicas (one per GPU) are the throughput-optimal layout — estimated ~${fmt(4 * p1.tpsTotal)} tok/s per server (estimate, not measured).`]));
{
  const c4cap = capacity(tp4), c1cap = capacity(tp1);
  body.push(B([{ text: `Sizing for TTFT ≤ ${SLO.ttftMs / 1000} s and ≥ ${SLO.tps} tok/s per user (10k prompt): `, bold: true },
    `TP=4 serves ${c4cap} concurrent users per 4-GPU server (${fmt(4 / c4cap, 2)} GPU per user); TP=1 serves ${c1cap} per GPU (${fmt(1 / c1cap, 2)} GPU per user). TP=4 needs half the GPUs for this target.`]));
}
body.push(B([{ text: "Concurrency: ", bold: true },
  `with 10k prompts TP=4 keeps ≥35 tok/s per user up to 4 concurrent users (${fmt(c4 && c4.tpsUser)} tok/s); at 8 users it drops to ${fmt(c8 && c8.tpsUser)} tok/s.`]));

// 2. Setup
body.push(H1("2. Test setup"));
body.push(table([{ h: "Item", w: 2 }, { h: "Value", w: 6, align: AlignmentType.LEFT }], [
  ["Platform", "Akamai Cloud — LKE (managed Kubernetes), plan g3-gpu-rtxpro6000-blackwell-4"],
  ["GPUs", "4× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each, 600 W, compute capability 12.0"],
  ["GPU interconnect", "PCIe Gen5 x16 via host bridge (PHB) — no NVLink"],
  ["CPU / RAM / disk", "64 vCPUs / 692 GB / ~4.2 TB local NVMe"],
  ["Driver / CUDA", "615.71.10 / CUDA 13.4 driver; vLLM image built with CUDA 13.0, PyTorch 2.13"],
  ["Model", "Qwen/Qwen3.8-27B, BF16 (16-bit), 27.8 B parameters (~51.7 GiB), native context 262,144 tokens"],
  ["Serving engine", "vLLM 0.32.0 (vllm/vllm-openai:v0.32.0), prefix caching disabled, GPU memory utilization 0.90"],
  ["Configurations", "TP=4: one replica across the 4 GPUs · TP=1: one replica on one GPU (--max-num-seqs 256)"],
  ["Load generator", "vllm bench serve inside the server pod; random prompts of fixed length; output fixed at 512 tokens (--ignore-eos); 1 warm-up request"],
], { firstColLeft: true }));
body.push(note("TTFT = time to first token · ITL = inter-token latency · TPOT = time per output token after the first · E2E = end-to-end latency · tok/s per user = 1000 / TPOT · p50 = median, p99 = 99th percentile."));

// 3. Single user
body.push(H1("3. Single-user latency by prompt length"));
body.push(H2("3.1 TP=4 vs TP=1 side by side"));
const sizes = [...new Set([...single(tp4), ...single(tp1)].map((r) => r.inp))].sort((a, b) => a - b);
body.push(table(
  [{ h: "Prompt (tokens)", w: 2 }, { h: "TTFT p50 TP=4 (s)", w: 2 }, { h: "TTFT p50 TP=1 (s)", w: 2 },
   { h: "ITL p50 TP=4 (ms)", w: 2 }, { h: "ITL p50 TP=1 (ms)", w: 2 }, { h: "tok/s/user TP=4", w: 2 }, { h: "tok/s/user TP=1", w: 2 },
   { h: "E2E p50 TP=4 (s)", w: 2 }, { h: "E2E p50 TP=1 (s)", w: 2 }],
  sizes.map((s) => { const a = find(tp4, s, 1) || {}, b = find(tp1, s, 1) || {};
    return [k(s), fmt(a.ttft50 / 1000, 2), fmt(b.ttft50 / 1000, 2), fmt(a.itl50, 1), fmt(b.itl50, 1),
            fmt(a.tpsUser, 1), fmt(b.tpsUser, 1), fmt(a.e2e50, 1), fmt(b.e2e50, 1)]; }),
  { fills: (r, c) => (c === 5 ? C.good : c === 6 ? C.bad : undefined) },
));
body.push(note("Output length 512 tokens in every case. Green/orange: per-user generation speed vs the 35–45 tok/s target."));

const detail = (rows) => table(
  [{ h: "Prompt", w: 2 }, { h: "Requests", w: 2 }, { h: "TTFT p50 (ms)", w: 2 }, { h: "TTFT p99 (ms)", w: 2 },
   { h: "ITL p50 (ms)", w: 2 }, { h: "ITL p99 (ms)", w: 2 }, { h: "TPOT p50 (ms)", w: 2 }, { h: "E2E p50 (s)", w: 2 },
   { h: "tok/s per user", w: 2 }],
  rows.map((r) => [k(r.inp), r.n, fmt(r.ttft50), fmt(r.ttft99), fmt(r.itl50, 1), fmt(r.itl99, 1), fmt(r.tpot50, 1), fmt(r.e2e50, 1), fmt(r.tpsUser, 1)]),
);
body.push(H2("3.2 TP=4 — detailed"));
body.push(detail(single(tp4)));
body.push(H2("3.3 TP=1 — detailed"));
body.push(detail(single(tp1)));

// 4. Concurrency
body.push(H1("4. Concurrency with a 10k-token prompt"));
body.push(P("Several users send 10k-token prompts at the same time. Per-user speed drops as the GPUs are shared, while total throughput rises until the server saturates (the plateau)."));
const concTable = (rows) => table(
  [{ h: "Concurrent users", w: 2 }, { h: "Requests", w: 2 }, { h: "TTFT p50 (s)", w: 2 }, { h: "TTFT p99 (s)", w: 2 },
   { h: "ITL p50 (ms)", w: 2 }, { h: "ITL p99 (ms)", w: 2 }, { h: "tok/s per user", w: 2 }, { h: "tok/s total", w: 2 }],
  rows.map((r) => [r.c, r.n, fmt(r.ttft50 / 1000, 2), fmt(r.ttft99 / 1000, 2), fmt(r.itl50, 1), fmt(r.itl99, 1), fmt(r.tpsUser, 1), fmt(r.tpsTotal, 1)]),
  { fills: (ri, ci) => { const r = rows[ri]; return ci === 6 ? (r.tpsUser >= 35 ? C.good : C.bad) : undefined; } },
);
body.push(H2("4.1 TP=4"));
body.push(concTable(conc(tp4)));
body.push(H2("4.2 TP=1"));
body.push(concTable(conc(tp1)));
const plateauNote = (rows, name) => {
  const r = conc(rows); if (!swept(rows)) return note(`${name}: concurrency sweep (16–128 users) in progress — this section updates when it completes.`);
  const best = r.reduce((a, b) => (b.tpsTotal > a.tpsTotal ? b : a));
  const ok = r.filter((x) => x.tpsUser >= 35).pop();
  return runs({ text: `${name}: `, bold: true },
    `peak total throughput ${fmt(best.tpsTotal)} tok/s at ${best.c} concurrent users; highest concurrency still at ≥35 tok/s per user: ${ok ? ok.c : "none"}.`);
};
body.push(H2("4.3 Throughput plateau"));
body.push(plateauNote(tp4, "TP=4"));
body.push(plateauNote(tp1, "TP=1"));
body.push(note("Green: per-user speed ≥ 35 tok/s (target). Orange: below target. Requests per level = 2× concurrency for levels ≥ 16 so the server reaches steady state."));
if (swept(tp4) && swept(tp1)) {
  body.push(H2("4.4 What the plateau means for deployment"));
  const ok4 = conc(tp4).filter((x) => x.tpsUser >= 35).pop(), ok1 = conc(tp1).filter((x) => x.tpsUser >= 35).pop();
  body.push(table(
    [{ h: "Layout on one 4-GPU server", w: 4 }, { h: "Per-user speed, 1 user (tok/s)", w: 3 }, { h: "Users at ≥35 tok/s each", w: 3 },
     { h: "Peak total output (tok/s)", w: 3 }, { h: "Basis", w: 4, align: AlignmentType.LEFT }],
    [
      ["1× TP=4 (all 4 GPUs, one replica)", fmt(find(tp4, 10000, 1).tpsUser), ok4 ? ok4.c : "none", fmt(p4.tpsTotal), "Measured"],
      ["4× TP=1 (one replica per GPU)", fmt(find(tp1, 10000, 1).tpsUser), ok1 ? `${ok1.c * 4}` : "none (per-user speed < 35)", `~${fmt(4 * p1.tpsTotal)}`, "Estimated: 4 × measured single-GPU plateau"],
      ["2× TP=2 (two replicas of 2 GPUs)", "not measured", "not measured", "not measured", "Next test: vLLM's custom all-reduce works on 2 PCIe GPUs (disabled for 4)"],
    ],
    { fills: (r, c) => (r === 2 ? "EDEDED" : undefined) },
  ));
  body.push(note("TP=4 telemetry under saturation: ~190 W average per GPU at 100% SM busy, i.e. the GPUs mostly wait on the PCIe all-reduce. A single GPU (TP=1) under the same load runs at ~500 W average."));
}

// 4.5 Sizing for an SLO
{
  const c4cap = capacity(tp4), c1cap = capacity(tp1);
  body.push(H2(`4.5 Sizing for TTFT ≤ ${SLO.ttftMs / 1000} s and ≥ ${SLO.tps} tok/s per user (10k prompt)`));
  body.push(P(`A concurrency level passes when TTFT p99 ≤ ${SLO.ttftMs / 1000} s (worst case, not just the median) and per-user generation speed ≥ ${SLO.tps} tok/s.`));
  const levels = [...new Set([...conc(tp4), ...conc(tp1)].map((r) => r.c))].filter((c) => c <= 16).sort((a, b) => a - b);
  const mark = (r) => (r ? (r.ttft99 <= SLO.ttftMs && r.tpsUser >= SLO.tps ? "pass" : "fail") : "—");
  body.push(table(
    [{ h: "Concurrent users", w: 2 }, { h: "TP=4 TTFT p50 / p99 (s)", w: 3 }, { h: "TP=4 tok/s per user", w: 2 }, { h: "TP=4", w: 1, align: AlignmentType.CENTER },
     { h: "TP=1 TTFT p50 / p99 (s)", w: 3 }, { h: "TP=1 tok/s per user", w: 2 }, { h: "TP=1", w: 1, align: AlignmentType.CENTER }],
    levels.map((c) => { const a = find(tp4, 10000, c), b = find(tp1, 10000, c);
      return [c, a ? `${fmt(a.ttft50 / 1000, 1)} / ${fmt(a.ttft99 / 1000, 1)}` : "—", a ? fmt(a.tpsUser, 1) : "—", mark(a),
              b ? `${fmt(b.ttft50 / 1000, 1)} / ${fmt(b.ttft99 / 1000, 1)}` : "—", b ? fmt(b.tpsUser, 1) : "—", mark(b)]; }),
    { fills: (ri, ci) => { const c = levels[ri]; const r = ci === 3 ? find(tp4, 10000, c) : ci === 6 ? find(tp1, 10000, c) : null;
      return r ? (mark(r) === "pass" ? C.good : C.bad) : undefined; } },
  ));
  body.push(gap());
  const perServer4 = c4cap, perServer1 = c1cap * 4;
  body.push(table(
    [{ h: "Concurrent users to serve", w: 3 }, { h: `TP=4 (${perServer4} users per 4-GPU server)`, w: 4 }, { h: `TP=1 (${c1cap} user per GPU, ${perServer1} per server)`, w: 4 }],
    [8, 40, 100].map((n) => { const s4 = Math.ceil(n / perServer4), s1 = Math.ceil(n / perServer1);
      return [n, `${s4 * 4} GPUs (${s4} server${s4 > 1 ? "s" : ""})`, `${s1 * 4} GPUs (${s1} server${s1 > 1 ? "s" : ""})`]; }),
    { fills: (r, c) => (c === 1 ? C.good : undefined) },
  ));
  const b2 = find(tp1, 10000, 2);
  body.push(note(`"Concurrent users" = requests in flight at the same time. At ${perServer4} concurrent requests a TP=4 server completes ~${fmt(perServer4 / find(tp4, 10000, perServer4).e2e50, 2)} requests/s (~${fmt(3600 * perServer4 / find(tp4, 10000, perServer4).e2e50)} per hour). Levels between 8 and 16 were not measured, so TP=4 capacity is between ${perServer4} and ~12; ${perServer4} is the safe measured value. TP=1 at 2 users gives ${fmt(b2 && b2.tpsUser, 1)} tok/s — if that is accepted as 25, TP=1 matches TP=4 (8 users per server). Valid for 10k prompts with 512-token answers; FP8 weights would likely raise TP=1 capacity (not measured).`));
}

// 5. GPU telemetry
body.push(H1("5. GPU telemetry during the runs"));
const tel = [["TP=4 — 8k to 120k + concurrency", "tp4"], ["TP=4 — 240k", "tp4-262k"], ["TP=1 — all scenarios", "tp1"],
             ["TP=4 — concurrency sweep", "tp4-sweep"], ["TP=1 — concurrency sweep", "tp1-sweep"]]
  .map(([n, t]) => [n, gpuStats(t)]).filter(([, s]) => s);
body.push(table(
  [{ h: "Run", w: 4 }, { h: "GPUs sampled", w: 2 }, { h: "Power avg busy (W)", w: 2 }, { h: "Power max (W)", w: 2 },
   { h: "Temp max (°C)", w: 2 }, { h: "SM util busy (%)", w: 2 }, { h: "SM clock busy (MHz)", w: 2 }],
  tel.map(([n, s]) => [n, s.gpus, fmt(s.pwrBusy), fmt(s.pwrMax), fmt(s.tMax), fmt(s.smBusy), fmt(s.clkBusy)]),
));
body.push(note("nvidia-smi dmon every 5 s; power limit 600 W per GPU. With TP=4 the load is spread over 4 GPUs and stays far below the limit. With TP=1 the single GPU runs at the 600 W limit (short peaks above it) and ~78 °C, with a slightly lower average clock: mild power capping, which is part of why TP=1 is slower."));

// 6. Incidents
body.push(H1("6. Incidents during testing"));
body.push(table([{ h: "Time (UTC)", w: 2 }, { h: "What happened", w: 5, align: AlignmentType.LEFT }, { h: "Cause (from the node's journal)", w: 5, align: AlignmentType.LEFT }, { h: "Impact", w: 3, align: AlignmentType.LEFT }], [
  ["2026-10-25 20:46:56", "Node stopped abruptly during the first TP=4 run", "Host-side failure: guest frozen ~3.2 s at 20:42:44 (clocksource warning), then stopped; no GPU, PCIe, OOM or kernel errors", "Run discarded and repeated from scratch"],
  ["2026-10-25 22:30:24", "Node powered off at the start of the TP=1 sweep", "Platform ACPI power-button shutdown (repeated every 10 s), not triggered inside the VM", "TP=1 sweep restarted after reboot"],
], { firstColLeft: true }));
body.push(note("Neither incident was caused by the workload. Details: results/node-incidents.md in the repository."));

// 7. Method
body.push(H1("7. Methodology notes"));
body.push(B("Prefix caching disabled so every prompt is fully prefilled (otherwise repeated prompts would show artificially low TTFT)."));
body.push(B("Fixed-length random prompts (--random-range-ratio 0) and fixed 512-token outputs (--ignore-eos), so all metrics are comparable across prompt sizes."));
body.push(B("Benchmark client runs inside the server pod: results exclude client-to-server network latency."));
body.push(B("Tests up to 120k used --max-model-len 131072; the 240k test used 262144. The context limit does not change per-request latency."));
body.push(B("Single-node cluster: results represent one 4-GPU server. Production should use at least two GPU nodes for availability."));
body.push(note("Source data, manifests and scripts: github.com/ricardolins/akamai-cloud-qwen38-27b (results/, k8s/, scripts/)."));

// ---------- document ----------
const doc = new Document({
  creator: "Akamai Cloud benchmark",
  title: "Qwen3.8-27B on 4x RTX PRO 6000 Blackwell — results",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 44, bold: true, color: C.head, font: FONT }, paragraph: { spacing: { after: 80 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: C.head, font: FONT }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "2E5597", font: FONT }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
      R_({ text: "Qwen3.8-27B on Akamai Cloud — page ", size: 16, color: "7F7F7F" }),
      R_({ children: [PageNumber.CURRENT], size: 16, color: "7F7F7F" })] })] }) },
    children: body,
  }],
});

fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log(`wrote ${OUT} (${buf.length} bytes); tp4 rows=${tp4.length}, tp1 rows=${tp1.length}`); });
