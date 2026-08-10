#!/usr/bin/env python3
"""Add info-ic popover icons to dashboard section headers."""
import re
from pathlib import Path

p = Path("index.html")
c = p.read_text(encoding="utf-8")

# (heading text → glossary key) pairs. Only applies when the h2/h3 does NOT
# already contain an info-ic span.
HEADERS = [
    ("<h3>Decision memos (portfolio inclusion)</h3>", "dualp"),
    ("<h3>Value trifecta (EV/EBITDA≤9, P/B≤1.5, MktCap/Assets≤0.5)</h3>", "trifecta"),
    ("<h3>Buy candidates (gates + momentum + factors)</h3>", "oosc"),
    ("<h3>Portfolio valuation rank</h3>", "pb"),
    ("<h3>Holdings</h3>", "maxdd"),
    ("<h3>Asset ↔ sector correlation (home vs market)</h3>", "avgcorr"),
    ("<h3>Shadow book — paper-trade replay (FIFO lots, kill switches)</h3>", "shadow"),
    ("<h3>Lowest EV/EBITDA (top 20)</h3>", "evEbitda"),
    ("<h3>Lowest P/B (top 15)</h3>", "pb"),
    ("<h3>Implied cost of capital (Ohlson-Rueangsuwan 2026 · r = 2·ROE/(P/B+1))</h3>", "impliedR"),
    ("<h3>HMM regime conditional correlations</h3>", "hmm"),
    ("<h3>RPT regime models — best config per ticker</h3>", "rpt"),
    ("<h3>RPT vs IBM regime comparison</h3>", "rptVsIbm"),
    ("<h2>Signals &amp; Alpha — technicals, options, revisions, filings, blended model</h2>", "oosc"),
    ("<h3>Options skew (ATM IV, skew, put/call)</h3>", "skew"),
    ("<h3>Per-name fragility screen", "tailIdx"),
    ("<h3>Tail index (Hill α) — variance is nearly meaningless below α≈3</h3>", "tailIdx"),
    ("<h3>Portfolio tail vs Gaussian</h3>", "tailIdx"),
    ("<h3>Ergodicity / ruin (portfolio)</h3>", "ergodic"),
    ("<h3>Barbell structure / convexity</h3>", "barbell"),
    ("<h3>Tail dependence (upper/lower)</h3>", "tailDep"),
    ("<h3>Gap risk — the risk that arrives overnight</h3>", "gapRisk"),
    ("<h2>Rolling / walk-forward", "oosc"),
    ("<h3>Pair Engine — walk-forward OOS (net of costs)</h3>", "pairs"),
    ("<h3>Cross-Section — L/S vs equal-weight long</h3>", "crossSec"),
    ("<h3>Signal Aggregator — OOS IC weights</h3>", "oosc"),
    ("<h2>Sprint Engines — signal aggregation, pairs, cross-section, earnings</h2>", "sprint"),
    ("<h2>Single Stock Analytics — peer signals, trends, recovery</h2>", "peer"),
]

n = 0
for hdr, key in HEADERS:
    # Find the exact header (or prefix for the two fuzzy ones)
    idx = c.find(hdr)
    if idx == -1:
        # try a looser match
        pat = re.compile(re.escape(hdr))
        m = pat.search(c)
        if m:
            idx = m.start()
    if idx == -1:
        print(f"  MISS {hdr[:60]}")
        continue
    # Find end of this h-tag
    end = c.find("</h3>", idx)
    if end == -1:
        end = c.find("</h2>", idx)
    if end == -1:
        print(f"  NO-CLOSE {hdr[:60]}")
        continue
    seg = c[idx:end]
    if "info-ic" in seg:
        print(f"  skip (already has icon) {hdr[:50]}")
        continue
    # Replace `</h3>` with ` <span class="info-ic" data-gloss="{key}">i</span></h3>`
    c = c[:end] + f' <span class="info-ic" data-gloss="{key}">i</span>' + c[end:]
    n += 1
    print(f"  ADD {key} -> {hdr[:60]}")

p.write_text(c, encoding="utf-8")
print(f"\nAdded {n} info icons")
