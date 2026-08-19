#!/usr/bin/env python3
"""
reconcile_companyfacts_html.py — cross-check XBRL companyfacts against 10-Q HTML.

edgar_v2 (XBRL companyfacts) and html_10q (scraped 10-Q HTML) are PEERS at rank 110
in update_fundamentals.SOURCE_RANK: both are the company's own filing for the same
period, read two different ways. html_10q covers line items companyfacts states no
explicit fact for.

Equal rank means either may overwrite the other, so the last batch to run wins. That
is only safe if a DISAGREEMENT between them is surfaced rather than resolved by
ranking. When both sources carry a value for the same (ticker, period, field) and
they differ beyond tolerance, this WARNS LOUDLY and picks no winner: two readings of
one filing that disagree mean one reader is wrong, and the extractor is what needs
fixing.

TOLERANCE

Small differences are expected: HTML tables round to millions, XBRL carries units,
and a restatement can shift a figure between filings. Default is 1% relative with an
absolute floor so rounding noise does not drown real breaks. A SIGN FLIP always
counts, however small -- the two readers use opposite conventions for capex and
cash-flow items.

USAGE
    python reconcile_companyfacts_html.py --tickers AAPL,MSFT --max-quarters 8
    python reconcile_companyfacts_html.py --from-panel --limit 50
    python reconcile_companyfacts_html.py --tickers AAPL --json out.json

Exits 2 when any disagreement is found, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"

# Fields both paths emit under canonical names. Only these are comparable; the XBRL
# path additionally derives ratios the HTML path never sees.
COMPARE = [
    "revenue_quarterly",
    "net_income_quarterly",
    "operating_income_quarterly",
    "operating_cash_flow_ttm",
    "capital_expenditure_ttm",
    "free_cash_flow",
    "total_assets",
    "shareholders_equity",
    "total_debt",
    "cash_and_equivalents",
]

REL_TOL = 0.01        # 1%
ABS_FLOOR = 50_000.0  # ignore differences under $50k -- HTML rounds to millions


def _norm_date(x):
    if x is None:
        return None
    try:
        return pd.Timestamp(x).date()
    except Exception:
        return None


def disagrees(a: float, b: float) -> tuple[bool, float]:
    """Do two values differ beyond tolerance? Returns (flag, relative diff)."""
    if a is None or b is None:
        return False, 0.0
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False, 0.0
    if pd.isna(a) or pd.isna(b):
        return False, 0.0
    diff = abs(a - b)
    if diff <= ABS_FLOOR:
        return False, 0.0
    denom = max(abs(a), abs(b))
    if denom == 0:
        return False, 0.0
    rel = diff / denom
    # A SIGN FLIP is always a real break, however small: capex and cash-flow items
    # differ in sign convention between the two readers and that must not be
    # averaged away as rounding.
    if (a < 0) != (b < 0):
        return True, rel
    return rel > REL_TOL, rel


def reconcile_ticker(ticker: str, cik_map: dict, max_quarters: int) -> dict:
    import edgar_companyfacts_v2 as V
    import edgar_html_10q as H

    cik = cik_map.get(ticker)
    if not cik:
        return {"ticker": ticker, "error": "no CIK"}

    # XBRL path
    try:
        fin = V.extract_raw_financials(cik)
        xbrl_rows = V.compute_quarterly_fundamentals(fin, ticker) if fin else []
    except Exception as e:                                    # noqa: BLE001
        return {"ticker": ticker, "error": f"companyfacts: {type(e).__name__}: {e}"}

    # HTML path
    try:
        html_rows = H.extract_quarterly_from_html(cik, ticker,
                                                  max_quarters=max_quarters)
    except Exception as e:                                    # noqa: BLE001
        return {"ticker": ticker, "error": f"html_10q: {type(e).__name__}: {e}"}

    xb = {_norm_date(r.get("as_of_date")): r for r in xbrl_rows}
    ht = {_norm_date(r.get("report_date")): r for r in html_rows}
    shared = sorted(d for d in set(xb) & set(ht) if d is not None)

    out = {
        "ticker": ticker,
        "xbrl_quarters": len(xb),
        "html_quarters": len(ht),
        "shared_quarters": len(shared),
        "breaks": [],
        "xbrl_only_fields": {},
        "html_only_fields": {},
    }

    for d in shared:
        x, h = xb[d], ht[d]
        for col in COMPARE:
            xv, hv = x.get(col), h.get(col)
            x_has = xv is not None and not pd.isna(xv)
            h_has = hv is not None and not pd.isna(hv)
            if x_has and h_has:
                bad, rel = disagrees(xv, hv)
                if bad:
                    out["breaks"].append({
                        "date": str(d), "field": col,
                        "companyfacts": float(xv), "html_10q": float(hv),
                        "rel_diff": round(rel, 4),
                    })
            elif x_has and not h_has:
                out["xbrl_only_fields"][col] = out["xbrl_only_fields"].get(col, 0) + 1
            elif h_has and not x_has:
                # These are the rows html_10q exists FOR: companyfacts had nothing.
                out["html_only_fields"][col] = out["html_only_fields"].get(col, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="comma-separated")
    ap.add_argument("--from-panel", action="store_true",
                    help="take tickers that already have edgar_v2 rows")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-quarters", type=int, default=8)
    ap.add_argument("--json", help="write full findings to this path")
    args = ap.parse_args()

    import edgar_companyfacts_v2 as V
    cik_map = V.load_cik_map()
    cik_map.update(V.CIK_OVERRIDES)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.from_panel:
        import polars as pl
        f = pl.read_parquet(FUND, columns=["ticker", "source"])
        t = (f.filter(pl.col("source") == "edgar_v2")["ticker"]
             .unique().sort().to_list())
        tickers = t[: args.limit]
    else:
        print("pass --tickers or --from-panel")
        return 1

    print(f"reconciling {len(tickers)} ticker(s), "
          f"tolerance {REL_TOL:.0%} rel / ${ABS_FLOOR:,.0f} abs")
    print()

    results = []
    total_breaks = 0
    for t in tickers:
        r = reconcile_ticker(t, cik_map, args.max_quarters)
        results.append(r)
        if r.get("error"):
            print(f"  {t:8} !! {r['error']}")
            continue
        nb = len(r["breaks"])
        total_breaks += nb
        flag = "  " if nb == 0 else "!!"
        print(f"{flag} {t:8} shared={r['shared_quarters']:2}  "
              f"xbrl={r['xbrl_quarters']:2} html={r['html_quarters']:2}  "
              f"breaks={nb}")
        for b in r["breaks"][:6]:
            print(f"       DISAGREE {b['date']} {b['field']:26} "
                  f"companyfacts={b['companyfacts']:,.0f} "
                  f"html={b['html_10q']:,.0f} ({b['rel_diff']:.1%})")
        if r["html_only_fields"]:
            print(f"       html-only (companyfacts had nothing): "
                  f"{r['html_only_fields']}")

    print()
    print("=" * 72)
    ok = [r for r in results if not r.get("error")]
    print(f"tickers reconciled : {len(ok)}")
    print(f"total disagreements: {total_breaks}")
    if total_breaks:
        by_field: dict[str, int] = {}
        for r in ok:
            for b in r["breaks"]:
                by_field[b["field"]] = by_field.get(b["field"], 0) + 1
        print()
        print("WARNING -- companyfacts and 10-Q HTML disagree. Fields by frequency:")
        for f_, n in sorted(by_field.items(), key=lambda kv: -kv[1]):
            print(f"  {f_:28} {n}")
        print()
        print("These are two readings of the SAME filing, so a disagreement means")
        print("one reader is wrong. Investigate the field, do not average them.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print()
        print(f"full findings -> {args.json}")

    # non-zero exit when disagreements exist, so this can gate a pipeline
    return 2 if total_breaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
