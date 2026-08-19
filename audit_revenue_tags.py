#!/usr/bin/env python3
"""
audit_revenue_tags.py — find filers whose revenue tag is too NARROW.

THE FAILURE MODE

REV_TAGS is a list of top-line names. If a filer's real top line is not in it, the
extractor still succeeds: it picks whichever listed tag is present and current, even
when that tag is a minor line item. ABR (a mortgage REIT) filed
OperatingLeaseLeaseIncome at $1.51M/quarter while its actual top line,
InterestIncomeOperating, ran at $235M/quarter -- a 155x understatement that surfaced
downstream as fcf_margin values in the hundreds.

The tell is INTERNAL INCONSISTENCY, not an absolute threshold: revenue below
operating cash flow. A company can out-earn its revenue in cash for a quarter or two
(working-capital release, asset sales), but sustained OCF far above revenue means the
denominator is the wrong scope.

WHAT THIS DOES

For each suspect ticker, scan EVERY us-gaap/ifrs-full tag with current quarterly
facts and report any whose median quarterly magnitude materially exceeds the selected
revenue tag's. Tags that recur across many suspects are candidate additions to
REV_TAGS. Nothing is changed -- this only reports.

Usage:
    python audit_revenue_tags.py --limit 400
    python audit_revenue_tags.py --tickers ABR,AGNC,NLY --verbose
    python audit_revenue_tags.py --limit 900 --json revenue_tag_gaps.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent
FUND = DATA_DIR / "fundamentals.parquet"

# a tag must look like an income/revenue measure to be a candidate top line
INCOME_WORDS = re.compile(
    r"Revenue|Sales|InterestIncome|InterestAndDividend|PremiumsEarned|"
    r"RealEstate|LeaseIncome|Fees|Commission|PolicyholderBenefit", re.I)

# never propose these: they are net-of-expense, partial, or not a top line
NEVER = re.compile(
    r"Net$|Expense|Cost|Deferred|Unearned|Receivable|Payable|Liability|Asset|"
    r"Allowance|Impair|Tax|ProForma|Acquiree|PerShare|Percent|Ratio|"
    r"AccruedInterest|Discount|Amortization", re.I)


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[len(xs) // 2]


def quarterly_median(V, tag_data: dict, since: str) -> tuple[float, str]:
    """Median |value| of quarterly facts ending on/after `since`, and the last end."""
    vals, last = [], ""
    for unit_key in ("USD",):
        for e in tag_data.get("units", {}).get(unit_key, []):
            if not e.get("start"):
                continue
            sm = V._span_months(e)
            if sm is None or not (2 <= sm <= 4):
                continue
            end = e.get("end", "")
            if end < since:
                continue
            v = e.get("val")
            if isinstance(v, (int, float)):
                vals.append(abs(float(v)))
                if end > last:
                    last = end
    return median(vals), last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--since", default="2024-01-01",
                    help="only consider quarterly facts ending on/after this")
    ap.add_argument("--factor", type=float, default=3.0,
                    help="flag a tag whose median exceeds revenue's by this factor")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    import edgar_companyfacts_v2 as V

    cm = V.load_cik_map()
    cm.update(V.CIK_OVERRIDES)

    if args.tickers:
        suspects = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        # SUSPECTS: revenue present but well below operating cash flow, sustained.
        f = pl.read_parquet(FUND)
        g = (f.filter(pl.col("revenue_ttm").is_not_null()
                      & (pl.col("revenue_ttm") > 0)
                      & pl.col("operating_cash_flow_ttm").is_not_null())
             .with_columns((pl.col("operating_cash_flow_ttm")
                            / pl.col("revenue_ttm")).alias("ocf_rev")))
        agg = (g.group_by("ticker")
               .agg(pl.col("ocf_rev").median().alias("med"),
                    pl.len().alias("n"))
               .filter((pl.col("med") > 1.5) & (pl.col("n") >= 4))
               .sort("med", descending=True))
        suspects = [t for t in agg["ticker"].to_list() if t in cm][: args.limit]
        print(f"suspects: {agg.height} tickers with median OCF/revenue > 1.5 "
              f"over >=4 quarters; scanning {len(suspects)}")

    hits = collections.Counter()
    per_ticker = {}
    current_rev = set(V_rev_tags(V))

    for t in suspects:
        cik = cm.get(t)
        if not cik:
            continue
        try:
            facts = V._facts(V.fetch_companyfacts(cik))
        except Exception:
            continue
        if not facts:
            continue

        picked = V._pick_tag(facts, list(current_rev)) or ""
        pick_med = 0.0
        if picked and picked in facts:
            pick_med, _ = quarterly_median(V, facts[picked], args.since)

        bigger = []
        for tag, data in facts.items():
            if tag in current_rev:
                continue
            if not INCOME_WORDS.search(tag) or NEVER.search(tag):
                continue
            med, last = quarterly_median(V, data, args.since)
            if med <= 0 or not last:
                continue
            if pick_med > 0 and med < pick_med * args.factor:
                continue
            bigger.append((med, tag, last))
        if not bigger:
            continue
        bigger.sort(reverse=True)
        per_ticker[t] = {"picked": picked, "picked_median": pick_med,
                         "candidates": [(tag, m, l) for m, tag, l in bigger[:4]]}
        for _, tag, _ in bigger[:3]:
            hits[tag] += 1
        if args.verbose or len(per_ticker) <= 25:
            top = bigger[0]
            print(f"  {t:8} picked {picked[:38]:38} ${pick_med/1e6:9,.2f}M"
                  f"  <- candidate {top[1][:40]:40} ${top[0]/1e6:9,.2f}M")

    print()
    print(f"tickers with a materially larger unlisted income tag: {len(per_ticker)}")
    print()
    print("CANDIDATE TAGS by how many suspects they would fix:")
    for tag, n in hits.most_common(20):
        print(f"  {n:4}  {tag}")

    if args.json:
        Path(args.json).write_text(json.dumps(per_ticker, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


def V_rev_tags(V) -> list[str]:
    """REV_TAGS as the extractor defines it, read from the source (it is a local)."""
    import ast

    src = (DATA_DIR / "edgar_companyfacts_v2.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "REV_TAGS":
                    return [e.value for e in node.value.elts
                            if isinstance(e, ast.Constant)]
    raise RuntimeError("REV_TAGS not found")


if __name__ == "__main__":
    raise SystemExit(main())
