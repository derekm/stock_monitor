#!/usr/bin/env python3
"""
migrate_fundamentals_schema.py — collapse duplicate columns and make the period
basis explicit in fundamentals.parquet.

WHY

The panel accumulated two names for the same concept (`equity` and
`shareholders_equity`), and used a bare-vs-`ttm_` prefix to distinguish a single
quarter from a trailing twelve months. Both are ambiguous at the call site:
`debt` does not say *total*, `shares` does not say *outstanding*, and
`net_income` does not say which period it covers. The `ttm_` prefix also reads as
a variant of the same measure rather than a different period basis.

WHAT THIS DOES

1. COALESCE aliases into the more expressive canonical name, canonical wins:
       equity, stockholders_equity -> shareholders_equity
       assets                      -> total_assets
       cash                        -> cash_and_equivalents
   The alias only fills rows where the canonical is NULL, so no populated value
   is ever replaced.

2. RENAME to an explicit period basis:
       revenue              -> revenue_quarterly       ttm_revenue     -> revenue_ttm
       net_income           -> net_income_quarterly    ttm_net_income  -> net_income_ttm
       operating_income     -> operating_income_quarterly / _ttm
       operating_cash_flow  -> operating_cash_flow_quarterly / _ttm
       capital_expenditure  -> capital_expenditure_quarterly / _ttm

3. DROPS `shares`, which is corrupt legacy data rather than a naming variant.

WHAT IT DELIBERATELY DOES NOT DO

- `ttm_*` and bare columns are NOT merged. Measured median ratio ttm/bare is
  3.992 (net_income) and 4.222 (revenue): the bare column is a single QUARTER and
  `ttm_` is a twelve-month sum. They are different metrics.
- `total_debt` and `total_liabilities` are NOT merged (median ratio 2.515 --
  debt is a subset of liabilities).
- `total_revenue` is NOT dropped: it was NULL in all 311,489 rows for a long
  time (while 5,456 rows carried total_revenue_provenance='reported'), but the v2
  extractor writes that name and has since populated 256 rows. It is coalesced
  into `revenue` -> `revenue_quarterly` so those values survive.

`shares` EVIDENCE (why dropped, not coalesced)

  shares vs shares_outstanding: both set on 4,866 rows, DISAGREE on 3,547.
  FITB 2010-09-30 shares = 7.96e14 -- 796 trillion shares, physically impossible;
  shares_outstanding = 9.05e8 is correct. Of the 73 rows where only `shares` is
  populated, 45 are 0.0 and 1 is 4.9e13, leaving 27 usable values against 3,547
  active disagreements. Coalescing would import corruption.

Idempotent: re-running is a no-op once the canonical names are in place.

Usage:
    python migrate_fundamentals_schema.py --dry-run
    python migrate_fundamentals_schema.py --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
BACKUP_DIR = DATA_DIR / "backfill_backups"

# alias -> canonical (alias fills only where canonical is NULL)
#
# `total_revenue` and `revenue` are the same single-quarter measure under two
# names: the v2 extractor writes BOTH (total_revenue at L775, then revenue as a
# compatibility alias). total_revenue was empty in all 311,489 rows until the v2
# runs populated 256 of them, so it is a live column now -- coalesced, not
# dropped. `net_income_quarterly` already exists with 272 rows that agree
# exactly with `net_income` on every shared row (0 disagreements) and add no
# unique rows, so net_income folds into it rather than renaming onto a clash.
COALESCE = {
    "equity": "shareholders_equity",
    "stockholders_equity": "shareholders_equity",
    "assets": "total_assets",
    "cash": "cash_and_equivalents",
    "total_revenue": "revenue",
    "net_income": "net_income_quarterly",
}

# old -> new, making the period basis explicit.
#
# CRITICAL: the bare-vs-ttm_ prefix did NOT reliably indicate period basis. Verified
# by median ratio ttm_X / X on rows where both were set:
#     net_income          3.992  -> bare really was a QUARTER
#     revenue             4.222  -> bare really was a QUARTER
#     operating_income    ~4     -> bare really was a QUARTER
#     capital_expenditure 1.000  -> bare was ALREADY TTM
#     operating_cash_flow 1.000  -> bare was ALREADY TTM
# The cash-flow pair was mislabelled because the v2 writer did
# `row["capital_expenditure"] = row.get("ttm_capital_expenditure")`. Mapping those
# two to *_quarterly would have put twelve-month sums under a quarterly name --
# exactly the false authority this migration exists to remove. They are coalesced
# into their *_ttm partners instead (they agree at ratio 1.0 where both are set:
# 436 rows for capex, 287 for OCF).
RENAME = {
    "revenue": "revenue_quarterly",
    "ttm_revenue": "revenue_ttm",
    "ttm_net_income": "net_income_ttm",
    "operating_income": "operating_income_quarterly",
    "ttm_operating_income": "operating_income_ttm",
}

# bare name -> its *_ttm partner (bare already held TTM values)
COALESCE_TTM = {
    "operating_cash_flow": "ttm_operating_cash_flow",
    "capital_expenditure": "ttm_capital_expenditure",
}

# after the TTM coalesce, give the survivors their explicit names
RENAME_LATE = {
    "ttm_operating_cash_flow": "operating_cash_flow_ttm",
    "ttm_capital_expenditure": "capital_expenditure_ttm",
}

# corrupt columns removed outright (see module docstring for evidence)
DROP = ["shares"]

# provenance columns that follow their metric's rename
PROV_RENAME = {
    "revenue_provenance": "revenue_quarterly_provenance",
    "net_income_provenance": "net_income_quarterly_provenance",
    "net_income_quarterly_provenance": "net_income_quarterly_provenance",
}


def plan(df: pd.DataFrame) -> dict:
    """What the migration would do to this frame, without doing it."""
    out = {"coalesce": [], "rename": [], "drop": [], "skip": []}
    for alias, canon in list(COALESCE.items()) + list(COALESCE_TTM.items()):
        if alias not in df.columns:
            out["skip"].append(f"{alias} (absent)")
            continue
        if canon not in df.columns:
            out["skip"].append(f"{alias} -> {canon} (canonical absent)")
            continue
        fills = int((df[canon].isna() & df[alias].notna()).sum())
        out["coalesce"].append((alias, canon, fills))
    for src, dst in list(RENAME.items()) + list(RENAME_LATE.items()):
        if src not in df.columns:
            out["skip"].append(f"{src} (already renamed or absent)")
            continue
        if dst in df.columns:
            out["skip"].append(f"{src} -> {dst} (target already exists!)")
            continue
        out["rename"].append((src, dst, int(df[src].notna().sum())))
    for c in DROP:
        if c in df.columns:
            out["drop"].append((c, int(df[c].notna().sum())))
    return out


def migrate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. coalesce aliases (canonical wins; alias only fills NULLs)
    for alias, canon in COALESCE.items():
        if alias in df.columns and canon in df.columns:
            df[canon] = df[canon].where(df[canon].notna(), df[alias])
            df = df.drop(columns=[alias])
        elif alias in df.columns and canon not in df.columns:
            df = df.rename(columns={alias: canon})
    # 2. fold bare cash-flow columns into their ttm_ partners: both already held
    #    TTM values (median ratio 1.000), so the bare name was never quarterly.
    for bare, ttm in COALESCE_TTM.items():
        if bare in df.columns and ttm in df.columns:
            df[ttm] = df[ttm].where(df[ttm].notna(), df[bare])
            df = df.drop(columns=[bare])
        elif bare in df.columns and ttm not in df.columns:
            df = df.rename(columns={bare: ttm})
    # 3. drop corrupt/empty
    for c in DROP:
        if c in df.columns:
            df = df.drop(columns=[c])
    # 4. explicit period basis
    for mapping in (RENAME, RENAME_LATE):
        ren = {s: d for s, d in mapping.items()
               if s in df.columns and d not in df.columns}
        if ren:
            df = df.rename(columns=ren)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only (default)")
    ap.add_argument("--apply", action="store_true", help="write the migrated panel")
    args = ap.parse_args()

    df = pd.read_parquet(FUND)
    print(f"fundamentals.parquet: {len(df):,} rows x {len(df.columns)} cols")
    print()

    p = plan(df)
    print("COALESCE (alias fills only NULL canonical):")
    for alias, canon, fills in p["coalesce"]:
        print(f"  {alias:22} -> {canon:24} fills {fills:6} NULL rows")
    print()
    print("RENAME (explicit period basis):")
    for src, dst, n in p["rename"]:
        print(f"  {src:26} -> {dst:32} ({n:,} values)")
    print()
    print("DROP:")
    for c, n in p["drop"]:
        print(f"  {c:26} ({n:,} non-null)")
    if p["skip"]:
        print()
        print("SKIPPED:")
        for s in p["skip"]:
            print(f"  {s}")

    if not args.apply:
        print()
        print("dry run -- nothing written. Re-run with --apply.")
        return 0

    # Value-preservation check. Must account for CHAINING: total_revenue
    # coalesces into revenue, which is then renamed to revenue_quarterly, and an
    # alias contributes rows the canonical lacked. So the expectation for a final
    # column is "any of its contributing source columns was non-null", and the
    # contributors must be resolved through both maps.
    def _final_name(col: str) -> str:
        seen = set()
        while col not in seen:
            seen.add(col)
            if col in COALESCE:
                col = COALESCE[col]
            elif col in COALESCE_TTM:
                col = COALESCE_TTM[col]
            else:
                break
        return RENAME_LATE.get(RENAME.get(col, col), RENAME.get(col, col))

    contributors: dict[str, list[str]] = {}
    for col in df.columns:
        if col in DROP:
            continue
        final = _final_name(col)
        contributors.setdefault(final, []).append(col)

    before = {
        final: int(df[cols].notna().any(axis=1).sum())
        for final, cols in contributors.items()
        if len(cols) > 1 or cols[0] != final
    }

    out = migrate(df)

    problems = []
    for col, exp in before.items():
        got = int(out[col].notna().sum()) if col in out.columns else -1
        if got != exp:
            problems.append((col, exp, got))
    if problems:
        print()
        print("ABORT -- value count changed:")
        for c, e, g in problems:
            print(f"  {c}: expected {e}, got {g}")
        return 1

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = BACKUP_DIR / f"fundamentals_pre_schema_migration_{stamp}.parquet"
    shutil.copy2(FUND, bk)

    tmp = FUND.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, FUND)

    print()
    print(f"backup : {bk.name}")
    print(f"written: {len(out):,} rows x {len(out.columns)} cols "
          f"({len(df.columns)} -> {len(out.columns)})")
    print("value counts preserved for every migrated column ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
