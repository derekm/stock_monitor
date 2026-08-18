#!/usr/bin/env python3
"""
backfill_edgar.py — Real decades-long point-in-time fundamentals from SEC EDGAR.

Uses edgar_companyfacts_v2.py for extraction (with quarterly differencing, FCF proxy,
and M&A data). Overwrites existing rows unless source is edgar_v2 or html_10q.
"""

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import requests

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
UA = {"User-Agent": "personal-research derek.moore@example.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Overrides: SEC's ticker->CIK map is sometimes stale/wrong.
CIK_OVERRIDES = {
    "XOM": "0000034088",
    "AEP": "0000004904",
    "SATS": "0001415404",
    "SPR": "0001364885",
}
NO_COMPANYFACTS = {"BAYRY"}

# M&A tags to extract
MA_TAGS = [
    'BusinessCombinationConsiderationTransferred',
    'BusinessCombinationConsiderationTransferredEquityInterestsIssuedAndIssuable',
    'BusinessAcquisitionPurchasePriceAllocationGoodwillAmount',
    'BusinessAcquisitionPurchasePriceAllocationAssetsAcquiredLiabilitiesAssumedNet',
    'BusinessCombinationRecognizedIdentifiableAssetsAcquiredGoodwillAndLiabilitiesAssumedNet',
    'BusinessCombinationContingentConsiderationLiabilityCurrent',
    'BusinessCombinationContingentConsiderationLiabilityNoncurrent',
    'PaymentsOfMergerRelatedCostsFinancingActivities',
    'StockIssuedDuringPeriodValueAcquisitions',
    'NoncashOrPartNoncashAcquisitionFixedAssetsAcquired1',
    'GoodwillPurchaseAccountingAdjustments',
]


def load_cik_map() -> dict:
    """Local cik_ticker_map.json wins; live SEC map fills gaps; overrides last."""
    out = {}
    local = DATA_DIR / "cik_ticker_map.json"
    if local.exists():
        with open(local) as f:
            raw = json.load(f)
        for t, c in raw.items():
            cs = str(c).zfill(10)
            if cs.isdigit() and len(cs) == 10:
                out[str(t).upper()] = cs
    try:
        r = requests.get(TICKERS_URL, headers=UA, timeout=30)
        r.raise_for_status()
        for row in r.json().values():
            t = str(row["ticker"]).upper()
            cik = str(row["cik_str"]).zfill(10)
            if t not in out:
                out[t] = cik
    except Exception as e:
        print(f"  live SEC ticker map failed ({e}); using local map only ({len(out)})")
    return out


def fetch_ma_data(cik: str) -> list[dict]:
    """Fetch M&A-related XBRL facts from SEC EDGAR companyfacts."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200:
            return []
        d = r.json()
        facts = d.get("facts", {}).get("us-gaap", {})
        
        ma_records = []
        for tag in MA_TAGS:
            if tag not in facts:
                continue
            units = facts[tag].get("units", {}).get("USD", [])
            for entry in units:
                val = entry.get("val", 0)
                if val and abs(val) > 1_000_000:
                    ma_records.append({
                        "tag": tag,
                        "end": entry.get("end"),
                        "value": val,
                        "frame": entry.get("frame", ""),
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                    })
        return ma_records
    except Exception:
        return []


def fetch_and_build(ticker: str, cik: str, px: dict[str, pd.Series]) -> list[dict]:
    """Fetch companyfacts and build rows using edgar_v2 logic."""
    from edgar_companyfacts_v2 import extract_raw_financials, compute_quarterly_fundamentals
    
    financials = extract_raw_financials(cik)
    if financials is None:
        return []
    
    rows = compute_quarterly_fundamentals(financials, ticker, px)
    
    # Also fetch M&A data
    ma_data = fetch_ma_data(cik)
    
    # Enrich rows with M&A flags
    if ma_data:
        ma_dates = set()
        for ma in ma_data:
            ma_dates.add(ma.get("end", "")[:7])
        
        for row in rows:
            row_date = str(row.get("as_of_date", ""))[:7]
            if row_date in ma_dates:
                row["has_ma_activity"] = True
                row["ma_source"] = "companyfacts"
    
    return rows


FLUSH_EVERY = 40
PROTECTED_SOURCES = {"edgar_v2", "html_10q"}

# Columns where we store prior estimates for future quarters
ESTIMATE_COLS = [
    "total_revenue", "operating_income", "net_income",
    "free_cash_flow", "total_assets", "total_debt",
    "shareholders_equity", "cash_and_equivalents",
    "total_liabilities", "capital_expenditure",
    "ebitda", "gross_profit", "interest_expense",
    "operating_cash_flow", "shares_outstanding",
]


def merge_into_fundamentals(new_rows: list[dict], force: bool = False) -> int:
    """Additive merge + parquet write. Safe to call mid-run.

    Future-quarter EDGAR estimates are preserved in prior_estimate_* columns
    for the most recent actual quarter.

    force=True lets this batch overwrite PROTECTED_SOURCES rows. Needed only to
    push a CORRECTION through: once a row is stamped edgar_v2 it is protected, so
    a bug fix in the extractor cannot otherwise replace values it already wrote.
    Off by default -- protection is the norm.
    """
    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows)
    new_df["source"] = "edgar_v2"
    existing = pd.read_parquet(FUND) if FUND.exists() else pd.DataFrame()
    from update_fundamentals import _as_date
    new_df["as_of_date"] = new_df["as_of_date"].map(_as_date)

    # Split into actual (≤ today) and future (> today) rows
    today = date.today()
    actual_rows = new_df[new_df["as_of_date"] <= today].copy()
    future_rows = new_df[new_df["as_of_date"] > today].copy()

    # Store future estimates as prior_estimate_* on the latest actual quarter per ticker
    if len(future_rows) > 0 and len(actual_rows) > 0:
        for ticker in future_rows["ticker"].unique():
            ticker_future = future_rows[future_rows["ticker"] == ticker]
            ticker_actual = actual_rows[actual_rows["ticker"] == ticker]
            if len(ticker_actual) == 0:
                continue
            # Latest actual quarter for this ticker
            latest_actual = ticker_actual.loc[ticker_actual["as_of_date"].idxmax()]
            latest_idx = latest_actual.name
            # Write future estimates as prior_estimate_* columns
            for col in ESTIMATE_COLS:
                if col in ticker_future.columns:
                    # Take the nearest future quarter estimate
                    est_val = ticker_future[col].iloc[0]
                    if pd.notna(est_val):
                        actual_rows.at[latest_idx, f"prior_estimate_{col}"] = est_val

    # Merge actual rows into fundamentals
    new_df = actual_rows  # Only merge actual rows
    idx = ["ticker", "as_of_date"]
    if len(existing):
        ex = existing.copy()
        nd = new_df.copy()
        for df in (ex, nd):
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype("float64")
        # Merge with suffixes
        merged = ex.merge(nd, on=idx, how="left", suffixes=("_old", "_new"))
        # Start with old columns (protected sources keep their values)
        result_cols = {}
        for col in ex.columns:
            if col in idx:
                result_cols[col] = merged[col]
            else:
                old_c = f"{col}_old"
                if old_c in merged.columns:
                    result_cols[col] = merged[old_c]
                else:
                    result_cols[col] = merged[col]
        # Overwrite non-protected columns with new values.
        #
        # CRITICAL: only rows present in THIS batch may change. The previous
        # implementation did `vals = merged[new_col]; vals[~overwrite_mask] = nan`,
        # which set every row absent from the batch to NaN for every column the
        # batch carried. With per-ticker flushing that wiped the whole panel one
        # flush at a time (proved in a sandbox: merging ZZTEST 2024-03-31 took
        # 2024-09-30 net_income from 120.0 to NaN). Now the old column is the
        # base and we assign only into the masked subset, so untouched rows keep
        # their values bit-for-bit.
        overlap_mask = merged["source_new"].notna() if "source_new" in merged.columns else merged.filter(regex="_new$").notna().any(axis=1)
        if overlap_mask.any():
            protected_mask = merged["source_old"].isin(PROTECTED_SOURCES)
            if force:
                protected_mask = protected_mask & False   # honour every new value
            overwrite_mask = overlap_mask & ~protected_mask
            for col in nd.columns:
                if col in idx or col == "source":
                    continue
                new_col = f"{col}_new"
                if new_col not in merged.columns:
                    continue
                base = result_cols[col].copy() if col in result_cols else merged.get(f"{col}_old")
                if base is None:
                    continue
                new_s = merged[new_col]
                # assign only where this batch supplies a value AND the row is
                # not protected; NaN in the batch must not erase a real value
                apply = overwrite_mask & new_s.notna()
                if apply.any():
                    base = base.copy()
                    base.loc[apply] = new_s.loc[apply].to_numpy()
                result_cols[col] = base
            # Restamp provenance on rows this batch actually updated, so the row
            # reports the extractor that produced it and becomes protected on
            # subsequent runs (the old code left them as source='edgar', which
            # made the merge non-idempotent: a re-run changed 999 -> 777).
            if "source" in result_cols:
                src = result_cols["source"].copy()
                src.loc[overwrite_mask] = "edgar_v2"
                result_cols["source"] = src
            # Fill missing in protected columns
            remaining_mask = overlap_mask & protected_mask
            FILL_COLS = [
                "market_cap", "market_cap_b", "total_assets", "total_assets_b",
                "pb_ratio", "mktcap_to_assets", "ev_ebitda", "roe", "roic",
                "debt_to_equity", "shares_outstanding", "interest_coverage",
                "earnings_stability", "total_revenue", "operating_income", "net_income",
                "free_cash_flow", "operating_cash_flow", "capital_expenditure",
            ]
            for c in FILL_COLS:
                old_col, new_col = f"{c}_old", f"{c}_new"
                if old_col not in merged.columns or new_col not in merged.columns:
                    continue
                try:
                    old_s = merged[old_col]
                    new_s = merged[new_col]
                    fill = remaining_mask & old_s.isna() & new_s.notna()
                    if fill.any():
                        result_cols[c] = result_cols[c].copy()
                        result_cols[c].loc[fill] = new_s.loc[fill].to_numpy()
                except Exception:
                    continue
        # Build combined: existing (updated) + brand new rows
        brand_new = new_df[~new_df.set_index(idx).index.isin(ex.set_index(idx).index)].copy()
        combined = pd.DataFrame(result_cols)
        if len(brand_new):
            combined = pd.concat([combined, brand_new], ignore_index=True)
    else:
        combined = new_df
    if "last_updated" in combined.columns:
        combined["last_updated"] = pd.to_datetime(combined["last_updated"], errors="coerce")
    combined = combined.sort_values(["ticker", "as_of_date"]).drop_duplicates(
        subset=["ticker", "as_of_date"], keep="last"
    )
    for col in combined.columns:
        if pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("float64")
    _atomic_write_parquet(combined, FUND)
    return len(combined)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via temp file + os.replace so an interrupt cannot truncate `path`.

    A direct df.to_parquet(path) leaves a partially-written file if the process
    dies mid-write. That is how a 40-ticker run once took fundamentals.parquet
    from 33.5 MB to 4.6 MB and required restoring from a dated backup. os.replace
    is atomic on the same filesystem, so the original file survives any crash.

    A sanity floor guards the other half of that incident: refuse to shrink the
    panel by more than 20% in one write, since every legitimate merge here is
    additive.
    """
    if path.exists():
        try:
            prev_rows = pq.ParquetFile(path).metadata.num_rows
            if prev_rows > 100 and len(df) < prev_rows * 0.8:
                raise RuntimeError(
                    f"refusing to write {len(df):,} rows over {prev_rows:,} existing "
                    f"({len(df)/prev_rows:.1%}); merges here are additive, so this "
                    "indicates dropped data. Inspect before overwriting."
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # unreadable metadata: fall through to the normal write
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _backup_fundamentals() -> Path:
    """Timestamped backup before any destructive rewrite."""
    from datetime import datetime
    bdir = DATA_DIR / "backfill_backups"
    bdir.mkdir(exist_ok=True)
    dest = bdir / f"fundamentals_backup_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    import shutil
    shutil.copy2(FUND, dest)
    print(f"  backup -> {dest.name}")
    return dest


def purge_test_tickers() -> int:
    """Drop smoke-test ticker rows (TEST*) left behind by merge tests."""
    f = pd.read_parquet(FUND)
    mask = f["ticker"].astype(str).str.fullmatch(r"TEST[A-Z0-9]*")
    n = int(mask.sum())
    print(f"purge-test-tickers: {n} rows across {f.loc[mask, 'ticker'].nunique()} tickers")
    if n == 0:
        return 0
    _backup_fundamentals()
    out = f[~mask].copy()
    out.to_parquet(FUND, index=False)
    print(f"  {len(f)} -> {len(out)} rows")
    return n


def migrate_future_estimates() -> int:
    """Fold pre-existing future-dated rows into prior_estimate_* columns.

    Historical rows with as_of_date > today were written before the estimate
    split existed. For each ticker, the nearest future quarter's values are
    copied onto that ticker's latest actual quarter as prior_estimate_<col>,
    then the future rows are removed from the time series.
    """
    f = pd.read_parquet(FUND)
    d = pd.to_datetime(f["as_of_date"], errors="coerce").dt.date
    today = date.today()
    fut_mask = d > today
    n_fut = int(fut_mask.sum())
    print(f"migrate-future-estimates: {n_fut} future rows, {f.loc[fut_mask, 'ticker'].nunique()} tickers")
    if n_fut == 0:
        return 0

    _backup_fundamentals()
    f["_d"] = d
    future = f[fut_mask]
    actual = f[~fut_mask].copy()

    moved = 0
    for ticker, grp in future.groupby("ticker"):
        act = actual[actual["ticker"] == ticker]
        if act.empty:
            # No actual history to attach the estimate to — nothing to preserve.
            continue
        latest_idx = act["_d"].idxmax()
        nearest = grp.sort_values("_d").iloc[0]
        for col in ESTIMATE_COLS:
            if col not in grp.columns:
                continue
            val = nearest[col]
            if pd.notna(val):
                actual.at[latest_idx, f"prior_estimate_{col}"] = val
                moved += 1

    actual = actual.drop(columns=["_d"])
    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]):
            actual[col] = pd.to_numeric(actual[col], errors="coerce").astype("float64")
    actual.to_parquet(FUND, index=False)
    print(f"  {len(f)} -> {len(actual)} rows; {moved} estimate values preserved")
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=None, help="Comma-separated subset")
    ap.add_argument("--max-tickers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Fetch CIKs and report coverage, write nothing")
    ap.add_argument("--quarantine", action="store_true", help="Use bad CIK quarantine to skip invalid CIKs")
    ap.add_argument("--validate-ciks", action="store_true", help="Validate CIKs via SEC submissions and quarantine bad ones")
    ap.add_argument("--clear-quarantine", action="store_true", help="Clear bad CIK quarantine")
    ap.add_argument("--purge-test-tickers", action="store_true",
                    help="Remove smoke-test ticker rows (TEST*) from fundamentals.parquet")
    ap.add_argument("--migrate-future-estimates", action="store_true",
                    help="Fold pre-existing future-dated rows into prior_estimate_* on the latest actual quarter")
    args = ap.parse_args()

    if args.purge_test_tickers:
        purge_test_tickers()
        return

    if args.migrate_future_estimates:
        migrate_future_estimates()
        return

    # Handle clear-quarantine
    if args.clear_quarantine:
        quarantine_path = Path("backfill_checkpoints/bad_ciks.json")
        if quarantine_path.exists():
            quarantine_path.unlink()
            print("Cleared bad CIK quarantine")
        else:
            print("No quarantine file found")
        return

    # Load bad CIK quarantine if requested
    bad_cik_set = set()
    if args.quarantine:
        quarantine_path = Path("backfill_checkpoints/bad_ciks.json")
        if quarantine_path.exists():
            with open(quarantine_path) as f:
                quarantine = json.load(f)
            bad_cik_set = set(quarantine.get("bad_ciks", {}).keys())
            print(f"Quarantine: skipping {len(bad_cik_set)} tickers with bad CIKs")

    # Handle validate-ciks
    if args.validate_ciks:
        print("Validating CIKs via SEC submissions API...")
        from update_fundamentals import universe_tickers
        tickers = universe_tickers()
        if not tickers:
            print("No universe found")
            return
        
        cik_map = load_cik_map()
        cik_map.update(CIK_OVERRIDES)
        
        bad_count = 0
        good_count = 0
        bad_ciks = {}
        
        for i, ticker in enumerate(tickers):
            cik = cik_map.get(ticker)
            if not cik:
                bad_ciks[ticker] = {"cik": "NONE", "reason": "NO_CIK_IN_MAP"}
                bad_count += 1
                continue
            
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            try:
                r = requests.get(url, headers=UA, timeout=10)
                if r.status_code == 404:
                    bad_ciks[ticker] = {"cik": cik, "reason": "SEC_404"}
                    bad_count += 1
                elif r.status_code == 200:
                    name = (r.json().get("name") or "").upper()
                    tk = ticker.upper().split("-")[0]
                    if len(tk) >= 2 and tk not in name and not any(
                        tok.startswith(tk[:3]) for tok in name.replace(",", " ").split() if len(tok) >= 3
                    ):
                        bad_ciks[ticker] = {
                            "cik": cik,
                            "reason": "ENTITY_MISMATCH",
                            "suggestion": r.json().get("name", ""),
                        }
                        bad_count += 1
                    else:
                        good_count += 1
                else:
                    good_count += 1
                time.sleep(0.05)
                
                if i % 100 == 0:
                    print(f"  {i}/{len(tickers)}...")
            except Exception as e:
                bad_ciks[ticker] = {"cik": cik, "reason": str(e)}
                bad_count += 1
        
        quarantine = {
            "bad_ciks": bad_ciks,
            "metadata": {
                "total_checked": len(tickers),
                "bad_count": bad_count,
                "good_count": good_count
            }
        }
        with open("backfill_checkpoints/bad_ciks.json", "w") as f:
            json.dump(quarantine, f, indent=2)
        
        print(f"\nValidation complete:")
        print(f"  Good: {good_count}")
        print(f"  Bad: {bad_count}")
        print(f"Saved to backfill_checkpoints/bad_ciks.json")
        return

    from update_fundamentals import universe_tickers
    tickers = universe_tickers()
    if not tickers:
        stocks = pd.read_parquet(STOCKS_FILE) if STOCKS_FILE.exists() else pd.DataFrame()
        if stocks.empty or "ticker" not in stocks.columns:
            raise SystemExit("no universe")
        tickers = sorted(stocks["ticker"].astype(str).str.upper().unique().tolist())
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"EDGAR universe: {len(tickers)} tickers")

    print("Loading SEC ticker→CIK map...")
    cik_map = load_cik_map()
    cik_map.update(CIK_OVERRIDES)
    matched = [(t, cik_map[t]) for t in tickers if t in cik_map and t not in bad_cik_set]
    unmatched = [t for t in tickers if t not in cik_map]
    print(f"  {len(matched)}/{len(tickers)} tickers have a CIK")
    if unmatched:
        print(f"  no CIK ({len(unmatched)}): {', '.join(unmatched[:20])}...")
    matched = [(t, c) for t, c in matched if t not in NO_COMPANYFACTS]
    if args.dry_run:
        return

    # price series for mktcap
    from analytics_common import load_adj_prices_pandas
    prices = load_adj_prices_pandas(tickers=tickers)
    px = {tk: g.set_index("date")["close"] for tk, g in prices.groupby("ticker")}

    all_rows: list[dict] = []
    ok = 0
    newly_bad = 0
    processed = 0
    flushed = 0
    for t, cik in matched:
        try:
            rows = fetch_and_build(t, cik, px)
            if rows:
                all_rows.extend(rows)
                ok += 1
                print(f"  {t}: {len(rows)} quarters (cik={cik})")
            time.sleep(0.12)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                bad_cik_set.add(t)
                newly_bad += 1
                print(f"  !! {t}: 404 (quarantined)")
            else:
                print(f"  !! {t}: {e}")
            time.sleep(0.12)
        except Exception as e:
            print(f"  !! {t}: {e}")
            time.sleep(0.12)
        processed += 1
        if processed % FLUSH_EVERY == 0 and all_rows:
            n = merge_into_fundamentals(all_rows)
            flushed += len({r.get("ticker") for r in all_rows})
            print(f"  flushed → {n} fundamentals rows")
            all_rows = []
    if newly_bad:
        quarantine = {"bad_ciks": {t: {"cik": cik_map[t], "reason": "SEC_404"} for t in bad_cik_set if t in cik_map}}
        Path("backfill_checkpoints").mkdir(exist_ok=True)
        with open("backfill_checkpoints/bad_ciks.json", "w") as f:
            json.dump(quarantine, f, indent=2)
        print(f"Quarantined {newly_bad} new bad CIKs (total: {len(bad_cik_set)})")
    if all_rows:
        n = merge_into_fundamentals(all_rows)
        print(f"EDGAR v2 final: {n} rows after last flush ({ok} tickers ok)")
    elif flushed:
        print(f"EDGAR v2 done: {ok} tickers ok, already flushed")
    else:
        print("No rows fetched.")


if __name__ == "__main__":
    main()
