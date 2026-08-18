#!/usr/bin/env python3
"""
snapshot_history.py — append point-in-time history for snapshot-only tables.

WHY THIS EXISTS

`buy_candidates_oos.py` could only validate 7 of ~13 scorer components. The rest
(`composite`, `fragile_veto`, `skew`, ...) live in tables that are OVERWRITTEN on
every run and carry no date column, so there is no way to know what the score
saw on a past date. Backtesting them would require lookahead, which is why they
were reported as EXCLUDED rather than silently zeroed.

This module adds an append-only `*_history.parquet` beside each snapshot. Once a
few months of history accumulate, those components become testable on the same
walk-forward harness as the rest.

DESIGN

- Append-only. A run for an `as_of_date` that already exists REPLACES just that
  date's rows (idempotent re-runs), never the whole file.
- `as_of_date` is a real `datetime.date`, matching the repo's DATE-native
  convention (see docs/SCHEMAS.md) so it round-trips as date32[day] in parquet.
- The snapshot write is untouched. Downstream readers keep working; history is
  purely additive.
- Never silently swallows a write failure: the caller's snapshot is written
  first, so a history bug cannot corrupt the primary artifact.

USAGE

    from snapshot_history import append_history
    append_history(df, "signal_aggregator_scores")   # -> signal_aggregator_scores_history.parquet
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def history_path(name: str) -> Path:
    """Path of the history file for a snapshot table name (no extension)."""
    return DATA_DIR / f"{name}_history.parquet"


def _normalize_as_of(value) -> _date:
    """Coerce to a plain datetime.date (DATE-native; never a Timestamp)."""
    if value is None:
        return _date.today()
    if isinstance(value, _date) and not isinstance(value, pd.Timestamp):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"as_of_date could not be parsed: {value!r}")
    return ts.date()


def append_history(df: pd.DataFrame, name: str, as_of=None,
                   quiet: bool = False) -> Path | None:
    """Append `df` to `<name>_history.parquet` stamped with `as_of` (default today).

    Re-running for the same as_of_date replaces only that date's rows, so daily
    automation is idempotent. Returns the history path, or None if `df` is empty.
    """
    if df is None or len(df) == 0:
        if not quiet:
            print(f"  history: nothing to append for {name}")
        return None

    stamp = _normalize_as_of(as_of)
    new = df.copy()
    # Drop a pre-existing as_of_date so the caller cannot double-stamp.
    new = new.drop(columns=[c for c in ("as_of_date",) if c in new.columns])
    new.insert(0, "as_of_date", stamp)

    path = history_path(name)
    if path.exists():
        try:
            old = pd.read_parquet(path)
        except Exception as exc:  # corrupt/partial file must not kill the run
            if not quiet:
                print(f"  history: could not read {path.name} ({exc}); starting fresh")
            old = None
        if old is not None and len(old):
            if "as_of_date" in old.columns:
                # normalize both sides to date objects before comparing
                old["as_of_date"] = old["as_of_date"].map(_normalize_as_of)
                old = old[old["as_of_date"] != stamp]
            combined = pd.concat([old, new], ignore_index=True)
        else:
            combined = new
    else:
        combined = new

    # keep as_of_date a real DATE, not midnight TIMESTAMP
    combined["as_of_date"] = combined["as_of_date"].map(_normalize_as_of)
    combined.to_parquet(path, index=False)
    if not quiet:
        n_dates = combined["as_of_date"].nunique()
        print(f"  history: {path.name} -> {len(combined):,} rows, {n_dates} dates "
              f"(+{len(new):,} for {stamp})")
    return path


def load_history(name: str, as_of=None) -> pd.DataFrame:
    """Read a history table. With `as_of`, return only that date's rows.

    Returns an empty frame when no history exists yet, so callers can degrade
    gracefully instead of raising during the accumulation period.
    """
    path = history_path(name)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "as_of_date" in df.columns:
        df["as_of_date"] = df["as_of_date"].map(_normalize_as_of)
    if as_of is not None and "as_of_date" in df.columns:
        df = df[df["as_of_date"] == _normalize_as_of(as_of)]
    return df.reset_index(drop=True)


def asof_join(left: pd.DataFrame, name: str, value_cols: list[str],
              left_date: str = "date", by: str = "ticker",
              tolerance_days: int = 400) -> pd.DataFrame:
    """Point-in-time join of a history table onto a dated panel.

    Uses merge_asof backward so a row dated D only ever sees history stamped
    <= D. This is the join buy_candidates_oos needs once history accumulates.
    """
    hist = load_history(name)
    if hist.empty:
        return left
    cols = [by, "as_of_date"] + [c for c in value_cols if c in hist.columns]
    hist = hist[cols].copy()
    # merge_asof requires IDENTICAL datetime resolution on both keys. Parquet
    # date32 reads back as datetime64[s] while a pandas panel is typically
    # datetime64[us]/[ns], which raises "incompatible merge keys". Pin both.
    hist["as_of_date"] = pd.to_datetime(hist["as_of_date"]).astype("datetime64[ns]")
    out = left.copy()
    out[left_date] = pd.to_datetime(out[left_date]).astype("datetime64[ns]")
    out = out.sort_values(left_date, kind="mergesort")
    hist = hist.sort_values("as_of_date", kind="mergesort")
    return pd.merge_asof(out, hist, left_on=left_date, right_on="as_of_date",
                         by=by, direction="backward",
                         tolerance=pd.Timedelta(days=tolerance_days))
