#!/usr/bin/env python3
"""
index_registry.py — Discover and resolve portfolio / screen indexes from data files.

Indexes are derived from:
  - monitored_stocks.parquet boolean membership columns
  - portfolio_holdings.parquet / trades.parquet (portfolio)
  - sector_tickers.csv / sector_prices.parquet (sectors)

Canonical names and aliases:
  fertilizer   ← index_member
  defensive    ← defensive_value_index
  growth       ← growth_tech_index  (alias: growth_tech)
  dual         ← dual_pass_member
  portfolio    ← holdings / in_portfolio / trades
  sectors      ← SECT_* sector EW series

Special alias:
  all          ← every index currently available in the data files

Usage:
  from index_registry import available_indexes, tickers_for_index, parse_indexes

  print(available_indexes())
  print(tickers_for_index("defensive"))
  print(parse_indexes("portfolio,all"))  # expands 'all'
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).parent
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
HOLDINGS_FILE = DATA_DIR / "portfolio_holdings.parquet"
TRADES_FILE = DATA_DIR / "trades.parquet"
# trades may live one level up (artifacts root) in some layouts
TRADES_FILE_ALT = DATA_DIR.parent / "trades.parquet"
SECTOR_META = DATA_DIR / "sector_tickers.csv"
SECTOR_PRICES = DATA_DIR / "sector_prices.parquet"

# Membership column on monitored_stocks → canonical index name
MEMBERSHIP_COLUMNS: dict[str, str] = {
    "index_member": "fertilizer",
    "defensive_value_index": "defensive",
    "growth_tech_index": "growth",
    "dual_pass_member": "dual",
    "sp500_member": "sp500",
}

# Name aliases → canonical
ALIASES: dict[str, str] = {
    "growth_tech": "growth",
    "growth_ai": "growth",
    "fertiliser": "fertilizer",
    "defensive_value": "defensive",
    "dual_pass": "dual",
    "personal": "portfolio",
    "spx": "sp500",
    "s&p500": "sp500",
    "s&p": "sp500",
    "sector": "sectors",
}


def _load_stocks() -> pd.DataFrame:
    if not STOCKS_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(STOCKS_FILE)
    except Exception:
        return pd.DataFrame()


def _portfolio_tickers() -> list[str]:
    if HOLDINGS_FILE.exists():
        try:
            h = pd.read_parquet(HOLDINGS_FILE)
            if "ticker" in h.columns:
                return sorted(h["ticker"].astype(str).str.upper().unique().tolist())
        except Exception:
            pass
    stocks = _load_stocks()
    if not stocks.empty and "in_portfolio" in stocks.columns:
        return sorted(
            stocks.loc[stocks["in_portfolio"] == True, "ticker"].astype(str).str.upper().tolist()
        )
    for path in (TRADES_FILE, TRADES_FILE_ALT):
        if path.exists():
            try:
                t = pd.read_parquet(path)
                if "ticker" in t.columns:
                    return sorted(t["ticker"].astype(str).str.upper().unique().tolist())
            except Exception:
                pass
    return []


def _sector_tickers() -> list[str]:
    if SECTOR_META.exists():
        try:
            return pd.read_csv(SECTOR_META)["ticker"].astype(str).tolist()
        except Exception:
            pass
    if SECTOR_PRICES.exists():
        try:
            return sorted(pd.read_parquet(SECTOR_PRICES)["ticker"].astype(str).unique().tolist())
        except Exception:
            pass
    return []


def discover_membership_indexes(stocks: pd.DataFrame | None = None) -> dict[str, str]:
    """Return {canonical_name: column} for membership flags present in data."""
    stocks = stocks if stocks is not None else _load_stocks()
    found: dict[str, str] = {}
    if stocks.empty:
        return found
    for col, name in MEMBERSHIP_COLUMNS.items():
        if col in stocks.columns:
            # only advertise if at least one member
            try:
                if int((stocks[col] == True).sum()) > 0:
                    found[name] = col
            except Exception:
                found[name] = col
    return found


def available_indexes(*, include_empty: bool = False) -> list[str]:
    """Sorted list of index names currently available in data files."""
    names: set[str] = set()
    stocks = _load_stocks()
    membership = discover_membership_indexes(stocks)
    names.update(membership.keys())

    if include_empty or _portfolio_tickers():
        names.add("portfolio")
    if include_empty or _sector_tickers():
        names.add("sectors")

    # Always expose known membership names if columns exist even when empty? default no
    if include_empty:
        for col, name in MEMBERSHIP_COLUMNS.items():
            if not stocks.empty and col in stocks.columns:
                names.add(name)

    return sorted(names)


def canonicalize(name: str) -> str:
    n = str(name).strip().lower()
    return ALIASES.get(n, n)


def parse_indexes(raw) -> list[str]:
    """Parse comma-separated / list index names; expand 'all'.

    Raises ValueError on unknown names (after aliasing).
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        for item in raw:
            parts.extend(str(item).split(","))
    else:
        parts = str(raw).split(",")

    avail = available_indexes(include_empty=True)
    avail_set = set(avail)
    # also accept aliases of available names
    out: list[str] = []
    for p in parts:
        token = p.strip().lower()
        if not token:
            continue
        if token == "all":
            for n in available_indexes(include_empty=False):
                if n not in out:
                    out.append(n)
            continue
        canon = canonicalize(token)
        if canon not in avail_set and canon not in MEMBERSHIP_COLUMNS.values():
            # allow requesting a known canonical even if currently empty
            if canon not in set(MEMBERSHIP_COLUMNS.values()) | {"portfolio", "sectors"}:
                raise ValueError(
                    f"Unknown index: {token!r}. Available: {', '.join(available_indexes())} (or 'all')"
                )
        if canon not in out:
            out.append(canon)
    return out


def tickers_for_index(name: str, stocks: pd.DataFrame | None = None) -> list[str]:
    """Resolve tickers for one index name (aliases allowed)."""
    stocks = stocks if stocks is not None else _load_stocks()
    idx = canonicalize(name)

    if idx == "all":
        seen: set[str] = set()
        ordered: list[str] = []
        for n in available_indexes(include_empty=False):
            for t in tickers_for_index(n, stocks):
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
        return ordered

    if idx == "portfolio":
        return _portfolio_tickers()

    if idx == "sectors":
        return _sector_tickers()

    # membership column
    col = None
    for c, n in MEMBERSHIP_COLUMNS.items():
        if n == idx:
            col = c
            break
    if col and not stocks.empty and col in stocks.columns:
        return (
            stocks.loc[stocks[col] == True, "ticker"]
            .astype(str)
            .str.upper()
            .tolist()
        )
    return []


def tickers_for_indexes(names: Iterable[str]) -> dict[str, list[str]]:
    """Map each requested index → tickers (after parse / all expansion)."""
    stocks = _load_stocks()
    parsed = parse_indexes(list(names) if not isinstance(names, str) else names)
    return {n: tickers_for_index(n, stocks) for n in parsed}


def ticker_index_map(names: Iterable[str] | None = None) -> dict[str, list[str]]:
    """Map ticker → list of index names it belongs to under the request.

    If names is None, use all available indexes.
    """
    stocks = _load_stocks()
    if names is None:
        indexes = available_indexes(include_empty=False)
    else:
        indexes = parse_indexes(list(names) if not isinstance(names, str) else names)

    mapping: dict[str, list[str]] = {}
    for idx in indexes:
        for t in tickers_for_index(idx, stocks):
            mapping.setdefault(t, [])
            if idx not in mapping[t]:
                mapping[t].append(idx)
    return mapping


def index_help_text() -> str:
    avail = available_indexes()
    return (
        "Index name(s). Comma-separate or repeat flag. "
        f"Available: {', '.join(avail) if avail else '(none found)'}. "
        "Alias: all = every available index. "
        "Also: growth_tech→growth, personal→portfolio."
    )


def argparse_index_kwargs() -> dict:
    """Convenience kwargs for argparse add_argument('--index', **kwargs)."""
    return {
        "action": "append",
        "default": None,
        "help": index_help_text(),
    }


if __name__ == "__main__":
    print("Available indexes:", available_indexes())
    for n in available_indexes():
        t = tickers_for_index(n)
        print(f"  {n:12s}  n={len(t):3d}  {t[:6]}{'...' if len(t) > 6 else ''}")
    print("all n=", len(tickers_for_index("all")))
    print("parse all,portfolio →", parse_indexes("all"))
