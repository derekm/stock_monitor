#!/usr/bin/env python3
"""
cli_common.py — Shared CLI conventions for stock_monitor programs.

Standard flags (use these everywhere):
  --index     Index name(s), comma-separated or repeatable. Alias: --universe
  --ticker    Comma-separated tickers (overrides --index). Alias: --tickers
  --sector    Sector name or SECT_ slug
  --save      Write CSV/parquet outputs
  --window    Lookback window in trading days
  --horizon   Forecast horizon
  --freq      D / W / M

Prefer --index over --universe; --universe is accepted as a hidden alias.
Prefer --ticker over --tickers; both accepted.

Usage:
  from cli_common import (
      add_index_args, add_save_arg, add_window_arg,
      resolve_tickers_from_args, build_parser,
  )
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

from index_registry import (
    available_indexes,
    index_help_text,
    parse_indexes,
    tickers_for_index,
    ticker_index_map,
)


def add_index_args(
    ap: argparse.ArgumentParser | argparse._ActionsContainer,
    *,
    default: str | None = None,
    required: bool = False,
) -> None:
    """Add --index and hidden --universe alias."""
    ap.add_argument(
        "--index",
        action="append",
        default=None,
        dest="index",
        help=index_help_text()
        + (f" Default: {default}." if default else ""),
    )
    ap.add_argument(
        "--universe",
        action="append",
        default=None,
        dest="universe",
        help=argparse.SUPPRESS,  # alias for --index
    )
    ap.set_defaults(_index_default=default)


def add_ticker_args(ap: argparse.ArgumentParser | argparse._ActionsContainer) -> None:
    """Add --ticker and hidden --tickers alias."""
    ap.add_argument(
        "--ticker",
        default=None,
        dest="ticker",
        help="Comma-separated tickers (overrides --index)",
    )
    ap.add_argument(
        "--tickers",
        default=None,
        dest="tickers",
        help=argparse.SUPPRESS,
    )


def add_sector_arg(ap: argparse.ArgumentParser | argparse._ActionsContainer) -> None:
    ap.add_argument(
        "--sector",
        default=None,
        help="Sector name(s) or SECT_ slug(s), comma-separated",
    )


def add_save_arg(ap: argparse.ArgumentParser | argparse._ActionsContainer) -> None:
    ap.add_argument("--save", action="store_true", help="Write output files")


def add_window_arg(
    ap: argparse.ArgumentParser | argparse._ActionsContainer,
    default: int = 126,
) -> None:
    ap.add_argument(
        "--window",
        type=int,
        default=default,
        help=f"Lookback window in trading days (default {default})",
    )


def add_horizon_arg(
    ap: argparse.ArgumentParser | argparse._ActionsContainer,
    default: int = 10,
) -> None:
    ap.add_argument(
        "--horizon",
        type=int,
        default=default,
        help=f"Forecast horizon in days (default {default})",
    )


def add_freq_arg(
    ap: argparse.ArgumentParser | argparse._ActionsContainer,
    default: str = "D",
) -> None:
    ap.add_argument(
        "--freq",
        default=default,
        choices=["D", "W", "M", "d", "w", "m"],
        help=f"Sampling frequency D/W/M (default {default})",
    )


def build_parser(
    description: str,
    *,
    with_index: bool = True,
    with_ticker: bool = True,
    with_sector: bool = False,
    with_save: bool = True,
    with_window: bool = False,
    with_horizon: bool = False,
    with_freq: bool = False,
    index_default: str | None = None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    if with_index:
        add_index_args(ap, default=index_default)
    if with_ticker:
        add_ticker_args(ap)
    if with_sector:
        add_sector_arg(ap)
    if with_window:
        add_window_arg(ap)
    if with_horizon:
        add_horizon_arg(ap)
    if with_freq:
        add_freq_arg(ap)
    if with_save:
        add_save_arg(ap)
    return ap


def _merge_index_args(args: argparse.Namespace) -> list[str] | None:
    """Combine --index and --universe (alias) into one list."""
    parts: list[str] = []
    for attr in ("index", "universe"):
        val = getattr(args, attr, None)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            parts.extend(str(x) for x in val)
        else:
            parts.append(str(val))
    if parts:
        return parts
    default = getattr(args, "_index_default", None)
    if default:
        return [default]
    return None


def _ticker_string(args: argparse.Namespace) -> str | None:
    for attr in ("ticker", "tickers"):
        val = getattr(args, attr, None)
        if val:
            return str(val)
    return None


def resolve_tickers_from_args(
    args: argparse.Namespace,
    *,
    default_index: str | None = "portfolio",
) -> list[str]:
    """Resolve tickers from standard CLI flags.

    Priority: --ticker/--tickers > --sector > --index/--universe > default_index.
    """
    tickers_raw = _ticker_string(args)
    if tickers_raw:
        return [x.strip().upper() for x in tickers_raw.split(",") if x.strip()]

    sector = getattr(args, "sector", None)
    if sector:
        # sector names → SECT_ or stocks in that sector
        from pathlib import Path
        import pandas as pd

        data = Path(__file__).parent
        stocks_path = data / "daily_prices.parquet"
        meta_path = data / "monitored_stocks.parquet"
        out: list[str] = []
        raw = [s.strip() for s in str(sector).split(",") if s.strip()]
        meta = data / "sector_tickers.parquet"
        slug_map: dict[str, str] = {}
        if meta.exists():
            m = pd.read_parquet(meta)
            slug_map = dict(zip(m["sector_name"].str.lower(), m["ticker"]))
            slug_map.update({t.lower(): t for t in m["ticker"]})
        stocks = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()
        for s in raw:
            key = s.lower()
            if key in slug_map:
                out.append(slug_map[key])
            elif s.upper().startswith("SECT_"):
                out.append(s.upper())
            elif not stocks.empty and "sector" in stocks.columns:
                hits = stocks.loc[
                    stocks["sector"].str.lower() == key, "ticker"
                ].astype(str).str.upper().tolist()
                out.extend(hits)
            else:
                out.append("SECT_" + "".join(ch if ch.isalnum() else "_" for ch in s).upper()[:24])
        # dedupe
        seen, deduped = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    index_parts = _merge_index_args(args)
    if not index_parts and default_index:
        index_parts = [default_index]
    if not index_parts:
        return []

    try:
        names = parse_indexes(index_parts)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    seen, out = set(), []
    for n in names:
        for tk in tickers_for_index(n):
            if tk not in seen:
                seen.add(tk)
                out.append(tk)
    return out


def resolve_index_names_from_args(
    args: argparse.Namespace,
    *,
    default_index: str | None = None,
) -> list[str]:
    """Return canonical index names requested (no ticker expansion)."""
    parts = _merge_index_args(args)
    if not parts and default_index:
        parts = [default_index]
    if not parts:
        return []
    try:
        return parse_indexes(parts)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def ticker_index_map_from_args(
    args: argparse.Namespace,
    *,
    default_index: str | None = None,
) -> dict[str, list[str]]:
    """Map ticker → index labels for the current CLI request."""
    tickers_raw = _ticker_string(args)
    if tickers_raw:
        return {
            t.strip().upper(): ["custom"]
            for t in tickers_raw.split(",")
            if t.strip()
        }
    names = resolve_index_names_from_args(args, default_index=default_index)
    if not names:
        return {}
    return ticker_index_map(names)


if __name__ == "__main__":
    ap = build_parser(
        "cli_common demo",
        with_sector=True,
        with_window=True,
        index_default="portfolio",
    )
    args = ap.parse_args()
    print("indexes:", resolve_index_names_from_args(args, default_index="portfolio"))
    print("tickers:", resolve_tickers_from_args(args)[:10], "...")
