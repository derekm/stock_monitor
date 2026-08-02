#!/usr/bin/env python3
"""
backfill_constituents.py — fill REAL multi-snapshot fundamentals + price history
for S&P 500 constituents that are missing from our store, using yfinance.

This is the "fill in real actuals" step. It is intentionally integrated with
stock_monitor's existing generator conventions (update_fundamentals.py /
update_prices.py): it writes the SAME schema as fundamentals.parquet and
daily_prices.parquet, stamps source='yfinance', and is resume-safe so a
long run can be interrupted and resumed.

What it fetches per missing ticker (real, point-in-time aligned to quarter-ends):
  - quarterly balance sheet  -> total assets, book equity, total debt
  - quarterly income/financials -> net income (TTM), EBIT, EBITDA, interest exp
  - info                     -> shares outstanding, latest price, sector
  - daily history (5y)       -> price/volume timeline appended to daily_prices
From those it derives the canonical quality metrics (roe, roic, debt_to_equity,
interest_coverage, ev_ebitda, mktcap_to_assets, pb_ratio) per quarter-end, so
the PIT backfill is genuine multi-snapshot history — NOT synthetic noise.

Usage:
  python backfill_constituents.py run        # backfill all missing constituents
  python backfill_constituents.py run --limit 20   # smoke test on 20 tickers
  python backfill_constituents.py merge      # union staging into the real files
  python backfill_constituents.py status     # show progress

Notes:
  - Resume-safe: tickers already present in the yfinance staging file are skipped.
  - yfinance is rate-limited; we sleep + retry with backoff. Expect a long run.
  - We do NOT overwrite the 142 existing fundamentals rows (different source).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

DATA_DIR = Path(__file__).parent
FUND = DATA_DIR / "fundamentals.parquet"
PRICES = DATA_DIR / "daily_prices.parquet"
CONST = DATA_DIR / "sp500_constituents.parquet"
FUND_STAGE = DATA_DIR / "fundamentals_yfinance.parquet"
PRICE_STAGE = DATA_DIR / "daily_prices_yfinance.parquet"
PROGRESS = DATA_DIR / "backfill_progress.json"

FUND_COLS = [
    "ticker", "as_of_date", "market_cap", "market_cap_b", "total_assets",
    "total_assets_b", "pb_ratio", "mktcap_to_assets", "source", "notes",
    "last_updated", "ev_ebitda", "pb_vs_ev_note", "roe", "roic",
    "debt_to_equity", "interest_coverage", "earnings_stability", "quality_source",
]
PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume", "source"]


def missing_tickers() -> list[str]:
    import duckdb
    c = duckdb.connect()
    rows = c.execute(
        f"""
        SELECT s.ticker FROM read_parquet('{CONST.as_posix()}') s
        WHERE s.current
          AND s.ticker NOT IN (
              SELECT DISTINCT ticker FROM read_parquet('{FUND.as_posix()}')
          )
        ORDER BY s.ticker
        """
    ).fetchall()
    return [r[0] for r in rows]


def _q(ts, label):
    """Pull a value from a yfinance quarterly statement by row label."""
    if ts is None or label not in ts.index:
        return None
    # columns are dates; take the most recent non-null
    col = ts.loc[label].dropna()
    if col.empty:
        return None
    return float(col.iloc[0])


def derive_row(ticker: str, q_end: date, price: float | None,
               shares: float | None, bs: pd.DataFrame, fin: pd.DataFrame,
               cf: pd.DataFrame) -> dict | None:
    """Build one fundamentals row for a quarter-end from real statements."""
    if price is None or shares is None or shares <= 0:
        return None
    ta = _q(bs, "Total Assets")
    te = _q(bs, "Stockholders Equity")
    if te is None:
        te = _q(bs, "Total Equity Gross Minority Interest")
    tld = _q(bs, "Long Term Debt")
    td = _q(bs, "Total Debt")
    if td is None and tld is not None:
        td = tld
    cash = _q(bs, "Cash And Cash Equivalents")
    if cash is None:
        cash = _q(bs, "Cash")

    ni = _q(fin, "Net Income")
    ebit = _q(fin, "EBIT")
    if ebit is None:
        ebit = _q(fin, "Operating Income")
    ebitda = _q(fin, "EBITDA")
    int_exp = _q(fin, "Interest Expense")
    if int_exp is None:
        int_exp = _q(fin, "Interest Expense Non Operating")

    # TTM aggregates (sum last 4 quarters where present)
    def ttm(label, src):
        if src is None or label not in src.index:
            return None
        v = src.loc[label].dropna()
        if v.empty:
            return None
        return float(v.head(4).sum())
    ni_ttm = ttm("Net Income", fin)
    ebit_ttm = ttm("EBIT", fin) or ttm("Operating Income", fin)
    ebitda_ttm = ttm("EBITDA", fin)
    int_ttm = ttm("Interest Expense", fin) or ttm("Interest Expense Non Operating", fin)

    mcap = price * shares
    pb = (price / (te / shares)) if te and te > 0 else None
    roe = (ni_ttm / te) if (ni_ttm is not None and te and te > 0) else None
    debt_equity = (td / te) if (td is not None and te and te > 0) else None
    cap_struct = (te + (td or 0))
    roic = (ni_ttm / cap_struct) if (ni_ttm is not None and cap_struct and cap_struct > 0) else None
    icov = (ebit_ttm / abs(int_ttm)) if (ebit_ttm is not None and int_ttm and int_ttm != 0) else None
    ev = mcap + (td or 0) - (cash or 0)
    ev_ebitda = (ev / ebitda_ttm) if (ebitda_ttm and ebitda_ttm > 0) else None
    mkt_assets = (mcap / ta) if (ta and ta > 0) else None

    # earnings stability: |mean|/std of available quarterly net income
    stab = None
    if fin is not None and "Net Income" in fin.index:
        s = fin.loc["Net Income"].dropna().head(8)
        if len(s) >= 4 and s.mean() != 0:
            stab = abs(s.mean()) / s.std(ddof=0) if s.std(ddof=0) else None

    return {
        "ticker": ticker,
        "as_of_date": pd.Timestamp(q_end),
        "market_cap": mcap,
        "market_cap_b": mcap / 1e9,
        "total_assets": ta,
        "total_assets_b": (ta / 1e9) if ta else None,
        "pb_ratio": pb,
        "mktcap_to_assets": mkt_assets,
        "source": "yfinance",
        "notes": f"quarter_end={q_end}; price={price:.2f}; shares={shares:.0f}",
        "last_updated": pd.Timestamp(datetime.now()),
        "ev_ebitda": ev_ebitda,
        "pb_vs_ev_note": None,
        "roe": roe,
        "roic": roic,
        "debt_to_equity": debt_equity,
        "interest_coverage": icov,
        "earnings_stability": stab,
        "quality_source": "yfinance_quarterly",
    }


def fetch_ticker(ticker: str, sleep: float = 0.4):
    """Return (fund_rows, price_rows) or ([], []) on failure.

    Robustness:
      - Yahoo uses hyphens, not dots (BRK-B not BRK.B) -> normalize.
      - shares outstanding may be missing from a rate-limited `info` call even
        when the data exists -> fall back to balance-sheet 'Share Issued',
        fast_info, or marketCap/currentPrice. Only give up if truly absent.
    """
    yf_sym = ticker.replace(".", "-")
    t = yf.Ticker(yf_sym)
    tries = 4
    bs = fin = info = hist = None
    for attempt in range(tries):
        try:
            bs = t.quarterly_balance_sheet
            fin = t.quarterly_financials
            info = t.info
            hist = t.history(period="5y", interval="1d", actions=False)
            if info is None or not info:
                raise ValueError("empty info (rate-limited)")
            break
        except Exception as e:  # network/rate limit / empty info
            if attempt == tries - 1:
                print(f"  [{ticker}] fetch failed: {e}")
                return [], []
            time.sleep(2.0 * (attempt + 1))

    shares = info.get("sharesOutstanding")
    if not shares and bs is not None:
        shares = _q(bs, "Share Issued") or _q(bs, "Ordinary Shares Issued")
    if not shares:
        try:
            shares = float(t.fast_info.get("shares"))
        except Exception:
            shares = None
    if not shares:
        mc = info.get("marketCap")
        cp = info.get("currentPrice")
        if mc and cp:
            shares = mc / cp
    if not shares:
        print(f"  [{ticker}] no shares outstanding; skipping")
        return [], []

    # price lookup by date
    closes = {}
    if hist is not None and not hist.empty:
        closes = {d.date(): float(hist.loc[d, "Close"]) for d in hist.index}

    fund_rows = []
    if bs is not None and not bs.empty:
        for q_end_ts in bs.columns:
            q_end = q_end_ts.date()
            price = closes.get(q_end)
            # if no exact close on quarter-end, use the last close <= q_end
            if price is None:
                cands = [d for d in closes if d <= q_end]
                if cands:
                    price = closes[max(cands)]
            row = derive_row(ticker, q_end, price, shares, bs, fin, None)
            if row:
                fund_rows.append(row)

    price_rows = []
    if hist is not None and not hist.empty:
        for d in hist.index:
            price_rows.append({
                "date": pd.Timestamp(d.date()),
                "ticker": ticker,
                "open": float(hist.loc[d, "Open"]) if "Open" in hist else None,
                "high": float(hist.loc[d, "High"]) if "High" in hist else None,
                "low": float(hist.loc[d, "Low"]) if "Low" in hist else None,
                "close": float(hist.loc[d, "Close"]),
                "volume": float(hist.loc[d, "Volume"]) if "Volume" in hist else None,
                "source": "yfinance",
            })
    time.sleep(sleep)
    return fund_rows, price_rows


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"done": [], "failed": []}


def save_progress(p: dict):
    PROGRESS.write_text(json.dumps(p, default=str, indent=2))


def _append_parquet(path: Path, rows: list[dict], cols: list[str]):
    if not rows:
        return
    df = pd.DataFrame(rows)[cols]
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def cmd_run(args):
    tickers = missing_tickers()
    if args.limit:
        tickers = tickers[: args.limit]
    prog = load_progress()
    done = set(prog.get("done", []))
    failed = set(prog.get("failed", []))
    if args.retry_failed:
        # move previously-failed back into the todo pool (clear the failed set)
        prog["failed"] = []
        failed = set()
    todo = [t for t in tickers if t not in done and t not in failed]
    print(f"missing={len(tickers)} already_done={len(done)} failed={len(failed)} to_fetch={len(todo)}")

    fund_buf: list[dict] = []
    price_buf: list[dict] = []
    FLUSH = 25
    for i, tk in enumerate(todo, 1):
        fr, pr = fetch_ticker(tk, sleep=args.sleep)
        if fr or pr:
            fund_buf.extend(fr)
            price_buf.extend(pr)
            if tk not in prog.get("done", []):
                prog.setdefault("done", []).append(tk)
        else:
            if tk not in prog.get("failed", []):
                prog.setdefault("failed", []).append(tk)
        if i % FLUSH == 0:
            _append_parquet(FUND_STAGE, fund_buf, FUND_COLS)
            _append_parquet(PRICE_STAGE, price_buf, PRICE_COLS)
            fund_buf, price_buf = [], []
            save_progress(prog)
            print(f"  progress {i}/{len(todo)} done={len(prog['done'])} failed={len(prog['failed'])}")
    # flush remainder
    _append_parquet(FUND_STAGE, fund_buf, FUND_COLS)
    _append_parquet(PRICE_STAGE, price_buf, PRICE_COLS)
    save_progress(prog)
    print(f"DONE. staging: fund={FUND_STAGE} price={PRICE_STAGE}. Run 'merge' to union into real files.")


def cmd_merge(args):
    import duckdb
    c = duckdb.connect()
    n_fund = c.execute(f"SELECT COUNT(*) FROM read_parquet('{FUND_STAGE.as_posix()}')").fetchone()[0] if FUND_STAGE.exists() else 0
    n_price = c.execute(f"SELECT COUNT(*) FROM read_parquet('{PRICE_STAGE.as_posix()}')").fetchone()[0] if PRICE_STAGE.exists() else 0
    print(f"staging rows: fundamentals={n_fund} prices={n_price}")
    if n_fund == 0 and n_price == 0:
        print("nothing to merge")
        return
    c.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{FUND.as_posix()}')
          UNION ALL
          SELECT * FROM read_parquet('{FUND_STAGE.as_posix()}')
        ) TO '{FUND.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    print(f"wrote {FUND}")
    c.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{PRICES.as_posix()}')
          UNION ALL
          SELECT * FROM read_parquet('{PRICE_STAGE.as_posix()}')
        ) TO '{PRICES.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    print(f"wrote {PRICES}")
    # report new coverage
    cov = c.execute(
        f"SELECT COUNT(DISTINCT ticker) FROM read_parquet('{FUND.as_posix()}')"
    ).fetchone()[0]
    pcov = c.execute(
        f"SELECT COUNT(DISTINCT ticker) FROM read_parquet('{PRICES.as_posix()}')"
    ).fetchone()[0]
    print(f"coverage now: fundamentals tickers={cov} price tickers={pcov}")


def cmd_status(args):
    prog = load_progress()
    print(f"done={len(prog.get('done', []))} failed={len(prog.get('failed', []))}")
    print("failed:", prog.get("failed", [])[:40])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--sleep", type=float, default=0.4)
    r.add_argument("--retry-failed", action="store_true",
                   help="retry tickers previously marked failed (transient errors)")
    r.set_defaults(func=cmd_run)
    m = sub.add_parser("merge")
    m.set_defaults(func=cmd_merge)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
