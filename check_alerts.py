#!/usr/bin/env python3
"""
check_alerts.py - Evaluate configured alert rules against current price history.

Usage:
  python check_alerts.py                  # run all enabled rules, print & log
  python check_alerts.py --dry-run        # print only, do not write log
  python check_alerts.py --priority high  # only high-priority rules
  python check_alerts.py --ticker CF      # only rules that apply to CF (or *)
  python check_alerts.py --list-rules     # show the current rule set
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
CONFIG_FILE = DATA_DIR / "alerts_config.parquet"
PRICES_FILE = DATA_DIR / "daily_prices.parquet"
STOCKS_FILE = DATA_DIR / "monitored_stocks.parquet"
LOG_FILE = DATA_DIR / "alerts_log.parquet"


def load_config(enabled_only: bool = True) -> pd.DataFrame:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing {CONFIG_FILE}. Run alerts_config.parquet.py first.")
    df = pd.read_parquet(CONFIG_FILE)
    if enabled_only:
        df = df[df["enabled"] == True]
    return df


def load_prices() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        raise FileNotFoundError(f"Missing {PRICES_FILE}")
    df = pd.read_parquet(PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def load_index_members() -> list[str]:
    if not STOCKS_FILE.exists():
        return []
    s = pd.read_parquet(STOCKS_FILE)
    return s[s["index_member"] == True]["ticker"].tolist()


def latest_by_ticker(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.groupby("ticker").tail(1).set_index("ticker")


def series_for(ticker: str, prices: pd.DataFrame) -> pd.DataFrame:
    return prices[prices["ticker"] == ticker].copy().reset_index(drop=True)


def eval_price_above(row, prices, latest) -> list[dict]:
    t = row["ticker"]
    threshold = float(row["param1"])
    if t not in latest.index:
        return []
    close = float(latest.loc[t, "close"])
    if close > threshold:
        return [{
            "ticker": t,
            "message": f"{t} closed at {close:.2f} > {threshold:.2f}",
            "value": close,
            "threshold": threshold,
        }]
    return []


def eval_price_below(row, prices, latest) -> list[dict]:
    t = row["ticker"]
    threshold = float(row["param1"])
    if t not in latest.index:
        return []
    close = float(latest.loc[t, "close"])
    if close < threshold:
        return [{
            "ticker": t,
            "message": f"{t} closed at {close:.2f} < {threshold:.2f}",
            "value": close,
            "threshold": threshold,
        }]
    return []


def eval_pct_change(row, prices, latest, direction: str) -> list[dict]:
    """direction = 'above' or 'below'. param1 = % threshold, param2 = lookback days."""
    threshold = float(row["param1"])
    lookback = int(row["param2"]) if pd.notna(row["param2"]) else 1
    tickers = prices["ticker"].unique() if row["ticker"] == "*" else [row["ticker"]]
    hits = []
    for t in tickers:
        s = series_for(t, prices)
        if len(s) < lookback + 1:
            continue
        current = float(s.iloc[-1]["close"])
        past = float(s.iloc[-(lookback + 1)]["close"])
        if past == 0:
            continue
        pct = (current - past) / past * 100.0
        triggered = (pct >= threshold) if direction == "above" else (pct <= threshold)
        if triggered:
            hits.append({
                "ticker": t,
                "message": f"{t} {pct:+.2f}% over {lookback}d (threshold {threshold:+.1f}%)",
                "value": pct,
                "threshold": threshold,
            })
    return hits


def eval_sma(row, prices, latest, side: str) -> list[dict]:
    """side = 'above' or 'below'. param1 = SMA window."""
    window = int(row["param1"])
    if row["ticker"] == "INDEX":
        tickers = load_index_members()
    elif row["ticker"] == "*":
        tickers = prices["ticker"].unique().tolist()
    else:
        tickers = [row["ticker"]]

    hits = []
    for t in tickers:
        s = series_for(t, prices)
        if len(s) < window:
            continue
        sma = s["close"].tail(window).mean()
        close = float(s.iloc[-1]["close"])
        triggered = (close > sma) if side == "above" else (close < sma)
        if triggered:
            hits.append({
                "ticker": t,
                "message": f"{t} close {close:.2f} is {side} {window}d SMA ({sma:.2f})",
                "value": close,
                "threshold": sma,
            })
    return hits


def eval_new_extreme(row, prices, latest, extreme: str) -> list[dict]:
    """extreme = 'high' or 'low'. param1 = lookback window."""
    window = int(row["param1"])
    tickers = prices["ticker"].unique() if row["ticker"] == "*" else [row["ticker"]]
    hits = []
    for t in tickers:
        s = series_for(t, prices)
        if len(s) < window:
            continue
        window_slice = s.tail(window)
        current = float(window_slice.iloc[-1]["close"])
        if extreme == "high":
            prior_max = float(window_slice.iloc[:-1]["close"].max()) if len(window_slice) > 1 else current
            if current >= prior_max and current == float(window_slice["close"].max()):
                hits.append({
                    "ticker": t,
                    "message": f"{t} new {window}d high at {current:.2f}",
                    "value": current,
                    "threshold": prior_max,
                })
        else:
            prior_min = float(window_slice.iloc[:-1]["close"].min()) if len(window_slice) > 1 else current
            if current <= prior_min and current == float(window_slice["close"].min()):
                hits.append({
                    "ticker": t,
                    "message": f"{t} new {window}d low at {current:.2f}",
                    "value": current,
                    "threshold": prior_min,
                })
    return hits


def eval_volume_spike(row, prices, latest) -> list[dict]:
    """param1 = multiple of average, param2 = lookback for average."""
    multiple = float(row["param1"])
    lookback = int(row["param2"]) if pd.notna(row["param2"]) else 10
    tickers = prices["ticker"].unique() if row["ticker"] == "*" else [row["ticker"]]
    hits = []
    for t in tickers:
        s = series_for(t, prices)
        if len(s) < lookback + 1:
            continue
        # skip if volume is zero (our synthetic/screenshot data often has 0)
        recent_vol = float(s.iloc[-1]["volume"])
        if recent_vol <= 0:
            continue
        avg_vol = float(s.iloc[-(lookback + 1):-1]["volume"].mean())
        if avg_vol <= 0:
            continue
        if recent_vol >= multiple * avg_vol:
            hits.append({
                "ticker": t,
                "message": f"{t} volume {recent_vol:,.0f} is {recent_vol/avg_vol:.1f}× the {lookback}d avg",
                "value": recent_vol,
                "threshold": avg_vol * multiple,
            })
    return hits



def load_fundamentals() -> pd.DataFrame:
    path = DATA_DIR / "fundamentals.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "as_of_date" in df.columns:
        df = df.sort_values("as_of_date").groupby("ticker").tail(1)
    return df.set_index("ticker")


def parse_conditions(rule) -> list[tuple[str, str, float]]:
    """
    Conditions from rule['conditions'] string:
      'ev_ebitda<=9;pb_ratio<=1.5;mktcap_to_assets<=0.5'
    or single metric via rule_type metric_below/above using param1 + metric column.
    Returns list of (metric, op, threshold).
    """
    raw = rule.get("conditions")
    if pd.isna(raw) if not isinstance(raw, str) else (not raw):
        # fall back: metric in notes prefix or param-based single
        metric = rule.get("metric")
        if isinstance(metric, str) and metric and pd.notna(rule.get("param1")):
            op = "<=" if "below" in str(rule.get("rule_type", "")) else ">="
            if rule.get("rule_type") == "metric_below":
                op = "<="
            elif rule.get("rule_type") == "metric_above":
                op = ">="
            return [(metric, op, float(rule["param1"]))]
        return []
    parts = []
    for token in str(raw).split(";"):
        token = token.strip()
        if not token:
            continue
        for op in ("<=", ">=", "==", "<", ">"):
            if op in token:
                m, thr = token.split(op, 1)
                parts.append((m.strip(), op, float(thr.strip())))
                break
    return parts


def eval_condition(value, op: str, thr: float) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    v = float(value)
    if op == "<=": return v <= thr
    if op == ">=": return v >= thr
    if op == "<": return v < thr
    if op == ">": return v > thr
    if op == "==": return abs(v - thr) < 1e-9
    return False


def eval_fundamentals_screen(row, prices, latest) -> list[dict]:
    """
    Flexible fundamental / trifecta screen.
    rule.conditions e.g. 'ev_ebitda<=9;pb_ratio<=1.5;mktcap_to_assets<=0.5'
    rule.match_mode: 'all' (AND, default) or 'any' (OR)
    ticker '*' = all names in fundamentals table.
    """
    fund = load_fundamentals()
    if fund.empty:
        return []
    conds = parse_conditions(row)
    if not conds:
        return []
    mode = str(row.get("match_mode") or "all").lower()
    if row["ticker"] in ("*", "INDEX"):
        tickers = list(fund.index)
    else:
        tickers = [row["ticker"]]

    hits = []
    for t in tickers:
        if t not in fund.index:
            continue
        fr = fund.loc[t]
        results = []
        detail = []
        for metric, op, thr in conds:
            if metric not in fr.index and metric not in fund.columns:
                results.append(False)
                detail.append(f"{metric}=NA")
                continue
            val = fr[metric] if metric in fr.index else fr.get(metric)
            ok = eval_condition(val, op, thr)
            results.append(ok)
            detail.append(f"{metric}={float(val):.3g}{op}{thr}" if pd.notna(val) else f"{metric}=NA")
        triggered = all(results) if mode == "all" else any(results)
        if triggered:
            hits.append({
                "ticker": t,
                "message": f"{t} screen ({mode}): " + "; ".join(detail),
                "value": float(sum(results)) / max(len(results), 1),
                "threshold": 1.0 if mode == "all" else 0.0,
            })
    return hits


def eval_metric(row, prices, latest, direction: str) -> list[dict]:
    """Single fundamental metric above/below param1. metric name in rule['metric']."""
    row = row.copy()
    row["rule_type"] = f"metric_{direction}"
    if not row.get("conditions") and row.get("metric"):
        op = "<=" if direction == "below" else ">="
        row["conditions"] = f"{row['metric']}{op}{float(row['param1'])}"
    return eval_fundamentals_screen(row, prices, latest)


RULE_DISPATCH = {
    "price_above": lambda r, p, l: eval_price_above(r, p, l),
    "price_below": lambda r, p, l: eval_price_below(r, p, l),
    "pct_change_above": lambda r, p, l: eval_pct_change(r, p, l, "above"),
    "pct_change_below": lambda r, p, l: eval_pct_change(r, p, l, "below"),
    "above_sma": lambda r, p, l: eval_sma(r, p, l, "above"),
    "below_sma": lambda r, p, l: eval_sma(r, p, l, "below"),
    "new_high": lambda r, p, l: eval_new_extreme(r, p, l, "high"),
    "new_low": lambda r, p, l: eval_new_extreme(r, p, l, "low"),
    "volume_spike": lambda r, p, l: eval_volume_spike(r, p, l),
    "fundamentals_screen": lambda r, p, l: eval_fundamentals_screen(r, p, l),
    "trifecta": lambda r, p, l: eval_fundamentals_screen(r, p, l),
    "metric_below": lambda r, p, l: eval_metric(r, p, l, "below"),
    "metric_above": lambda r, p, l: eval_metric(r, p, l, "above"),
}


def run_alerts(
    priority_filter: str | None = None,
    ticker_filter: str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    config = load_config(enabled_only=True)
    prices = load_prices()
    latest = latest_by_ticker(prices)

    if priority_filter:
        config = config[config["priority"] == priority_filter]
    if ticker_filter:
        t = ticker_filter.upper()
        config = config[(config["ticker"] == t) | (config["ticker"].isin(["*", "INDEX"]))]

    alerts = []
    now = pd.Timestamp.now()

    for _, rule in config.iterrows():
        rule_type = rule["rule_type"]
        if rule_type not in RULE_DISPATCH:
            print(f"  ⚠ Unknown rule_type: {rule_type} (rule_id={rule['rule_id']})")
            continue
        hits = RULE_DISPATCH[rule_type](rule, prices, latest)
        for h in hits:
            alerts.append({
                "timestamp": now,
                "rule_id": rule["rule_id"],
                "ticker": h["ticker"],
                "rule_type": rule_type,
                "priority": rule["priority"],
                "message": h["message"],
                "value": h.get("value"),
                "threshold": h.get("threshold"),
                "notes": rule.get("notes", ""),
            })

    if not alerts:
        print("No alerts triggered.")
        return pd.DataFrame()

    alert_df = pd.DataFrame(alerts)

    # Pretty print
    print(f"\n🚨 {len(alert_df)} alert(s) triggered  ({now.strftime('%Y-%m-%d %H:%M')})\n")
    for prio in ["high", "medium", "low"]:
        subset = alert_df[alert_df["priority"] == prio]
        if subset.empty:
            continue
        print(f"── {prio.upper()} ──")
        for _, a in subset.iterrows():
            print(f"  [{a['ticker']}] {a['message']}")
        print()

    if not dry_run:
        if LOG_FILE.exists():
            existing = pd.read_parquet(LOG_FILE)
            combined = pd.concat([existing, alert_df], ignore_index=True)
        else:
            combined = alert_df
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, LOG_FILE)
        print(f"Logged to {LOG_FILE} (total history: {len(combined)} rows)")

    return alert_df


def list_rules():
    df = load_config(enabled_only=False)
    print(df[["rule_id", "ticker", "rule_type", "param1", "param2", "enabled", "priority", "notes"]]
          .to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Stock alerts evaluation engine")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to alerts_log")
    parser.add_argument("--priority", choices=["high", "medium", "low"], help="Filter by priority")
    parser.add_argument("--ticker", help="Only evaluate rules that apply to this ticker (or *)")
    parser.add_argument("--list-rules", action="store_true", help="Show configured rules and exit")
    args = parser.parse_args()

    if args.list_rules:
        list_rules()
        return

    run_alerts(
        priority_filter=args.priority,
        ticker_filter=args.ticker,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
