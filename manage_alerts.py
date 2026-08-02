#!/usr/bin/env python3
"""
manage_alerts.py - Add, enable/disable, or list alert rules.

  python manage_alerts.py list
  python manage_alerts.py enable RULE_ID
  python manage_alerts.py disable RULE_ID
  python manage_alerts.py add --rule-id MY_RULE --ticker CF --type price_above --param1 140 --priority high --notes "..."
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent
CONFIG_FILE = DATA_DIR / "alerts_config.parquet"

VALID_TYPES = [
    "fundamentals_screen", "trifecta", "metric_below", "metric_above",
    "price_above", "price_below",
    "pct_change_above", "pct_change_below",
    "above_sma", "below_sma",
    "new_high", "new_low",
    "volume_spike",
]


def load() -> pd.DataFrame:
    if not CONFIG_FILE.exists():
        return pd.DataFrame(columns=[
            "rule_id", "ticker", "rule_type", "param1", "param2",
            "enabled", "priority", "notes", "created"
        ])
    return pd.read_parquet(CONFIG_FILE)


def save(df: pd.DataFrame):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, CONFIG_FILE)
    print(f"Saved {len(df)} rules → {CONFIG_FILE}")


def cmd_list(args):
    df = load()
    if args.enabled_only:
        df = df[df["enabled"] == True]
    print(df[["rule_id", "ticker", "rule_type", "param1", "param2", "enabled", "priority"]]
          .to_string(index=False))


def cmd_enable(args):
    df = load()
    mask = df["rule_id"] == args.rule_id
    if not mask.any():
        print(f"Rule '{args.rule_id}' not found")
        return
    df.loc[mask, "enabled"] = True
    save(df)
    print(f"Enabled {args.rule_id}")


def cmd_disable(args):
    df = load()
    mask = df["rule_id"] == args.rule_id
    if not mask.any():
        print(f"Rule '{args.rule_id}' not found")
        return
    df.loc[mask, "enabled"] = False
    save(df)
    print(f"Disabled {args.rule_id}")


def cmd_add(args):
    if args.type not in VALID_TYPES:
        print(f"Invalid type. Choose from: {VALID_TYPES}")
        return
    df = load()
    if args.rule_id in df["rule_id"].values:
        print(f"rule_id '{args.rule_id}' already exists")
        return
    new = {
        "rule_id": args.rule_id,
        "ticker": args.ticker.upper(),
        "rule_type": args.type,
        "param1": float(args.param1) if args.param1 is not None else None,
        "param2": float(args.param2) if args.param2 is not None else None,
        "enabled": True,
        "priority": args.priority,
        "notes": args.notes or "",
        "created": pd.Timestamp.now(),
        "conditions": getattr(args, "conditions", None),
        "match_mode": getattr(args, "match_mode", "all"),
        "metric": getattr(args, "metric", None),
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    save(df)
    print(f"Added rule {args.rule_id}")


def cmd_delete(args):
    df = load()
    before = len(df)
    df = df[df["rule_id"] != args.rule_id]
    if len(df) == before:
        print(f"Rule '{args.rule_id}' not found")
        return
    save(df)
    print(f"Deleted {args.rule_id}")


def main():
    parser = argparse.ArgumentParser(description="Manage alert rules")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("list")
    p.add_argument("--enabled-only", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("enable")
    p.add_argument("rule_id")
    p.set_defaults(func=cmd_enable)

    p = sub.add_parser("disable")
    p.add_argument("rule_id")
    p.set_defaults(func=cmd_disable)

    p = sub.add_parser("delete")
    p.add_argument("rule_id")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("add")
    p.add_argument("--rule-id", required=True)
    p.add_argument("--ticker", required=True, help="Ticker, '*' for all, or 'INDEX' for index members")
    p.add_argument("--type", required=True, choices=VALID_TYPES)
    p.add_argument("--param1", type=float, default=None)
    p.add_argument("--param2", type=float, default=None)
    p.add_argument("--priority", default="medium", choices=["high", "medium", "low"])
    p.add_argument("--notes", default="")
    p.add_argument("--conditions", default=None,
                   help="e.g. ev_ebitda<=9;pb_ratio<=1.5;mktcap_to_assets<=0.5")
    p.add_argument("--match-mode", default="all", choices=["all", "any"])
    p.add_argument("--metric", default=None, help="for metric_below/above")
    p.set_defaults(func=cmd_add)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
