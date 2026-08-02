#!/usr/bin/env python3
"""One-time helper: create the initial alerts_config.parquet with sensible default rules."""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path(__file__).parent
CONFIG_FILE = DATA_DIR / "alerts_config.parquet"

rules = [
    # --- Price level alerts (absolute thresholds) ---
    {"rule_id": "CF_above_130", "ticker": "CF", "rule_type": "price_above", "param1": 130.0, "param2": None, "enabled": True, "priority": "high", "notes": "CF breakout above recent resistance"},
    {"rule_id": "CF_below_110", "ticker": "CF", "rule_type": "price_below", "param1": 110.0, "param2": None, "enabled": True, "priority": "medium", "notes": "CF support test"},
    {"rule_id": "NTR_above_75", "ticker": "NTR", "rule_type": "price_above", "param1": 75.0, "param2": None, "enabled": True, "priority": "high", "notes": "Nutrien strength"},
    {"rule_id": "MOS_above_25", "ticker": "MOS", "rule_type": "price_above", "param1": 25.0, "param2": None, "enabled": True, "priority": "medium", "notes": "Mosaic recovery"},
    {"rule_id": "ICL_below_4_5", "ticker": "ICL", "rule_type": "price_below", "param1": 4.5, "param2": None, "enabled": True, "priority": "medium", "notes": "ICL weakness"},
    {"rule_id": "UAN_above_130", "ticker": "UAN", "rule_type": "price_above", "param1": 130.0, "param2": None, "enabled": True, "priority": "high", "notes": "UAN momentum"},
    {"rule_id": "SMG_above_75", "ticker": "SMG", "rule_type": "price_above", "param1": 75.0, "param2": None, "enabled": True, "priority": "medium", "notes": "Scotts breakout"},
    {"rule_id": "CTVA_above_95", "ticker": "CTVA", "rule_type": "price_above", "param1": 95.0, "param2": None, "enabled": True, "priority": "high", "notes": "Corteva strength"},

    # --- Daily % change alerts ---
    {"rule_id": "any_up_5pct", "ticker": "*", "rule_type": "pct_change_above", "param1": 5.0, "param2": 1, "enabled": True, "priority": "high", "notes": "Any monitored name +5% in 1 day"},
    {"rule_id": "any_down_5pct", "ticker": "*", "rule_type": "pct_change_below", "param1": -5.0, "param2": 1, "enabled": True, "priority": "high", "notes": "Any monitored name -5% in 1 day"},
    {"rule_id": "any_up_8pct_3d", "ticker": "*", "rule_type": "pct_change_above", "param1": 8.0, "param2": 3, "enabled": True, "priority": "medium", "notes": "+8% over 3 trading days"},
    {"rule_id": "any_down_8pct_3d", "ticker": "*", "rule_type": "pct_change_below", "param1": -8.0, "param2": 3, "enabled": True, "priority": "medium", "notes": "-8% over 3 trading days"},

    # --- Moving-average crossover / distance ---
    {"rule_id": "CF_above_sma20", "ticker": "CF", "rule_type": "above_sma", "param1": 20, "param2": None, "enabled": True, "priority": "low", "notes": "CF price > 20-day SMA"},
    {"rule_id": "NTR_above_sma20", "ticker": "NTR", "rule_type": "above_sma", "param1": 20, "param2": None, "enabled": True, "priority": "low", "notes": "NTR price > 20-day SMA"},
    {"rule_id": "MOS_above_sma20", "ticker": "MOS", "rule_type": "above_sma", "param1": 20, "param2": None, "enabled": True, "priority": "low", "notes": "MOS price > 20-day SMA"},
    {"rule_id": "index_members_below_sma20", "ticker": "INDEX", "rule_type": "below_sma", "param1": 20, "param2": None, "enabled": True, "priority": "medium", "notes": "Any index member below its 20-day SMA"},

    # --- New high / low ---
    {"rule_id": "new_20d_high", "ticker": "*", "rule_type": "new_high", "param1": 20, "param2": None, "enabled": True, "priority": "medium", "notes": "New 20-day high"},
    {"rule_id": "new_20d_low", "ticker": "*", "rule_type": "new_low", "param1": 20, "param2": None, "enabled": True, "priority": "medium", "notes": "New 20-day low"},

    # --- Volume spike (relative to recent average) ---
    {"rule_id": "vol_spike_2x", "ticker": "*", "rule_type": "volume_spike", "param1": 2.0, "param2": 10, "enabled": True, "priority": "low", "notes": "Volume > 2× 10-day average"},
]

df = pd.DataFrame(rules)
df["created"] = pd.Timestamp.now()
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, CONFIG_FILE)
print(f"Created {CONFIG_FILE} with {len(df)} rules")
print(df[["rule_id", "ticker", "rule_type", "param1", "enabled", "priority"]].to_string(index=False))
