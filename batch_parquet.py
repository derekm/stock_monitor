#!/usr/bin/env python3
"""Batch convert remaining CSV outputs to parquet in source files."""
import re
from pathlib import Path

FILES = [
    # (path, patterns_to_replace)
    # fisher_index.py
    ("fisher_index.py", [
        (r'OUT_FILE = DATA_DIR / "fisher_indexes\.csv"', r'OUT_FILE = DATA_DIR / "fisher_indexes.parquet"'),
        (r'OUT_PQ = DATA_DIR / "fisher_indexes\.parquet"', r''),  # remove duplicate
        (r'out\[rate_cols\]\.to_csv\(DATA_DIR / "fisher_rate_decomposition\.csv", index=False\)', 
         r'out[rate_cols].to_parquet(DATA_DIR / "fisher_rate_decomposition.parquet", index=False)'),
        (r'print\(f"Wrote \{OUT_FILE\} and fisher_rate_decomposition\.csv \(\{len\(out\)\} rows\)"\)',
         r'print(f"Wrote {OUT_FILE} and fisher_rate_decomposition.parquet ({len(out)} rows)")'),
    ]),
    # run_fisher_duckdb.py
    ("run_fisher_duckdb.py", [
        (r'OUT_CSV = DATA_DIR / "fisher_indexes_duckdb\.csv"', r'OUT_CSV = DATA_DIR / "fisher_indexes_duckdb.parquet"'),
        (r'out\.to_csv\(OUT_CSV, index=False\)', r'out.to_parquet(OUT_CSV, index=False)'),
    ]),
    # factor_rotation_defense.py
    ("factor_rotation_defense.py", [
        (r'GROUPS = DATA_DIR / "factor_groups\.csv"', r'GROUPS = DATA_DIR / "factor_groups.parquet"'),
        (r'MEMBERS = DATA_DIR / "factor_group_members\.csv"', r'MEMBERS = DATA_DIR / "factor_group_members.parquet"'),
        (r'OUT_W = DATA_DIR / "factor_rotation_weights\.csv"', r'OUT_W = DATA_DIR / "factor_rotation_weights.parquet"'),
        (r'OUT_PERF = DATA_DIR / "factor_rotation_performance\.csv"', r'OUT_PERF = DATA_DIR / "factor_rotation_performance.parquet"'),
        (r'OUT_SLEEVE = DATA_DIR / "factor_sleeve_returns\.csv"', r'OUT_SLEEVE = DATA_DIR / "factor_sleeve_returns.parquet"'),
        (r'GROUPS\.write_text\(cat\.to_csv\(index=False\), encoding="utf-8"\)', r'cat.to_parquet(GROUPS, index=False)'),
        (r'MEMBERS\.write_text\(mem\.to_csv\(index=False\), encoding="utf-8"\)', r'mem.to_parquet(MEMBERS, index=False)'),
        (r'wdf\.to_csv\(OUT_W, index=False\)', r'wdf.to_parquet(OUT_W, index=False)'),
        (r'perf_df\.to_csv\(OUT_PERF, index=False\)', r'perf_df.to_parquet(OUT_PERF, index=False)'),
        (r'full_sret\.reset_index\(\)\.rename\(columns=\{"index": "date"\}\)\.to_csv\(OUT_SLEEVE, index=False\)', 
         r'full_sret.reset_index().rename(columns={"index": "date"}).to_parquet(OUT_SLEEVE, index=False)'),
    ]),
    # crisis_correlation.py
    ("crisis_correlation.py", [
        (r'OUT = DATA_DIR / "crisis_correlation_summary\.csv"', r'OUT = DATA_DIR / "crisis_correlation_summary.parquet"'),
        (r'OUT_PAIR = DATA_DIR / "crisis_correlation_pairs\.csv"', r'OUT_PAIR = DATA_DIR / "crisis_correlation_pairs.parquet"'),
        (r'OUT_TS = DATA_DIR / "crisis_avg_corr_timeseries\.csv"', r'OUT_TS = DATA_DIR / "crisis_avg_corr_timeseries.parquet"'),
        (r'out\.to_csv\(OUT, index=False\)', r'out.to_parquet(OUT, index=False)'),
        (r'pair_out\.to_csv\(OUT_PAIR, index=False\)', r'pair_out.to_parquet(OUT_PAIR, index=False)'),
        (r'ts_out\.to_csv\(OUT_TS, index=False\)', r'ts_out.to_parquet(OUT_TS, index=False)'),
    ]),
    # allpairs_correlations.py
    ("allpairs_correlations.py", [
        (r'OUT_ASSET = DATA_DIR / "allpairs_asset_corr_history\.csv"', r'OUT_ASSET = DATA_DIR / "allpairs_asset_corr_history.parquet"'),
        (r'OUT_SECTOR = DATA_DIR / "allpairs_sector_corr_history\.csv"', r'OUT_SECTOR = DATA_DIR / "allpairs_sector_corr_history.parquet"'),
        (r'OUT_ASSET_LATEST = DATA_DIR / "allpairs_asset_corr_latest\.csv"', r'OUT_ASSET_LATEST = DATA_DIR / "allpairs_asset_corr_latest.parquet"'),
        (r'OUT_SECTOR_LATEST = DATA_DIR / "allpairs_sector_corr_latest\.csv"', r'OUT_SECTOR_LATEST = DATA_DIR / "allpairs_sector_corr_latest.parquet"'),
        (r'OUT_SUMMARY = DATA_DIR / "allpairs_corr_summary\.csv"', r'OUT_SUMMARY = DATA_DIR / "allpairs_corr_summary.parquet"'),
        (r'mat\.to_csv\(OUT_ASSET_LATEST\)', r'mat.to_parquet(OUT_ASSET_LATEST)'),
        (r'mat\.to_csv\(OUT_SECTOR_LATEST\)', r'mat.to_parquet(OUT_SECTOR_LATEST)'),
        (r'asset_df\.to_csv\(OUT_ASSET, index=False\)', r'asset_df.to_parquet(OUT_ASSET, index=False)'),
        (r'sector_df\.to_csv\(OUT_SECTOR, index=False\)', r'sector_df.to_parquet(OUT_SECTOR, index=False)'),
        (r'sdf\.to_csv\(OUT_SUMMARY, index=False\)', r'sdf.to_parquet(OUT_SUMMARY, index=False)'),
    ]),
]

def main():
    for path, patterns in FILES:
        full_path = Path(path)
        if not full_path.exists():
            print(f"SKIP {path}: not found")
            continue
        content = full_path.read_text()
        original = content
        for old, new in patterns:
            content = re.sub(old, new, content)
        if content != original:
            full_path.write_text(content)
            print(f"UPDATED {path}")
        else:
            print(f"NO CHANGE {path}")

if __name__ == "__main__":
    main()