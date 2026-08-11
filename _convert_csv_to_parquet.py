#!/usr/bin/env python3
"""
Convert CSV producers/consumers to parquet across the repo.

For each .py script, transforms:
  - `.to_parquet(PATH)`  -> `.to_parquet(PATH.replace('.csv','.parquet'))` (drop index=)
  - `.read_parquet(PATH, ...)`      -> `.read_parquet(PATH.replace('.csv','.parquet'))` (drop csv-only kwargs)
  - string literals "x.csv"     -> "x.parquet" for DERIVED data files (not external sources)
  - constant defs `X = .../ "x.parquet"` -> ".parquet"

Derived CSVs = files that have a producer in-repo (from the producer map). External
source CSVs (factor_groups.csv, factor_group_members.csv, sp500_constituents.csv,
parse inputs) are LEFT as .csv.
"""
import re
from pathlib import Path

DATA = Path(".")

# CSVs that are genuine external source inputs -> keep .csv
EXTERNAL_SOURCE = {
    "factor_groups.csv", "factor_group_members.csv",
    "sp500_constituents.csv",
}
# Files produced/consumed by a chain; all others stay as-is unless producer says convert.
# We convert EVERY .csv reference that is NOT an external source.

READ_CSV_ONLY_KWARGS = [
    "parse_dates", "dtype", "encoding", "engine", "index_col", "na_values",
    "keep_default_na", "skiprows", "skipfooter", "thousands", "decimal",
    "quotechar", "quoting", "delimiter", "sep", "header", "names", "usecols",
    "squeeze", "low_memory", "error_bad_lines", "warn_bad_lines", "comment",
    "converters", "true_values", "false_values", "skip_blank_lines", "nrows",
]

def _to_parquet_call(match):
    # match.group(1) = full inner args text
    inner = match.group(1)
    # split top-level args (naive: split on commas not inside brackets)
    parts = _split_top(inner)
    path_part = parts[0].strip() if parts else ""
    # convert .csv literal / DATA_DIR/"x.parquet" / CONST path
    new_parts = []
    new_path = None
    for i, p in enumerate(parts):
        p = p.strip()
        if i == 0:
            new_path = p
            # transform the path expression: replace .csv with .parquet
            new_path = _replace_csv_in_expr(new_path)
            new_parts.append(new_path)
        else:
            # drop index= / header= (csv-only) kwargs for to_parquet
            if re.match(r'^(index|header|index_col|columns|sep|delimiter|encoding|compression|line_terminator|quotechar|quoting|date_format|doublequote)\s*=', p):
                continue
            # keep only index=False dropped; index=True also dropped
            new_parts.append(p)
    call = ", ".join(new_parts)
    return f".to_parquet({call})"

def _read_parquet_call(match):
    inner = match.group(1)
    parts = _split_top(inner)
    new_parts = []
    for i, p in enumerate(parts):
        p = p.strip()
        if i == 0:
            new_parts.append(_replace_csv_in_expr(p))
        else:
            # drop csv-only kwargs
            kw = re.match(r'^([A-Za-z_]+)\s*=', p)
            if kw and kw.group(1) in READ_CSV_ONLY_KWARGS:
                continue
            new_parts.append(p)
    return f".read_parquet({', '.join(new_parts)})"

def _replace_csv_in_expr(expr):
    # Replace ".csv" with ".parquet" inside a path expression
    # handle: "x.csv"  or  DATA_DIR / "x.parquet"
    if re.search(r'\.csv["\']', expr):
        expr = expr.replace('.csv', '.parquet')
    return expr

def _split_top(s):
    """Split on commas that are not inside brackets/quotes."""
    parts, depth, cur, quote = [], 0, "", None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch; cur += ch; continue
        if ch in "[({":
            depth += 1; cur += ch; continue
        if ch in "])}":
            depth -= 1; cur += ch; continue
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""; continue
        cur += ch
    if cur.strip():
        parts.append(cur)
    return parts

def process_file(path: Path):
    src = path.read_text(encoding="utf-8", errors="ignore")
    orig = src
    # 1. Constant definitions: X = DATA_DIR / "x.parquet"  and  X = "x.parquet"
    def const_repl(m):
        val = m.group(0)
        # only convert if not an external source filename
        if any(e in val for e in EXTERNAL_SOURCE):
            return val
        return val.replace('.csv', '.parquet')
    src = re.sub(r'(DATA_DIR\s*/\s*["\'][^"\']*\.csv["\'])', const_repl, src)
    src = re.sub(r'(=.*?["\']([A-Za-z0-9_]+\.csv)["\'])', const_repl, src)
    # 2. to_csv calls
    src = re.sub(r'\.to_csv\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _to_parquet_call, src)
    # 3. read_csv calls
    src = re.sub(r'\.read_csv\(([^()]*(?:\([^()]*\)[^()]*)*)\)', _read_parquet_call, src)
    # 4. bare string literals ".csv" that are file paths (in DATA_DIR / "x.parquet" style) already handled
    if src != orig:
        path.write_text(src, encoding="utf-8")
        return True
    return False

# Process all scripts except helpers and the CSV ingestors that should stay
for py in sorted(DATA.glob("*.py")):
    if py.name in ("_map_csv.py", "_scripts_csv.py", "csv_to_parquet.py"):
        continue
    if py.name in ("update_prices.py", "backfill_historical.py", "parse_sp500.py",
                   "sp_index_methodology.py", "sp_history_simulation.py",
                   "sp_universe_tracking.py", "reconcile_sp500.py", "parse_sp500_changes.py"):
        # these are ingestors/source handlers; skip unless they have obvious derived output
        pass
    if ".to_csv(" not in py.read_text(encoding="utf-8", errors="ignore") and \
       ".read_csv(" not in py.read_text(encoding="utf-8", errors="ignore"):
        continue
    try:
        if process_file(py):
            print(f"converted {py.name}")
    except Exception as e:
        print(f"  !! {py.name}: {e}")

print("done")
