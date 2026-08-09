#!/usr/bin/env python3
"""Build data_catalog.json listing all CSV/Parquet resources for the dashboard."""
from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "dashboard_data" / "data_catalog.json"
OUT.parent.mkdir(exist_ok=True)

SKIP_DIRS = {".git", "node_modules", "logs", "__pycache__", "dashboard_data"}


def main():
    files = []
    for p in sorted(DATA_DIR.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in {".csv", ".parquet", ".json"}:
            continue
        # skip huge raw caches if any
        rel = p.relative_to(DATA_DIR).as_posix()
        if rel.startswith("dashboard_data/") and p.name not in ("data.json", "data_catalog.json"):
            pass
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size <= 0:
            # 0-byte parquet/csv would break DuckDB registration ("too small to be a parquet file")
            print(f"skip {rel} (0 bytes)")
            continue
        files.append({
            "name": p.stem,
            "filename": p.name,
            "path": rel,
            "url": rel,  # relative to static server root
            "kind": p.suffix.lower().lstrip("."),
            "size": size,
            "sql_name": p.stem.replace("-", "_").replace(".", "_"),
        })
    # de-dupe by sql_name preferring parquet over csv over json
    rank = {"parquet": 0, "csv": 1, "json": 2}
    best = {}
    for f in files:
        k = f["sql_name"]
        if k not in best or rank.get(f["kind"], 9) < rank.get(best[k]["kind"], 9):
            best[k] = f
    catalog = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "count": len(best),
        "files": sorted(best.values(), key=lambda x: x["sql_name"]),
    }
    OUT.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote {OUT} ({catalog['count']} resources)")
    return catalog


if __name__ == "__main__":
    main()
