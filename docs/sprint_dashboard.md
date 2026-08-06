# sprint_dashboard.html

Static page rendering the sprint engines (pair engine, earnings catalyst,
cross-section, signal aggregator) from `dashboard_data/data.json`.

## Why it exists (rationale)

The main `index.html` is a large single-page app; rather than risk editing its
172KB of tightly-coupled render logic, this page follows the `single_stock.html`
pattern — a focused static page that fetches `dashboard_data/data.json` and
renders the new tables.

## Sections

- **Signal aggregator**: per-family IC + normalized weights; composite leaders
- **Pair engine**: selected pairs (FDR + half-life), pair stats net of costs
- **Cross-section**: OOS stats vs EW-long baseline; last rebalance long/short
- **Earnings catalyst**: PEAD drift buckets; top catalyst scores

## Usage

Serve alongside the other static pages (the `start_dashboard.sh` static server
or any static file server rooted at the repo). No build step.

## Outputs

None (read-only consumer of `dashboard_data/data.json`).

## Related programs

- `export_dashboard_data.py` — produces the JSON it reads
- `single_stock.html` — sibling single-name page
- `pair_engine.py` / `earnings_catalyst.py` / `cross_section.py` /
  `signal_aggregator.py` — the engines whose tables it renders
