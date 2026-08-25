# build_bogle_funds.py

Construct Bogle-style index funds from StockMonitor data — implementing
John C. Bogle's six principles using our full universe, Fisher chained
decomposition, and explicit cost/turnover tracking.

## Why it exists (rationale)

Bogle's core principles mapped to our toolkit:

| Bogle Principle | Implementation |
|-----------------|----------------|
| **Own the whole market** | TMI = PIT liquid universe (`docs/bogle_inclusion.md`), cap-weighted (S&P divisor continuity) + Fisher chained |
| **Minimize costs** | Explicit `expense_bps` (default 3) + `turnover_bps` (default 5) tracked per fund, per period |
| **Broad diversification** | TMI: ~4200 PIT avg (NMS/NYQ/NCM/NGM/ASE + mcap + $5M ADV); QMI: NM top-qtl + filing; QMI_STRICT: Buffett 15/15/1.0 + filing; BPI: defensive-sector EW |
| **Stay the course** | Fixed rebalance calendar (Q/SA/Y) with multi-day glide path |
| **Simplicity** | Four books, each with a parquet + turnover log |
| **Low turnover** | QMI 3.5%/yr, QMI_STRICT 4.5%/yr (liquid EW) |

## Four Funds

### TMI — Total Market Index (The "Own the Market" Fund)
- **Universe:** PIT liquid per `docs/bogle_inclusion.md` — `instrument_type=stock` + `exchange ∈ {NMS,NYQ,NCM,NGM,ASE}` + `daily_mcap[D]` exists + `ADV20 ≥ $5M` (IPOs enter when mcap posts, ~5–20d; ~4200 names PIT avg)
- **Weighting:** Cap-weighted (S&P-style divisor continuity) + Fisher chained variant
- **Rebalance:** Quarterly (41 rebalances over 10 years)
- **Cost layer:** 3 bps/yr expense + 5 bps per 100% turnover
- **10y result (2016-2026):** CAGR 48.97%, Vol 47.45%, Sharpe 1.03

### QMI — Quality Market Index (NM rank)
- **Universe:** NM `nm_score` top quintile (≥2 legs) + PIT liquid (`mcap + ADV + filing`) — see `docs/bogle_inclusion.md`
- **Weighting:** Equal-weight + Fisher chained
- **Rebalance:** Semi-annual
- **10y net (2016-08-22–2026-08-20):** CAGR 15.48%, vol 21.88%, Sharpe 0.71, max DD −42.7%; COVID −38.8%; 2022 −26.6%

### QMI_STRICT — Buffett cut
- **Universe:** ROE ≥ 15%, ROIC ≥ 15%, 0 ≤ D/E ≤ 1.0, same PIT liquid + filing (see `docs/bogle_inclusion.md`)
- **Weighting / rebalance:** same as QMI
- **10y net:** CAGR 23.50%, vol 19.94%, Sharpe 1.18, max DD −37.2%; COVID −37.2%; 2022 −17.3%
- Kept beside QMI: higher CAGR/Sharpe and milder 2022 than the broad NM book. COVID still worse than BPI.

### BPI — Bond Proxy Index (The "Stay the Course" Anchor)
- **Universe:** Defensive sectors (Utilities, Consumer Staples, Health Care, Real Estate, Communication Services)
- **Weighting:** Equal-weight + Fisher chained
- **Rebalance:** Annual
- **Cost layer:** Same as TMI
- **10y result:** CAGR 14.46%, Vol 27.93%, Sharpe 0.52

### PMI — Pink Market Index (The "Complete the Market" Complement)
- **Universe:** the inverse of TMI's exchange gate — `exchange ∉ {NMS,NYQ,NCM,NGM,ASE}` (OTC/gray: `PNK/OID/OQB/OQX/PCX/BTS/None`), `instrument_type=stock`, `last ≥ $1`, `ADV20 ≥ $100k`, PIT `daily_mcap[D]`. Filing-free, like TMI. See `docs/bogle_inclusion.md`
- **Weighting:** Equal-weight + **5% single-name cap** (`cap_weights`) + Fisher chained
- **Rebalance:** Quarterly + 7-day glide
- **Cost layer:** 5 bps expense / 8 bps turnover (OTC costs more than TMI's 3/5); overridable via `--expense-bps` / `--turnover-bps`
- **Why:** TMI owns the exchange-listed tape; PMI owns everything else. `TMI ∪ PMI` = complete market, `TMI ∩ PMI = ∅` by construction (one gate, `exchange_mode` flag). **Not** SMCI — SMCI would be a 13F-selected subset nested inside PMI's universe; PMI is the unselected complement.

## Usage

```bash
# Build all four funds (10-year lookback, save to parquet)
python build_bogle_funds.py --fund all --save --years 10

# Build single fund
python build_bogle_funds.py --fund tmi --save --years 10
python build_bogle_funds.py --fund qmi --save --years 10
python build_bogle_funds.py --fund bpi --save --years 10
python build_bogle_funds.py --fund pmi --save --years 10

# Custom cost parameters
python build_bogle_funds.py --fund tmi --save --years 10 \
    --expense-bps 5 --turnover-bps 8

# Dry run (no save)
python build_bogle_funds.py --fund tmi --years 5
```

### Options
- `--fund {tmi,qmi,qmi_strict,bpi,pmi,all}` — which fund(s) to build
- `--save` — write parquet outputs (required for persistence)
- `--years N` — lookback window in years (default: 10)
- `--expense-bps N` — annual expense ratio in basis points (default: 3)
- `--turnover-bps N` — turnover cost in basis points per 100% turnover (default: 5)

## Outputs

| File | Rows | Description |
|------|------|-------------|
| `bogle_tmi.parquet` | 2,531 | Daily levels, returns (gross/net), Fisher chained variant, cost drag |
| `bogle_tmi_turnover.parquet` | 200 | Rebalance dates, one-way turnover, turnover cost |
| `bogle_qmi.parquet` | 2,531 | Same structure, quality-screened universe |
| `bogle_qmi_turnover.parquet` | 20 | Semi-annual rebalances |
| `bogle_bpi.parquet` | 2,531 | Defensive sectors, equal-weight |
| `bogle_bpi_turnover.parquet` | 50 | Annual rebalances |
| `bogle_pmi.parquet` | ~2,531 | OTC/gray complement of TMI, equal-weight + 5% cap |
| `bogle_pmi_turnover.parquet` | ~41 | Quarterly rebalances |

### Schema (fund parquet)
| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading date |
| `fund` | string | `TMI`, `QMI`, `QMI_STRICT`, `BPI`, or `PMI` |
| `weight_method` | string | `cap_weighted`, `equal_weighted`, or `equal_weighted_capped` (PMI) |
| `level` | float | Index level (base=1000) |
| `ret_gross` | float | Gross daily return |
| `ret_net` | float | Net daily return (after expense + turnover) |
| `expense_drag` | float | Daily expense drag (bps) |
| `turnover_cost` | float | Daily turnover cost (bps) |
| `fisher_p` | float | Fisher price index component |
| `fisher_q` | float | Fisher quantity index component |
| `fisher_p_net` | float | Fisher price index net of costs |
| `nominal_sqrt_fisher` | float | √(F_P × F_Q) nominal path |

### Schema (turnover parquet)
| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Rebalance date |
| `fund` | string | Fund identifier |
| `turnover` | float | One-way turnover (fraction) |
| `turnover_cost_bps` | float | Turnover cost in basis points |
| `n_names` | int | Number of names in universe at rebalance |

## Fisher Chained Decomposition

Each fund includes the **Fisher chained** variant (our de-biased arm that
re-anchors every rebalance period):

```
Laspeyres (L): Σ(p_t · q_b) / Σ(p_{t-1} · q_b)
Paasche   (P): Σ(p_t · q_t) / Σ(p_{t-1} · q_t)
Fisher    (F): √(L · P)

Level_t = base_level · exp( Σ_{τ≤t} ln(F_τ) )
```

This splits each period's return into:
- **Price component (F_P):** valuation changes
- **Quantity component (F_Q):** capital structure / share count changes

Bogle would appreciate this transparency — cap-weighting hides the quantity
drift; Fisher chained makes it explicit.

## Cost Model

Two cost drags applied daily:

```python
# Expense drag (daily accrual)
expense_daily = (1 + ret_gross) * (1 - expense_bps / 10000 / 252)

# Turnover cost (only on rebalance days)
turnover_cost_daily = 1 - (turnover_fraction * turnover_bps / 10000)

# Net return
ret_net = (1 + ret_gross) * expense_daily * turnover_cost_daily - 1
```

Both are tracked explicitly so you can see the drag:
- TMI 10y: 0.30% total expense drag, 0.45% total turnover cost
- QMI 10y: 0.30% expense, 0.18% turnover
- BPI 10y: 0.30% expense, 0.08% turnover

## Rebalance Calendar

| Fund | Frequency | Glide Path |
|------|-----------|------------|
| TMI | Quarterly | 7-day linear glide |
| QMI | Semi-annual | 7-day linear glide |
| BPI | Annual | 7-day linear glide |

The glide path reduces market impact and timing luck — do it linearly over 7 days.

## Integration with Daily Automation

Added to `daily_automation_dag.yaml`:

```yaml
bogle_tmi:
  cmd: ["build_bogle_funds.py", "--fund", "tmi", "--save", "--years", "10"]
  timeout: 300
  desc: "Bogle Total Market Index (cap-weighted + Fisher chained)"

bogle_qmi:
  cmd: ["build_bogle_funds.py", "--fund", "qmi", "--save", "--years", "10"]
  timeout: 300
  desc: "Bogle Quality Market Index (quality-screened + Fisher chained)"

bogle_bpi:
  cmd: ["build_bogle_funds.py", "--fund", "bpi", "--save", "--years", "10"]
  timeout: 300
  desc: "Bogle Bond Proxy Index (defensive sectors, equal-weight)"

# Depend on polygon_prices for fresh data
bogle_tmi: [polygon_prices]
bogle_qmi: [polygon_prices]
bogle_bpi: [polygon_prices]

# Feed into export for dashboard
export: [..., bogle_tmi, bogle_qmi, bogle_bpi]
```

Run via daily automation:
```bash
python run_daily_automation.py --only bogle_tmi,bogle_qmi,bogle_bpi
```

## Key Design Decisions

1. **Full universe from `daily_prices`** — not `monitored_stocks` (which was
   incomplete). Now `monitored_stocks` is 100% coverage with yfinance sectors.

2. **Fisher chained on top of cap-weight** — we keep S&P's divisor continuity
   for the cap-weighted aggregate, then add our chained Fisher as a parallel
   transparent variant.

3. **Cost layer in the index math** — not post-hoc. Every daily return is
   explicitly net of both expense accrual and turnover events.

4. **Turnover tracked at rebalance** — one-way turnover computed from weight
   changes; stored for auditability.

5. **Equal-weight for QMI/BPI** — reduces concentration risk (Bogle: "diversify
   broadly"). TMI keeps cap-weight as the "market portfolio" benchmark.

6. **YAML-driven DAG** — no hardcoded fallback in `run_daily_automation.py`.
   The YAML is the single source of truth.

7. **One gate, two directions (`exchange_mode`)** — `liquid_names` takes
   `exchange_mode="include"` (TMI/QMI/BPI) or `"exclude"` (PMI) against the same
   `{NMS,NYQ,NCM,NGM,ASE}` set. TMI and PMI are therefore disjoint and jointly
   complete *by construction*, not by convention — the two universes cannot drift
   apart or double-count a name when the exchange set is edited.

8. **Shared index machinery (libraryified)** — the glide expansion and Fisher
   merge were copy-pasted in every fund; they now live once as importable
   helpers, so a fix lands in all four funds at the same time:

   | Helper | Purpose |
   |--------|---------|
   | `expand_glide_weights(weights, rebal_dates, index, n_days=7)` | rebalance-date weights → daily panel via 7-day linear glide (Hoffstein rebalance-luck fix) |
   | `attach_fisher(levels, prices, daily_weights, expense_bps, turnover_bps)` | merges the Fisher-chained de-biased arm; warns and returns the nominal path if the arm cannot be built, so a fund never dies on the Fisher leg |
   | `cap_weights(weights, max_weight)` | iterative single-name cap with redistribution; falls back to equal-weight when `n × cap < 1` (cap unreachable) |
   | `liquid_names(..., exchange_mode=...)` | the one PIT gate for every fund |

   Import them directly for research:
   ```python
   from build_bogle_funds import (load_prices, liquid_names, cap_weights,
                                  expand_glide_weights, attach_fisher,
                                  build_tmi, build_pmi)
   ```

## Related Programs

- `run_daily_automation.py` — orchestrator (loads DAG from YAML)
- `daily_automation_dag.yaml` — job definitions + dependencies
- `export_dashboard_data.py` — exports fund tables to dashboard
- `fisher_index.py` — core Fisher chained implementation (Python)
- `run_fisher_duckdb.py` — DuckDB system-of-record Fisher pipeline
- `index_math.py` (stockmagic) — S&P divisor + 19-variant parallel index math
- `index_registry.py` — universe resolution (`all`, `portfolio`, `sectors`, etc.)

## Verification

```bash
# Verify outputs
python -c "
import pandas as pd
for f in ['bogle_tmi', 'bogle_qmi', 'bogle_bpi']:
    df = pd.read_parquet(f'{f}.parquet')
    print(f'{f}: {len(df)} rows, {df[\"date\"].min()} to {df[\"date\"].max()}')
    print(f'  CAGR: {(df[\"level\"].iloc[-1]/1000)**(1/((df[\"date\"].iloc[-1]-df[\"date\"].iloc[0]).days/365.25))-1:.2%}')
"
```