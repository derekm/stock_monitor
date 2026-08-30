# regime_clustering.py

HRP / codependence asset clustering — Phase 1.5 (López de Prado) deliverable.

## Why it exists (rationale)

`docs/RESEARCH_INTEGRATION_PLAN.md` item 1.5 asks to "replace HMM in
`hmm_regime_detection.py` with López de Prado's Hierarchical Risk Parity +
regime clustering (codependence + distance correlation)", with the success
metric "regime clusters reduce within-cluster correlation dispersion by ≥20%".

**It is an addition, not a replacement — deliberately.** The HMM labels
**dates** by market-level features (`mkt_ret`, `vol21`, `avg_corr`) and its
`date → regime` series is consumed by `pass6`/`pass8`/`regime_serving`.
HRP clustering groups **assets** by codependence. Those are different objects;
deleting the HMM would destroy the date labeller that the forecasting stack
depends on. The ≥20%-dispersion bar is an *asset-grouping* metric, so what it
actually tests is: do codependence clusters group assets more tightly than the
incumbent grouping (GICS sector)? That is the A/B this script runs.

## Method

- **Correlation distance** (AFML): `d_ij = sqrt(0.5·(1 − ρ_ij))` — a true
  metric (verified: triangle inequality holds; ρ=1→0, ρ=0→0.7071, ρ=−1→1).
- **Distance correlation** (Székely, `--metric dcor`): 0 iff independent, so it
  catches **non-linear** codependence Pearson misses. Verified on `q` vs `q²`:
  Pearson 0.016, dCor 0.539.
- **Linkage** single/average/complete/ward on the distance matrix.
- **HRP quasi-diagonal seriation** (`getQuasiDiag`) orders leaves so similar
  assets sit adjacent (`hrp_order` column).
- **k defaults to the number of GICS sectors** so the A/B compares groupings of
  equal granularity. (k=1 trivially maximises dispersion; k=n eliminates it —
  either would make the comparison meaningless.)
- **Dispersion** = size-weighted std-dev of within-group pairwise correlations.
  Singleton groups are **excluded**, not scored as zero dispersion (they would
  otherwise flatter whichever grouping produced more of them).

## Measured result (2026-08-24)

398 liquid exchange-listed names, k = 11 = #GICS sectors, bar ≥20%:

| linkage | 3y | 5y |
|---|---|---|
| ward | **+28.1% PASS** | **+20.3% PASS** |
| average | **+24.6% PASS** | **+20.0% PASS** (exactly on the bar) |
| complete | **+22.3% PASS** | +18.7% **FAIL** |
| single | +12.4% **FAIL** | +7.3% **FAIL** |

**5/8 configs clear the bar; range +7.3% → +28.1%.** This is a **fragile pass**:
the default (average/5y) sits exactly on 20.0%, and single linkage fails badly.
Always report the linkage and lookback with the number — hence `--sweep`.

**Distance correlation earns its keep.** Controlled comparison (same 150 names,
3y, ward — only the metric changes): **dcor +29.3% vs corr +23.0% (+6.3pp)**.

**Clusters are economically real and finer than GICS.** Ward/5y separates
GICS "Healthcare" into **drug distributors (MCK/COR/CAH)**, **managed care
(UNH/CVS/ELV/CI/HUM)** and **pharma (LLY/JNJ/PFE/MRK/ABBV)** — three 100%-pure
clusters — plus **gaming (EA/TTWO)** and **clean energy (ENPH/FSLR/PLUG)**.

**But the partition is unbalanced:** one cluster holds **286 of 400 names (72%)**,
the rest are size 1–49. Part of the dispersion win comes from peeling off tight
niches while leaving a heterogeneous core, so the aggregate ≥20% flatters the
method. The niches are useful; do not treat `cluster` as a drop-in `sector`
replacement until a balanced-partition variant is measured.

## Usage

```bash
python regime_clustering.py --save                        # default: corr, average, 5y, 400 assets
python regime_clustering.py --metric dcor --years 3 --max-assets 150 --linkage ward
python regime_clustering.py --sweep --save                # linkage x lookback robustness table
python regime_clustering.py --relabel-peers               # mixed_* + GICS-mismatch → sector; no re-cluster
```

### Options
- `--metric {corr,dcor}` — Pearson distance, or distance correlation (non-linear)
- `--k N` — cluster count (default: number of GICS sectors, for a fair A/B)
- `--years N` — lookback (default 5)
- `--min-cov F` — minimum return coverage per ticker (default 0.95)
- `--linkage {single,average,complete,ward}` (default average)
- `--max-assets N` — cap by **median dollar volume**, not alphabetically (default 400)
- `--sweep` — run the robustness table instead of one config
- `--save` — write parquet outputs
- `--relabel-peers` — rebuild `peer_group` on the saved file: `mixed_*` and names whose GICS sector ≠ the cluster's dominant sector fall back to GICS. Does not re-cluster.

## Outputs

| File | Description |
|---|---|
| `regime_clusters.parquet` | `ticker`, `cluster`, `hrp_order`, `sector`, `metric`, `linkage`, `k` |
| `regime_cluster_dispersion.parquet` | sector vs cluster dispersion, reduction %, `bar_pct`, `passes`, `n_assets` |
| `regime_cluster_sweep.parquet` | `metric`, `linkage`, `years`, `reduction_pct`, `passes` (from `--sweep`) |

Schema family: see `docs/SCHEMAS.md` → *Regime clustering*.

## Pitfalls

- **`pct_change()` makes row 0 all-NaN.** `dropna(axis=1, how="any")` therefore
  deleted **every** column (0 assets survived). Drop the first row first, then
  gate columns by coverage.
- **Universe must be the liquid one.** Clustering all 16k tickers clusters OTC
  noise; this reuses the TMI exchange gate (`NMS/NYQ/NCM/NGM/ASE`, stock only).
- **`squareform` rejects tiny float asymmetry** — the distance matrix is
  symmetrized and zero-diagonalised before linkage.
- **dCor is O(n²) per pair**, so the sample is deterministically subsampled to
  `max_n=750` rows; keep `--max-assets` modest (≤150) when using `--metric dcor`.
- **`peer_group` is not `cluster_name`.** Cluster-level dispersion can look tight while AAPL (Technology) sits in `financial_services_76`. `_hybrid_peer_group` falls back to GICS when the cluster is `mixed_*` or the name's sector ≠ the cluster's dominant sector. Relabel with `--relabel-peers`; do not denylist mega-caps.
- **Do not read a single config as the result.** The bar outcome flips with linkage; use `--sweep`.

## Related programs

- `hmm_regime_detection.py` — the **date** regime labeller (not replaced by this)
- `subindustry_regime.py`, `peer_analytics.py`, `cross_section.py` — consumers of
  asset groupings that could use `cluster` instead of `sector`
- `portfolio_optimization.py` — HRP seriation (`hrp_order`) is the natural input
  to a hierarchical risk-parity allocation
- `docs/RESEARCH_INTEGRATION_PLAN.md` — Phase 1.5 item and bar
