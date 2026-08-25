#!/usr/bin/env python3
"""Export key analytics tables to dashboard_data/data.json for DuckDB-Wasm / static UI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "dashboard_data"
OUT_DIR.mkdir(exist_ok=True)

TABLES = [
    "hmm_transition_triggers",
    "threshold_logic_screen",
    "regime_corr_breakdown",
    "hmm_posterior_summary",
    "posterior_entropy_dynamics",
    "kalman_state_estimates",
    "vix_term_structure_summary",
    "vix_term_structure",
    "kalman_gain_summary",
    "kalman_gain_path",
    "tail_risk_hedge_crisis",
    "tail_risk_hedge_performance",
    "rolling_corr_stability_by_asset",
    "rolling_corr_avg_timeseries",
    "risk_metrics",
    "factor_sleeve_returns",
    "factor_rotation_weights",
    "factor_rotation_performance",
    "crisis_avg_corr_timeseries",
    "crisis_correlation_pairs",
    "crisis_correlation_summary",
    "allpairs_sector_corr_latest",
    "allpairs_asset_corr_latest",
    "allpairs_corr_summary",
    "defensive_value_etfs",
    "dual_pass_sensitivity",
    "dual_pass_stress",
    "inclusion_candidates",
    "exclusion_candidates",
    "near_dual_candidates",
    "defensive_value_exploration",
    "asset_correlation_matrix",
    "sector_correlation_matrix_latest",
    "preferred_metrics",
    "preferred_screen_hits",
    "preferred_metrics_history",
    "screen_backtest",
    "rolling_window_metrics",
    "rolling_screen_stability",
    "dupont_analysis",
    "vol_target_vs_risk_parity",
    "erc_gmv_summary",
    "growth_tech_vol_returns",
    "growth_tech_backtest_stats",
    "growth_tech_risk_models",
    "fisher_indexes",
    "black_litterman_weights",
    "dual_screen_external_candidates",
    "robust_covariance_summary",
    "aerospace_supply",
    "sector_performance_summary",
    "fisher_sector_baskets_latest",
    "fisher_sector_baskets",
    "sp500_sleeve",
    "buy_candidates_top",
    "buy_candidates",
    "price_flatlines",
    "suspected_splits",
    "fundamental_missingness",
    "ticker_jump_rates",
    "factor_panel_top",
    "factor_panel",
    "momentum_ic",
    "momentum_quintiles",
    "momentum_metrics",
    "schema_check_report",
    "black_litterman_views",
    "forecast_reliability_report",
    "inclusion_walkforward",
    "portfolio_risk_summary",
    "risk_metrics_ext",
    "rebalance_calendar",
    "monte_carlo_summary",
    "monte_carlo_path_stats",
    "forecast_backtest_metrics",
    "forecast_reliability_rank",
    # Peer analytics
    "peer_analytics_signals",
    "peer_group_summary",
    "peer_fundamental_trends",
    "peer_recovery_signals",
    # Sprint: pair engine, earnings catalyst, cross-section
    "pair_engine_pairs",
    "pair_engine_trades",
    "pair_engine_stats",
    "earnings_catalyst_signals",
    "earnings_drift_stats",
    "cross_section_rankings",
    "cross_section_returns",
    "cross_section_stats",
    "signal_aggregator_scores",
    "signal_aggregator_ic",
    "technical_signals",
    "economic_calendar",
    "estimate_revisions",
    "shadow_book",
    "shadow_lots",
    "filings_sentiment",
    "options_skew",
    "signal_model_oos",
    "signal_model_weights",
    # Regime-selected Granite forecasting (pass6/7)
    "regime_model_best",
    "regime_model_oos",
    "regime_model_best_rpt",
    "regime_model_oos_rpt",
    "rpt_vs_ibm_compare",
    "regime_calibration",
    # Implied cost-of-capital screen (Ohlson-Rueangsuwan 2026, RIV reduced form)
    "implied_r_screen",
    # Taleb layer: fat tails, gaps, ergodicity, fragility, barbell
    "tail_index",
    "portfolio_tail",
    "tail_dependence",
    "gap_risk",
    "gap_events",
    "ergodicity_ruin",
    "portfolio_ergodic",
    "fragility_screen",
    "macro_fragility",
    "macro_shock",
    "macro_sector_shock",
    "basket_members",
    "factor_groups",
    "factor_group_members",
    "sp500_constituents",
    "shock_ride",
    "shock_ride_tickers",
    "arista_metrics",
    "arista_signals",
    "arista_backtest",
    "ride_now",
    "subindustry_regime",
    "barbell_check",
    "hidden_optionality",
    # Dashboard tables (top-level payload keys)
    "value_trifecta",
    "holdings",
    "fundamentals",
    "low_ev_ebitda",
    "low_pb",
    "anomalies",
    "decision_notes",
    "forecasts_hmax",
    "forecasts",
    "asset_sector_corr",
    "corr_stability",
    "corr_stability_metrics",
    "hmm_regime_corr",
    "sector_corr",
    "index_backtest",
    "forecast_backtest",
    "factor_rotation_performance",
    "sharpe_comparison",
    "fisher_indexes",
    "fisher_rate_decomposition",
    "fisher_indexes_duckdb",
    "fisher_universes",
    "monitored_stocks",
    "sp_history_sim",
    "sp500_changes",
    "price_qty_panel",
    # Bogle sleeves: TMI (exchange-listed) + PMI (OTC/gray complement) = complete market
    "bogle_tmi",
    "bogle_tmi_turnover",
    "bogle_qmi",
    "bogle_qmi_turnover",
    "bogle_qmi_strict",
    "bogle_qmi_strict_turnover",
    "bogle_bpi",
    "bogle_bpi_turnover",
    "bogle_pmi",
    "bogle_pmi_turnover",
    # Phase 1.5 (Lopez de Prado): HRP / codependence asset clustering
    "regime_clusters",
    "regime_cluster_dispersion",
    "regime_cluster_sweep",
]


def load(name: str) -> pd.DataFrame | None:
    p = DATA_DIR / f"{name}.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def df_records(df: pd.DataFrame | None) -> list:
    if df is None or df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype(str)
    out = out.replace({np.nan: None, np.inf: None, -np.inf: None})
    return json.loads(out.to_json(orient="records", date_format="iso"))


def latest_fundamentals() -> pd.DataFrame:
    f = load("fundamentals")
    if f is None or f.empty:
        return pd.DataFrame()
    if "as_of_date" in f.columns:
        f = f.copy()
        f["as_of_date"] = pd.to_datetime(f["as_of_date"], errors="coerce")
        f = f.sort_values("as_of_date").groupby("ticker", as_index=False).tail(1)
    return f


def build_value_trifecta(fund: pd.DataFrame, pm: pd.DataFrame | None) -> list:
    if pm is not None and "trifecta_pass" in pm.columns:
        hits = pm[pm["trifecta_pass"] == True].copy()
        cols = [c for c in [
            "ticker", "sector", "ev_ebitda", "pb_ratio", "mktcap_to_assets",
            "in_portfolio", "decision", "composite_score",
        ] if c in hits.columns]
        if "name" not in hits.columns:
            hits["name"] = hits["ticker"]
        cols = ["ticker", "name"] + [c for c in cols if c not in ("ticker",)]
        return df_records(hits[cols] if cols else hits)
    if fund is None or fund.empty:
        return []
    m = fund.copy()
    for c in ("ev_ebitda", "pb_ratio", "mktcap_to_assets"):
        if c not in m.columns:
            return []
    hits = m[
        (m["ev_ebitda"].notna()) & (m["ev_ebitda"] <= 9)
        & (m["pb_ratio"].notna()) & (m["pb_ratio"] <= 1.5)
        & (m["mktcap_to_assets"].notna()) & (m["mktcap_to_assets"] <= 0.5)
    ].copy()
    hits["name"] = hits["ticker"]
    return df_records(hits)


def decision_notes(pm: pd.DataFrame | None, holdings: pd.DataFrame | None) -> list:
    notes = []
    if holdings is not None and not holdings.empty:
        n = len(holdings)
        tickers = ", ".join(holdings["ticker"].astype(str).tolist())
        notes.append({
            "title": f"Personal portfolio · {n} positions",
            "detail": f"Current holdings: {tickers}. Weights and P&L from trades.parquet / price marks.",
        })
    if pm is not None and not pm.empty and "decision" in pm.columns:
        core = pm[pm["decision"].astype(str).str.contains("INCLUDE_CORE", na=False)]
        if len(core):
            notes.append({
                "title": "Dual-pass / INCLUDE_CORE",
                "detail": "Clears Buffett quality + value trifecta: "
                          + ", ".join(core["ticker"].astype(str).tolist()),
            })
        tri = pm[pm.get("trifecta_pass", False) == True] if "trifecta_pass" in pm.columns else pd.DataFrame()
        if len(tri):
            notes.append({
                "title": "Value trifecta passers",
                "detail": "EV/EBITDA≤9, P/B≤1.5, MktCap/Assets≤0.5 → "
                          + ", ".join(tri["ticker"].astype(str).tolist()),
            })
        port = pm[pm.get("in_portfolio", False) == True] if "in_portfolio" in pm.columns else pd.DataFrame()
        if len(port) and "decision" in port.columns:
            notes.append({
                "title": "Held names · screen status",
                "detail": "; ".join(
                    f"{r.ticker}: {r.decision}" for r in port.itertuples()
                ),
            })
    if not notes:
        notes.append({
            "title": "Decision surface",
            "detail": "Load preferred_metrics and holdings to populate inclusion rationale.",
        })
    return notes


def main():
    payload: dict = {"generated": pd.Timestamp.now().isoformat(), "tables": {}}

    for name in TABLES:
        df = load(name)
        recs = df_records(df)
        payload["tables"][name] = recs
        print(f"  {name}: {len(recs)} rows exported (cap 500)")

    # Dual passers convenience
    pm = load("preferred_metrics")
    if pm is not None and "decision" in pm.columns:
        dual = pm[pm["decision"].astype(str).str.contains("INCLUDE_CORE", na=False)]
        payload["tables"]["dual_passers"] = df_records(dual)
        print(f"  dual_passers: {len(payload['tables']['dual_passers'])}")

    # Top-level objects for Decisions / Portfolio / Value tabs
    holdings = load("portfolio_holdings")
    fund = latest_fundamentals()
    # enrich fundamentals with name/sector/in_portfolio from preferred_metrics
    if not fund.empty:
        if "name" not in fund.columns:
            fund["name"] = fund["ticker"]
        if pm is not None:
            meta_cols = [c for c in ["ticker", "sector", "in_portfolio", "defensive_value_index",
                                      "growth_tech_index", "index_member"] if c in pm.columns]
            if meta_cols:
                fund = fund.merge(pm[meta_cols].drop_duplicates("ticker"), on="ticker", how="left",
                                  suffixes=("", "_pm"))
        stocks = load("monitored_stocks")
        if stocks is not None and "index_member" in stocks.columns:
            if "index_member" not in fund.columns:
                fund = fund.merge(
                    stocks[["ticker", "index_member"]].drop_duplicates("ticker"),
                    on="ticker", how="left",
                )
            else:
                # fill nulls from stocks
                sm = stocks.set_index("ticker")["index_member"]
                fund["index_member"] = fund.apply(
                    lambda r: r["index_member"] if r.get("index_member") == r.get("index_member") and r["index_member"] is not None and str(r["index_member"]) != "nan"
                    else sm.get(r["ticker"], False),
                    axis=1,
                )

    payload["holdings"] = df_records(holdings)
    payload["fundamentals"] = df_records(fund)
    payload["value_trifecta"] = build_value_trifecta(fund, pm)
    payload["decision_notes"] = decision_notes(pm, holdings)

    # Low EV / low PB lists for Value tab (positive multiples only — negatives are accounting artifacts)
    if not fund.empty:
        f2 = fund.copy()
        if "name" not in f2.columns:
            f2["name"] = f2["ticker"]
        # merge index_member from monitored_stocks when missing
        if "index_member" not in f2.columns or f2["index_member"].isna().all():
            stocks = load("monitored_stocks")
            if stocks is not None and "index_member" in stocks.columns:
                f2 = f2.drop(columns=[c for c in ["index_member"] if c in f2.columns], errors="ignore")
                f2 = f2.merge(
                    stocks[["ticker", "index_member"]].drop_duplicates("ticker"),
                    on="ticker", how="left",
                )
        low_ev = f2.dropna(subset=["ev_ebitda"])
        low_ev = low_ev[low_ev["ev_ebitda"] > 0].sort_values("ev_ebitda")
        low_pb = f2.dropna(subset=["pb_ratio"])
        low_pb = low_pb[low_pb["pb_ratio"] > 0].sort_values("pb_ratio")
        payload["low_ev_ebitda"] = df_records(low_ev)
        payload["low_pb"] = df_records(low_pb)
        # refresh fundamentals with index_member if we enriched
        payload["fundamentals"] = df_records(f2)
    else:
        payload["low_ev_ebitda"] = []
        payload["low_pb"] = []

    # Anomalies — strongest absolute scores first, then most recent
    anom = load("anomalies_tspulse")
    if anom is None:
        anom = load("anomalies")
    if anom is not None and not anom.empty:
        a = anom.copy()
        if "score" in a.columns:
            a["_abs"] = a["score"].abs()
            sort_cols = ["_abs"]
            ascending = [False]
            if "date" in a.columns:
                a["date"] = pd.to_datetime(a["date"], errors="coerce")
                sort_cols.append("date")
                ascending.append(False)
            a = a.sort_values(sort_cols, ascending=ascending).drop(columns=["_abs"], errors="ignore")
        payload["anomalies"] = df_records(a)
    else:
        payload["anomalies"] = []

    # ── Persist computed dashboard tables as CSV so DuckDB-Wasm can register them ──
    # (these are computed payload keys — not producer files — so the catalog only
    #  knows them if we write them to disk here)
    computed_dash_tables = {
        "holdings": holdings,
        "value_trifecta": pd.DataFrame(payload["value_trifecta"]) if payload.get("value_trifecta") else pd.DataFrame(),
        "low_ev_ebitda": pd.DataFrame(payload.get("low_ev_ebitda", [])) if payload.get("low_ev_ebitda") else pd.DataFrame(),
        "low_pb": pd.DataFrame(payload.get("low_pb", [])) if payload.get("low_pb") else pd.DataFrame(),
        "anomalies": pd.DataFrame(payload.get("anomalies", [])) if payload.get("anomalies") else pd.DataFrame(),
        "decision_notes": pd.DataFrame(payload.get("decision_notes", [])),
    }
    for tname, tdf in computed_dash_tables.items():
        if tdf is not None and len(tdf) > 0:
            tdf.to_parquet(DATA_DIR / f"{tname}.parquet")
            print(f"  wrote {tname}.csv ({len(tdf)} rows)")

    # Forecasts snapshot if present
    fc = load("forecasts_granite")
    if fc is not None and not fc.empty and "horizon" in fc.columns:
        hmax = fc["horizon"].max()
        payload["forecasts_hmax"] = df_records(fc[fc["horizon"] == hmax])
        payload["forecasts"] = df_records(fc)
    else:
        payload["forecasts_hmax"] = []
        payload["forecasts"] = []

    # Corr / sector convenience
    _asc = load("asset_sector_correlations")
    if _asc is None or (hasattr(_asc, "empty") and _asc.empty):
        _asc = load("asset_sector_corr")
    payload["asset_sector_corr"] = df_records(_asc)
    # Stability: prefer cross-asset pair stability, then correlation_stability_metrics
    stab = load("cross_asset_stability")
    if stab is None or (hasattr(stab, "empty") and stab.empty):
        stab = load("correlation_stability_metrics")
    if stab is None or (hasattr(stab, "empty") and stab.empty):
        stab = load("corr_stability")
    payload["corr_stability"] = df_records(stab)
    payload["corr_stability_metrics"] = payload["corr_stability"]
    payload["hmm_regime_corr"] = df_records(load("hmm_regime_correlations"))
    # Sector corr matrix: normalize index column to "sector"
    sc = load("sector_correlation_matrix_latest")
    if sc is not None and not sc.empty:
        if "sector" not in sc.columns:
            for c in list(sc.columns):
                if str(c).startswith("Unnamed") or c in ("index",):
                    sc = sc.rename(columns={c: "sector"})
                    break
            if "sector" not in sc.columns and sc.index.name:
                sc = sc.reset_index().rename(columns={sc.columns[0]: "sector"})
        payload["sector_corr"] = df_records(sc)
    else:
        payload["sector_corr"] = []
    # Prefer clean 1y sleeve stats; fall back to growth_tech suite numbers
    ib = load("index_backtest_stats")
    if ib is None or (hasattr(ib, "empty") and ib.empty):
        ib = load("index_backtest")
    if ib is None or (hasattr(ib, "empty") and ib.empty):
        ib = load("growth_tech_backtest_stats")
    payload["index_backtest"] = df_records(ib)
    payload["forecast_backtest"] = df_records(load("forecast_backtest_metrics"))
    payload["factor_rotation_performance"] = df_records(load("factor_rotation_performance"))
    payload["sharpe_comparison"] = df_records(load("sharpe_comparison"))

    # Fisher indexes + universe membership for dashboard
    fi = load("fisher_indexes")
    if fi is None:
        fi = load("fisher_indexes_duckdb")
    payload["fisher_indexes"] = df_records(fi)
    payload["fisher_rate_decomposition"] = df_records(load("fisher_rate_decomposition"))
    fid = load("fisher_indexes_duckdb")
    payload["fisher_indexes_duckdb"] = df_records(fid) if fid is not None else []
    # Universes from membership flags
    stocks = load("monitored_stocks")
    holdings = load("portfolio_holdings")
    universes = {}
    if holdings is not None and "ticker" in holdings.columns:
        universes["portfolio"] = holdings["ticker"].astype(str).str.upper().tolist()
    if stocks is not None:
        if "index_member" in stocks.columns:
            universes["fertilizer"] = stocks.loc[stocks["index_member"] == True, "ticker"].astype(str).str.upper().tolist()
        if "defensive_value_index" in stocks.columns:
            universes["defensive"] = stocks.loc[stocks["defensive_value_index"] == True, "ticker"].astype(str).str.upper().tolist()
        if "growth_tech_index" in stocks.columns:
            universes["growth_tech"] = stocks.loc[stocks["growth_tech_index"] == True, "ticker"].astype(str).str.upper().tolist()
        if "sector" in stocks.columns:
            for sec, key in [("Materials", "Materials")]:
                universes[key] = stocks.loc[stocks["sector"] == sec, "ticker"].astype(str).str.upper().tolist()
    if fi is not None and "universe" in fi.columns:
        for u in fi["universe"].dropna().unique():
            universes.setdefault(str(u), universes.get(str(u), []))
    payload["fisher_universes"] = universes
    ms = load("monitored_stocks")
    payload["monitored_stocks"] = df_records(ms)

    # S&P 500 historical inclusion/exclusion simulation (ours vs actuals, w/ removals)
    try:
        import sp_history_simulation as shs
        sim_rows = shs.simulate("2024-01-01", None)
        payload["sp_history_sim"] = [
            {
                "date": r["date"],
                "n_actual": r["n_actual"],
                "n_predicted": r["n_predicted"],
                "true_positives": r["true_positives"],
                "false_positives": r["false_positives"],
                "false_negatives": r["false_negatives"],
                "precision": r["precision"],
                "recall": r["recall"],
                "agreement": r["agreement"],
            }
            for r in sim_rows
        ]
    except Exception as e:
        payload["sp_history_sim"] = []
        print("sp_history_sim skip", e)
    # Raw add/remove change log (real events)
    try:
        chg = load("sp500_changes")
        payload["sp500_changes"] = df_records(chg)
    except Exception as e:
        payload["sp500_changes"] = []
        print("sp500_changes skip", e)

    # Compact price/qty panel for JS recompute (last ~400 days, tracked names)
    try:
        px = load("daily_prices")
        if px is not None and not px.empty:
            px = px.copy()
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            cutoff = px["date"].max() - pd.Timedelta(days=420)
            px = px[px["date"] >= cutoff]
            tick_set = set()
            for v in universes.values():
                tick_set.update(v)
            if tick_set:
                px = px[px["ticker"].astype(str).str.upper().isin(tick_set)]
            panel = px.rename(columns={"close": "close", "volume": "volume"})[
                [c for c in ["date", "ticker", "close", "volume"] if c in px.columns]
            ].copy()
            mc = load("daily_mcap")
            if mc is not None and not mc.empty:
                mc = mc.copy()
                mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
                mc = mc[mc["date"] >= cutoff]
                if tick_set:
                    mc = mc[mc["ticker"].astype(str).str.upper().isin(tick_set)]
                panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
                panel = panel.merge(mc[["date", "ticker", "market_cap"]], on=["date", "ticker"], how="left")
            panel["date"] = panel["date"].astype(str).str.slice(0, 10)
            payload["price_qty_panel"] = df_records(panel)
        else:
            payload["price_qty_panel"] = []
    except Exception as e:
        payload["price_qty_panel"] = []
        print("price_qty_panel skip", e)


    out = OUT_DIR / "data.json"
    out.write_text(json.dumps(payload, default=str))
    print(f"Wrote {out} ({out.stat().st_size/1024:.1f} KB)")
    try:
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_DIR / "build_data_catalog.py")], cwd=str(DATA_DIR), check=False)
    except Exception as e:
        print("catalog", e)
    print(f"  holdings={len(payload['holdings'])} trifecta={len(payload['value_trifecta'])} "
          f"fund={len(payload['fundamentals'])} notes={len(payload['decision_notes'])}")


if __name__ == "__main__":
    main()
