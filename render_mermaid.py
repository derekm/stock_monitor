#!/usr/bin/env python3
"""render_mermaid.py — render .mmd Mermaid sources to PNG via mermaid.ink.

Used to produce the framework images for the repo. Writes:
  <name>.mmd   (source, versioned in repo)
  <name>.png   (rendered image, delivered to the user)

Usage:
  python render_mermaid.py
"""
import base64
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

DIAGRAMS = {}

DIAGRAMS["framework_architecture"] = """%%{init: {"theme": "dark", "themeVariables": {"background": "#020617", "primaryColor": "#0f172a", "primaryTextColor": "#e2e8f0", "primaryBorderColor": "#334155", "lineColor": "#475569", "secondaryColor": "#0c2a3a", "tertiaryColor": "#0f1f14", "fontFamily": "monospace", "fontSize": "14px"}}}%%
flowchart TB
    subgraph DATA["DATA SPINE - parquet tables"]
        P1["daily_prices/<br/>adj_close, split-adjusted"]
        P2["fundamentals.parquet<br/>EDGAR XBRL + yfinance"]
        P3["monitored_stocks.parquet<br/>GICS sectors"]
        P4["earnings_calendar.parquet"]
        P5["portfolio_holdings / trades"]
    end

    subgraph INGEST["INGESTION"]
        I1["update_prices.py<br/>yfinance daily"]
        I2["update_fundamentals.py<br/>fetch-history"]
        I3["backfill_edgar.py<br/>SEC XBRL companyfacts"]
        I4["update_polygon.py<br/>key-gated"]
        I5["economic_calendar.py<br/>FOMC + expiries"]
    end

    subgraph ENGINES["ANALYTICS ENGINES"]
        E1["hmm_regime_detection<br/>HMM 3-state"]
        E2["preferred_metrics<br/>quality/value/leverage"]
        E3["peer_analytics<br/>peer groups + trends"]
        E4["earnings_catalyst<br/>surprise + drift"]
        E5["pair_engine<br/>cointegration pairs"]
        E6["cross_section<br/>cross-sectional ranks"]
        E7["cost_model<br/>10bps + borrow"]
    end

    subgraph SIGNALS["SIGNAL AGGREGATION"]
        S1["signal_aggregator.py<br/>OOS IC-derived weights"]
        S2["signal_model.py<br/>GradientBoosting blend"]
        S3["technical_signals<br/>RSI/MACD/Bollinger"]
        S4["options_skew<br/>IV skew + put/call"]
        S5["estimate_revisions"]
        S6["filings_sentiment<br/>8-K lexicon"]
    end

    subgraph FORECAST["GRANITE TTM FORECASTING"]
        F1["pass5 / pass5_sweep<br/>direction vs persistence"]
        F2["pass6 regime-selected<br/>per-regime models"]
        F3["pass7 experiment matrix<br/>boundary/composition/lr"]
        F4["regime_serving.py<br/>checkpoint serving"]
        F5["forecast_granite.py<br/>ensemble + MC-dropout"]
        F6["analyze_granite_forecasts<br/>signal_gated BULL/BEAR"]
    end

    subgraph DECIDE["DECISIONS & PORTFOLIO"]
        D1["buy_candidates.py<br/>composite scoring"]
        D2["portfolio_optimization<br/>ERC / vol-target"]
        D3["shadow_book.py<br/>paper trade + kill switches"]
        D4["perf_metrics.py<br/>Sharpe/Calmar/capacity"]
    end

    subgraph OUT["OUTPUT"]
        O1["export_dashboard_data.py<br/>198 resources"]
        O2["data.json + data_catalog"]
        O3["DuckDB-Wasm dashboard"]
    end

    INGEST --> DATA
    DATA --> ENGINES
    ENGINES --> SIGNALS
    ENGINES --> FORECAST
    SIGNALS --> DECIDE
    FORECAST --> DECIDE
    DECIDE --> OUT
    I3 -.-> P2
    I1 -.-> P1
    E7 -.-> E5
    E7 -.-> E6"""

DIAGRAMS["daily_automation_dag"] = """%%{init: {"theme": "dark", "themeVariables": {"background": "#020617", "primaryColor": "#0f172a", "primaryTextColor": "#e2e8f0", "primaryBorderColor": "#334155", "lineColor": "#475569", "secondaryColor": "#0c2a3a", "tertiaryColor": "#0f1f14", "fontFamily": "monospace", "fontSize": "13px"}}}%%
flowchart LR
    HMM["hmm<br/>regime"] --> REB["rebalance<br/>calendar"]
    PREF["preferred<br/>metrics"] --> INCL["inclusion<br/>criteria"]
    PREF --> STRESS["stress<br/>dual-pass"]
    INCL --> STRESS
    PREF --> RISK["risk_enrich"]
    PREF --> ROLL["rolling<br/>window"]
    RISK --> ROLL
    PREF --> ROLC["rolling_corr"]
    RISK --> ROLC
    ROLL --> TAIL["tail_hedge"]
    HMM --> TAIL
    PREF --> AP["allpairs<br/>correlations"]
    PREF --> INCL2["screen_bt<br/>fundamentals bt"]
    INCL --> INCL2
    PREF --> DUP["dupont"]
    DUP --> GROW["growth_tech"]
    PREF --> GROW
    PREF --> PEER["peer<br/>analytics"]
    GROW --> ERN["earnings<br/>catalyst"]
    PEER --> ERN
    PEER --> PAIR["pair_engine"]
    ERN --> PAIR
    PEER --> CROSS["cross_section"]
    ERN --> CROSS
    PAIR --> CROSS
    CROSS --> AGG["signal_aggregator"]
    ERN --> AGG
    PAIR --> AGG
    PEER --> AGG
    PREF --> AGG
    AGG --> TECH["technical<br/>signals"]
    AGG --> SHAD["shadow_book"]
    PREF --> SHAD
    EC["econ_cal"] --> EXP["export<br/>dashboard"]
    ERV["est_rev"] --> EXP
    AGG --> EXP
    TECH --> EXP
    SHAD --> EXP"""

DIAGRAMS["signal_stack"] = """%%{init: {"theme": "dark", "themeVariables": {"background": "#020617", "primaryColor": "#0f172a", "primaryTextColor": "#e2e8f0", "primaryBorderColor": "#334155", "lineColor": "#475569", "secondaryColor": "#0c2a3a", "tertiaryColor": "#0f1f14", "fontFamily": "monospace", "fontSize": "14px"}}}%%
flowchart TB
    subgraph FAM["SIGNAL FAMILIES"]
        A1["preferred<br/>fundamental quality"]
        A2["peer<br/>relative value"]
        A3["cross<br/>cross-sectional rank"]
        A4["pairs<br/>cointegration z"]
        A5["earnings<br/>surprise + drift"]
        A6["technical / options /<br/>revisions / filings"]
    end

    subgraph AGG["AGGREGATION (OOS-IC WEIGHTED)"]
        B1["per-regime IC<br/>high_vol_stress: peer 52.9%<br/>cross 39.1% earnings 8%"]
        B2["signal_model.py<br/>GBM blend IC 0.237<br/>vs composite 0.152"]
    end

    subgraph USE["CONSUMPTION"]
        C1["buy_candidates.py<br/>composite + gates"]
        C2["shadow_book.py<br/>paper PnL"]
        C3["regime-gated<br/>BULL* / BEAR*"]
    end

    FAM --> AGG
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    A5 --> B1
    A6 --> B2
    B1 --> USE
    B2 --> USE
    B1 --> C1
    B1 --> C2
    B2 --> C1"""

DIAGRAMS["regime_forecasting"] = """%%{init: {"theme": "dark", "themeVariables": {"background": "#020617", "primaryColor": "#0f172a", "primaryTextColor": "#e2e8f0", "primaryBorderColor": "#334155", "lineColor": "#475569", "secondaryColor": "#0c2a3a", "tertiaryColor": "#0f1f14", "fontFamily": "monospace", "fontSize": "13px"}}}%%
flowchart LR
    subgraph RES["RESEARCH (HONEST OOS)"]
        R1["pass5: direction beats<br/>persistence 12/12"]
        R2["pass5_sweep: 648 exps<br/>stride=1 best, steps flat"]
        R3["pass6: per-regime models<br/>+26 to +42pt excess"]
        R4["pass7: boundary/composition<br/>matrix - cap=100 robust"]
    end

    subgraph SEL["SELECTION"]
        S1["regime_model_best.csv<br/>best config per ticker x regime"]
        S2["hmm_regime_states.csv<br/>current regime"]
    end

    subgraph SERVE["SERVING"]
        V1["regime_serving.py<br/>serve_regime_model()"]
        V2["checkpoints/regime/<br/>TICKER__regime__steps.pt"]
        V3["ensemble 0.5 general<br/>+ 0.5 regime model"]
    end

    subgraph OUT2["FORECAST OUTPUT"]
        O1["forecasts_granite.csv<br/>BULL/BEAR + signal_gated"]
        O2["regime_dir_h10..h96<br/>per-span direction"]
        O3["forecast_std<br/>MC-dropout band"]
        O4["ckpt age + calibration<br/>staleness warning"]
    end

    R1 --> R3
    R2 --> R3
    R3 --> R4
    R3 --> S1
    R4 --> S1
    S2 --> V1
    S1 --> V1
    V1 --> V2
    V1 --> V3
    V2 --> V3
    V3 --> OUT2"""


def render(name: str, code: str) -> Path | None:
    # kroki.io renders mermaid POST bodies (no URL-length limit like
    # mermaid.ink's GET path)
    req = urllib.request.Request(
        "https://kroki.io/mermaid/png",
        data=code.encode(),
        headers={"Content-Type": "text/plain", "User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        png = OUT / f"{name}.png"
        png.write_bytes(data)
        mmd = OUT / f"{name}.mmd"
        mmd.write_text(code, encoding="utf-8")
        print(f"OK  {name}: {len(data):,} bytes -> {png.name}")
        return png
    except Exception as e:
        print(f"ERR {name}: {e}")
        return None


def main():
    for name, code in DIAGRAMS.items():
        render(name, code)
        time.sleep(1.5)  # be polite to the API


if __name__ == "__main__":
    main()
