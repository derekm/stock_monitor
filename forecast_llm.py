#!/usr/bin/env python3
"""
forecast_llm.py — LLM directional forecasts with Damodaran context.

Llama-3.2 1B/3B Instruct GGUF via llama-cpp-python on MX550.
JSON-grammar constrained output; rationale is two outcome-only sentences.
Job profiles (`value`, `exuberant`) are a config map. Innermost loop is
profile per ticker. `llm.reset()` before every generation so profiles are
not KV order-dependent. One long parquet; `profile` is the identity.
"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from llama_cpp import Llama, LlamaGrammar

MODEL_1B = Path(r"C:\Users\derek\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")
MODEL_3B = Path(r"C:\Users\derek\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf")
MODEL_PATH = MODEL_3B
_llm = None
_grammar = None

FORECAST_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["up", "sideways", "down"]},
        "prob": {"type": "number", "minimum": 0.10, "maximum": 0.90},
        "horizon_days": {"type": "integer", "minimum": 1, "maximum": 252},
        "rationale": {"type": "string"},
    },
    "required": ["direction", "prob", "horizon_days", "rationale"],
}

def _get_llm(model_path: Path | None = None):
    global _llm, _grammar, MODEL_PATH
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if _llm is None or path != MODEL_PATH:
        if _llm is not None:
            del _llm
            _llm = None
        MODEL_PATH = path
        print(f"Initializing {path.name} on NVIDIA MX550...")
        kwargs = dict(
            model_path=str(path),
            n_gpu_layers=99,
            n_ctx=1024,
            n_batch=512,
            n_ubatch=512,
            flash_attn=True,
            chat_format="llama-3",
            verbose=False,
        )
        try:
            _llm = Llama(**kwargs)
        except Exception as e:
            print(f"  full GPU offload failed ({e}); retry n_gpu_layers=20 n_batch=128")
            kwargs["n_gpu_layers"] = 20
            kwargs["n_batch"] = 128
            kwargs["n_ubatch"] = 128
            _llm = Llama(**kwargs)
        _grammar = LlamaGrammar.from_json_schema(json.dumps(FORECAST_JSON_SCHEMA))
    return _llm, _grammar

DATA_DIR = Path(__file__).parent
STATES = DATA_DIR / "hmm_regime_states.parquet"
LIFE_CYCLE = DATA_DIR / "life_cycle_stage.parquet"
WACC_FILE = DATA_DIR / "wacc_per_ticker.parquet"
FAIR_MULTIPLES = DATA_DIR / "fair_multiples.parquet"
QUALITY = DATA_DIR / "quality_scores.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
PREFERRED = DATA_DIR / "preferred_metrics.parquet"
MOMENTUM = DATA_DIR / "momentum_metrics.parquet"
IMPLIED_R = DATA_DIR / "implied_r_screen.parquet"
FRAGILITY = DATA_DIR / "fragility_screen.parquet"
ER_DECOMP = DATA_DIR / "expected_returns_decomp.parquet"
FRAGILITY_VETO = DATA_DIR / "fragility_veto.parquet"
NEWS_NOTES = DATA_DIR / "ticker_news_notes.parquet"
OUT = DATA_DIR / "forecast_llm.parquet"


def _er_snap() -> pd.DataFrame | None:
    """Last date only. Full file is 12M rows — do not pandas-read it."""
    if not ER_DECOMP.exists():
        return None
    import duckdb
    con = duckdb.connect()
    path = str(ER_DECOMP).replace("\\", "/")
    return con.execute(
        f"""
        with last as (select max(date) d from read_parquet('{path}'))
        select ticker, n_pillars, expected_return, carry, value, momentum, defensive
        from read_parquet('{path}') e, last
        where e.date = last.d and n_pillars >= 2
        """
    ).df()


def _ticker_snap(path: Path, cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        return None
    keep = ["ticker"] + [c for c in cols if c in df.columns]
    return df[keep].drop_duplicates("ticker", keep="last")

def load_states():
    df = pd.read_parquet(STATES)
    if len(df):
        x = df["date"].iloc[0]
        if isinstance(x, datetime):
            df["date"] = [v.date() if isinstance(v, datetime) else v for v in df["date"]]
    return df.sort_values("date")

def load_damodaran_context():
    lc = pd.read_parquet(LIFE_CYCLE)
    w = pd.read_parquet(WACC_FILE)
    fm = pd.read_parquet(FAIR_MULTIPLES)
    q = pd.read_parquet(QUALITY)
    for tbl in (lc, w, fm, q):
        if len(tbl):
            x = tbl["as_of_date"].iloc[0]
            if isinstance(x, datetime):
                tbl["as_of_date"] = [v.date() if isinstance(v, datetime) else v for v in tbl["as_of_date"]]
    ctx = lc.merge(w, on=["ticker","as_of_date"], how="left")
    ctx = ctx.merge(fm, on=["ticker","as_of_date"], how="left")
    ctx = ctx.merge(q[["ticker","as_of_date","quality_score","roic_wacc_spread"]], on=["ticker","as_of_date"], how="left")
    fund = pd.read_parquet(FUND)
    if len(fund):
        x = fund["as_of_date"].iloc[0]
        if isinstance(x, datetime):
            fund["as_of_date"] = [v.date() if isinstance(v, datetime) else v for v in fund["as_of_date"]]
    mix = [c for c in (
        "net_income_quarterly", "operating_income_quarterly", "gains_strategic_investments",
    ) if c in fund.columns]
    if mix:
        ctx = ctx.merge(fund[["ticker", "as_of_date"] + mix], on=["ticker", "as_of_date"], how="left")
    for path, cols in (
        (PREFERRED, ["decision", "buffett_pass", "trifecta_pass", "mos_pass", "nm_quality", "ev_ebitda"]),
        (MOMENTUM, ["mom_12_1", "ret_63d"]),
        (IMPLIED_R, ["implied_r_clean_pct"]),
        (FRAGILITY, ["fragile_flag"]),
        (FRAGILITY_VETO, ["veto_flag"]),
        (NEWS_NOTES, ["news_note"]),
    ):
        snap = _ticker_snap(path, cols)
        if snap is not None and len(snap.columns) > 1:
            ctx = ctx.merge(snap, on="ticker", how="left")
    er = _er_snap()
    if er is not None and len(er):
        ctx = ctx.merge(er, on="ticker", how="left")
    return ctx

def _fmt_money(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:+.2f} billion"
    if ax >= 1e6:
        return f"{x / 1e6:+.0f} million"
    return f"{x:,.0f}"


def _fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return f"{x:.{digits}%}"


def _fmt_cagr(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if x >= 0:
        return f"sales growing {_fmt_pct(x)} a year"
    return f"sales shrinking {_fmt_pct(abs(x))} a year"


def _fmt_fcf(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if x < 0:
        ax = abs(x)
        if ax >= 1:
            return f"spends {ax:.1f} dollars of cash for every dollar of sales"
        return f"spends extra cash equal to {_fmt_pct(ax)} of sales"
    if x < 0.10:
        return f"only {_fmt_pct(x)} of sales is leftover cash (cash left after running the business) — thin, not rich"
    return f"{_fmt_pct(x)} of sales is leftover cash (cash left after running the business)"


def _fmt_spread(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    pp = f"{abs(x) * 100:.1f}"
    if x >= 0:
        return f"profit beats the cost of capital (the return lenders and shareholders require) by {pp} points — value is being created"
    return f"profit misses the cost of capital (the return lenders and shareholders require) by {pp} points — value is being destroyed"


def _fmt_px(name, x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    verb = "rose" if x >= 0 else "fell"
    return f"{name} {verb} {_fmt_pct(abs(x))}"


@dataclass(frozen=True)
class JobProfile:
    name: str
    system: str
    omit_growth_if_expensive: bool
    omit_mom_if_expensive: bool
    omit_implied_r_if_expensive: bool
    press_lead: str


VALUE_SYSTEM = """You are a buy-side analyst. Write a two-sentence forecast, not a restatement.
Sentence 1: where the shares go over the foreseeable future.
Sentence 2: leftover cash, profit versus the cost of capital, or dollars of firm value per dollar of operating profit — not sales growth and not a 21-day bounce when the brief says do not own or the business spends cash.
Do not own means hold or sell. Missing the cost of capital is value destruction, not a bargain. Cheap (buyers underpay) is not a sell.
Too expensive means the shares do not go up: leftover cash and sales growth do not override buyers overpaying.
Press is last week's headlines, not leftover cash, and does not override do-not-own or too-expensive.
JSON keys: direction (up, sideways, down), prob (0.10-0.90), horizon_days, rationale."""

EXUBERANT_SYSTEM = """You are a buy-side analyst. Write a two-sentence forecast, not a restatement.
Sentence 1: where the shares go over the foreseeable future.
Sentence 2: leftover cash, cost of capital, dollars of firm value, or whether a crowd is still paying up — not a 21-day bounce when the brief says do not own or the business spends cash.
Do not own means hold or sell. Missing the cost of capital is value destruction, not a bargain. Cheap (buyers underpay) is not a sell.
Buyers overpaying does not by itself send the shares down. A crowd can keep paying up; that is exuberance, not a bargain and not a reason to own a do-not-own name.
Press is last week's headlines. It can describe a crowd; it does not override do-not-own.
JSON keys: direction (up, sideways, down), prob (0.10-0.90), horizon_days, rationale."""

PROFILES: dict[str, JobProfile] = {
    "value": JobProfile(
        name="value",
        system=VALUE_SYSTEM,
        omit_growth_if_expensive=True,
        omit_mom_if_expensive=True,
        omit_implied_r_if_expensive=True,
        press_lead="Press (not a reason to own): ",
    ),
    "exuberant": JobProfile(
        name="exuberant",
        system=EXUBERANT_SYSTEM,
        omit_growth_if_expensive=False,
        omit_mom_if_expensive=False,
        omit_implied_r_if_expensive=True,
        press_lead="Press (crowd tape, not leftover cash): ",
    ),
}


def build_brief(ticker, mkt_ret_21d, row, ticker_ret_21d=None, profile: JobProfile | None = None) -> str:
    """Prose dossier. Binding facts first; no labels the model can echo as a buy."""
    profile = profile or PROFILES["value"]
    bits = []
    dec = row.get("decision")
    avoid = isinstance(dec, str) and dec == "AVOID"
    def _flag(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return False
        return bool(x)
    veto = _flag(row.get("fragile_flag")) or _flag(row.get("veto_flag"))
    fcf = row.get("fcf_margin")
    burn = pd.notna(fcf) and float(fcf) < 0

    stage = row.get("life_cycle_stage")
    if avoid or veto:
        bits.append(f"{ticker}.")
        if avoid:
            bits.append("Do not own — a hold-or-sell instruction, not a comment on last week's price.")
        if veto:
            bits.append("Do not own: too crash-prone (the crash-risk test failed even if the recent price is up).")
    elif stage and stage != "Unclassified":
        stage_l = str(stage).lower()
        gloss = {
            "young growth": "early high-growth, often still spending cash",
            "high growth": "sales still ramping, not yet a cash cow",
            "mature growth": "still growing, no longer a startup",
            "mature stable": "grown-up cash business",
            "decline": "sales shrinking",
        }.get(stage_l)
        if gloss:
            bits.append(f"{ticker} is in a {stage_l} phase ({gloss}).")
        else:
            bits.append(f"{ticker} is in a {stage_l} phase.")
    else:
        bits.append(f"{ticker}.")

    px_s = _fmt_px("the shares", ticker_ret_21d)
    mkt_s = _fmt_px("the market", mkt_ret_21d)
    bounce = pd.notna(ticker_ret_21d) and float(ticker_ret_21d) > 0 and (avoid or burn or veto)
    if px_s and mkt_s:
        line = f"Over the last 21 days {px_s}; {mkt_s}."
        if bounce:
            line += " That 21-day bounce is not a reason to own."
        bits.append(line)
    elif px_s:
        line = f"Over the last 21 days {px_s}."
        if bounce:
            line += " That 21-day bounce is not a reason to own."
        bits.append(line)

    g3 = row.get("revenue_growth_3y")
    spread = row.get("roic_wacc_spread")
    if pd.isna(spread):
        wacc, roic = row.get("wacc"), row.get("roic")
        if pd.notna(wacc) and pd.notna(roic):
            spread = roic - wacc
    fcf_s = _fmt_fcf(fcf)
    spread_s = _fmt_spread(spread)
    g3_s = _fmt_cagr(g3)
    for s in (fcf_s, spread_s):
        if s:
            bits.append(s[0].upper() + s[1:] + ".")

    fair = row.get("fair_ev_ebitda")
    ev = row.get("ev_ebitda")
    expensive = False
    if pd.notna(fair) and float(fair) > 0 and pd.notna(ev) and float(ev) > 0:
        evf, ff = float(ev), float(fair)
        if evf >= ff * 1.15:
            expensive = True
            bits.append(
                f"Too expensive (buyers overpay): they pay {evf:.1f} dollars of firm value (equity plus net debt) per 1 dollar of operating profit; {ff:.1f} would be enough."
            )
        elif evf <= ff * 0.85:
            bits.append(
                f"Cheap (buyers underpay): they pay {evf:.1f} dollars of firm value (equity plus net debt) per 1 dollar of operating profit; {ff:.1f} would be fair."
            )
        else:
            bits.append(
                f"Price is about right at {evf:.1f} dollars of firm value (equity plus net debt) per 1 dollar of operating profit (fair is {ff:.1f})."
            )
    elif pd.notna(fair) and float(fair) > 0:
        bits.append(
            f"A justified ratio is {float(fair):.1f} dollars of firm value (equity plus net debt) per 1 dollar of operating profit — that is not the traded price."
        )
    hide_g = expensive and profile.omit_growth_if_expensive
    if g3_s and (not avoid) and (not burn) and (not hide_g):
        bits.append(g3_s[0].upper() + g3_s[1:] + ".")

    ni = row.get("net_income_quarterly")
    oi = row.get("operating_income_quarterly")
    if pd.notna(ni) and pd.notna(oi) and oi != 0 and abs(ni - oi) / abs(oi) >= 0.25:
        bits.append(
            f"The profit they reported {_fmt_money(ni)} is not the profit from running the business {_fmt_money(oi)}."
        )
    si = row.get("gains_strategic_investments")
    if pd.notna(si) and pd.notna(ni) and abs(si) >= 0.20 * abs(ni):
        bits.append(f"Paper gains on investments, not sales, were {_fmt_money(si)} last quarter.")
    q = row.get("quality_score")
    if pd.notna(q) and float(q) >= 50:
        bits.append(f"Business quality is {float(q):.0f} of 100.")
    if _flag(row.get("trifecta_pass")):
        bits.append("It clears the three cheapness tests.")
    if _flag(row.get("mos_pass")) and pd.notna(ev) and float(ev) > 0 and pd.notna(fair) and float(fair) > 0:
        bits.append("Price is at least 15% below a fair ratio of firm value (equity plus net debt) to operating profit.")
    ir = row.get("implied_r_clean_pct")
    hide_ir = expensive and profile.omit_implied_r_if_expensive
    if (not hide_ir) and pd.notna(ir) and float(ir) > 0:
        bits.append(f"Owners need {float(ir):.1f}% a year from this stock to make today's price fair — that is an annual hurdle, not a P/E ratio.")
    er = row.get("expected_return")
    if pd.notna(er):
        if float(er) >= 0.70:
            bits.append("Carry, value, price-trend, and defensive ingredients of expected return rank near the top of the market.")
        elif float(er) <= 0.30:
            bits.append("Carry, value, price-trend, and defensive ingredients of expected return rank near the bottom of the market.")
    mom = row.get("mom_12_1")
    hide_m = expensive and profile.omit_mom_if_expensive
    if (not avoid) and (not burn) and (not hide_m) and pd.notna(mom) and abs(float(mom)) >= 0.20:
        bits.append(
            f"Over the past year excluding last month the shares {_fmt_px('the shares', float(mom)).removeprefix('the shares ')}."
        )
    r63 = row.get("ret_63d")
    if (
        (not avoid) and (not burn) and (not hide_m)
        and r63 is not None and pd.notna(r63) and ticker_ret_21d is not None and pd.notna(ticker_ret_21d)
        and (float(r63) * float(ticker_ret_21d) < 0 or (abs(float(r63)) >= 0.15 and abs(float(r63)) >= 2 * abs(float(ticker_ret_21d))))
    ):
        bits.append(f"Over the last quarter {_fmt_px('the shares', float(r63))}.")

    nn = row.get("news_note")
    if isinstance(nn, str) and nn.strip():
        bits.append(profile.press_lead + nn.strip().rstrip(".") + ".")

    return " ".join(bits)


def _parse_forecast(text, brief=""):
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError("no JSON object")
    data = json.loads(json_match.group(0))
    direction = str(data["direction"]).strip().lower()
    if direction not in {"up", "sideways", "down"}:
        raise ValueError(f"bad direction {direction!r}")
    prob = float(data["prob"])
    if not (0.10 <= prob <= 0.90):
        raise ValueError(f"bad prob {prob}")
    horizon = int(data["horizon_days"])
    if not (1 <= horizon <= 252):
        raise ValueError(f"bad horizon {horizon}")
    rationale = str(data["rationale"]).strip()
    if not rationale:
        raise ValueError("empty rationale")
    return direction, prob, horizon, rationale


def _clamp_horizon(horizon):
    return min(max(int(horizon), 1), 252)


def _coverage_tickers(ctx: pd.DataFrame) -> list[str]:
    """Classified stage + 3y sales growth + FCF + ROIC−WACC on each ticker's last as_of."""
    last = ctx.sort_values("as_of_date").groupby("ticker", sort=False).tail(1)
    stage = last["life_cycle_stage"]
    mask = (
        stage.notna()
        & (stage.astype(str) != "Unclassified")
        & last["revenue_growth_3y"].notna()
        & last["fcf_margin"].notna()
        & last["roic_wacc_spread"].notna()
    )
    return last.loc[mask, "ticker"].astype(str).tolist()


def _llm_predict(ticker, brief, system: str):
    llm, grammar = _get_llm()
    # Reset every generation. Skipping it makes later profiles continue prior JSON (order-dependent).
    if hasattr(llm, "reset"):
        llm.reset()
    user_msg = brief
    last_err = None
    last_text = ""
    for attempt in range(6):
        extra = ""
        if attempt:
            extra = " Forecast; use one operating number as the reason."
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg + extra},
            ],
            max_tokens=120,
            temperature=min(0.3 + 0.1 * attempt, 0.7),
            top_p=0.9,
            grammar=grammar,
        )
        last_text = out["choices"][0]["message"]["content"]
        try:
            direction, prob, horizon, rationale = _parse_forecast(last_text, brief)
            return direction, prob, _clamp_horizon(horizon), rationale
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"LLM JSON parse failed: {last_err}; raw={last_text!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--tickers-file", type=str, default="")
    ap.add_argument("--model", type=str, default="3b", choices=["1b", "3b"])
    ap.add_argument(
        "--profiles",
        type=str,
        default="value,exuberant",
        help="Comma list of JobProfile names. Innermost loop per ticker. Default both.",
    )
    args = ap.parse_args()
    wanted = []
    for name in args.profiles.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in PROFILES:
            raise SystemExit(f"unknown profile {name!r}; known {sorted(PROFILES)}")
        if name not in wanted:
            wanted.append(name)
    profiles = [PROFILES[n] for n in wanted]
    if not profiles:
        raise SystemExit("no profiles")
    _get_llm(MODEL_3B if args.model == "3b" else MODEL_1B)

    st = load_states()
    recent = st.tail(max(int(args.lookback), 21)).copy()
    recent["mkt_ret_21d"] = (1.0 + recent["mkt_ret"]).rolling(21).apply(lambda x: float(x.prod() - 1.0), raw=True)
    last_rows = recent.dropna(subset=["mkt_ret_21d"])
    if last_rows.empty:
        print("No HMM row with 21-day market return")
        return
    r = last_rows.iloc[-1]
    fc_date = r["date"]
    if isinstance(fc_date, datetime):
        fc_date = fc_date.date()
    regime = r["regime"]
    vol21 = r["vol21"]
    avg_corr = r["avg_corr"]
    mkt_21 = float(r["mkt_ret_21d"])

    ctx = load_damodaran_context()
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        tickers = [t.strip() for t in Path(args.tickers_file).read_text(encoding="utf-8").splitlines() if t.strip()]
    else:
        tickers = _coverage_tickers(ctx)
        print(f"Coverage-gated tickers: {len(tickers)}")
    if not tickers:
        print("No tickers")
        return
    ctx = ctx[ctx["ticker"].isin(tickers)]

    px = None
    try:
        from analytics_common import load_adj_prices_pandas
        px = load_adj_prices_pandas(tickers=tickers)
        if len(px):
            x = px["date"].iloc[0]
            if isinstance(x, datetime):
                px["date"] = [v.date() if isinstance(v, datetime) else v for v in px["date"]]
    except Exception as e:
        print(f"price load failed ({e}); 21-day path omitted", flush=True)
        px = None

    d_out = fc_date
    done = set()
    rows = []
    if OUT.exists():
        try:
            prev = pd.read_parquet(OUT)
            if "ticker" in prev.columns and len(prev):
                prev_dates = prev["date"]
                same = True
                if "date" in prev.columns:
                    sample = prev_dates.iloc[0]
                    sample_d = sample.date() if hasattr(sample, "date") and callable(sample.date) else sample
                    same = sample_d == d_out
                if same:
                    if "profile" not in prev.columns:
                        prev["profile"] = "value"
                    rows = prev.to_dict("records")
                    done = {
                        (str(t), str(p))
                        for t, p in zip(prev["ticker"].astype(str), prev["profile"].astype(str))
                    }
                    print(f"Resume: {len(done)} already written for {d_out}", flush=True)
                else:
                    print(f"Existing parquet is a different date; not mixing. Snapshot kept at {OUT}", flush=True)
        except Exception as e:
            print(f"resume skipped ({e})", flush=True)

    jobs = [(t, p) for t in tickers for p in profiles]
    n = len(jobs)
    print(f"Jobs: {n} ({len(tickers)} tickers × {len(profiles)} profiles {wanted})", flush=True)
    ctx_by_ticker = {}
    tr_by_ticker = {}
    for i, (ticker, profile) in enumerate(jobs, 1):
        if (ticker, profile.name) in done:
            print(f"{i}/{n} {ticker} {profile.name} skip resume", flush=True)
            continue
        if ticker not in ctx_by_ticker:
            t_ctx = ctx[(ctx["ticker"] == ticker) & (ctx["as_of_date"] <= fc_date)]
            if t_ctx.empty:
                ctx_by_ticker[ticker] = None
            else:
                ctx_by_ticker[ticker] = t_ctx.sort_values("as_of_date").iloc[-1]
            tr = None
            if px is not None:
                g = px[(px["ticker"] == ticker) & (px["date"] <= fc_date)].sort_values("date")
                if len(g) >= 22:
                    c = g["adj_close"] if "adj_close" in g.columns else g.get("close")
                    a, b = c.iloc[-1], c.iloc[-22]
                    if pd.notna(a) and pd.notna(b) and float(b) != 0:
                        tr = float(a) / float(b) - 1
            tr_by_ticker[ticker] = tr
        row = ctx_by_ticker[ticker]
        if row is None:
            print(f"  skip {ticker}: no context", flush=True)
            continue
        tr = tr_by_ticker.get(ticker)
        brief = build_brief(ticker, mkt_21, row, tr, profile)
        if n <= 32:
            print(f"  SYSTEM {profile.name}: {profile.system}", flush=True)
            print(f"  brief {ticker} {profile.name}: {brief}", flush=True)
        wacc = row.get("wacc")
        fair_ev = row.get("fair_ev_ebitda")
        quality = row.get("quality_score")
        life_cycle = row.get("life_cycle_stage")
        try:
            direction, prob, horizon, narrative = _llm_predict(ticker, brief, profile.system)
        except RuntimeError as e:
            print(f"  skip {ticker} {profile.name}: {e}", flush=True)
            continue
        uncertainty = "high" if (vol21 > recent["vol21"].quantile(0.75) or regime == "high_vol_stress") else "normal"
        rows.append({
            "date": d_out,
            "ticker": ticker,
            "profile": profile.name,
            "regime": regime,
            "mkt_ret_21d": mkt_21,
            "vol21": float(vol21),
            "avg_corr": float(avg_corr),
            "forecast_dir": direction,
            "forecast_prob": prob,
            "horizon_days": horizon,
            "narrative": narrative,
            "uncertainty_flag": uncertainty,
            "life_cycle_stage": life_cycle,
            "wacc": float(wacc) if pd.notna(wacc) else None,
            "fair_ev_ebitda": float(fair_ev) if pd.notna(fair_ev) else None,
            "quality_score": float(quality) if pd.notna(quality) else None,
            "damodaran_narrative": brief,
        })
        done.add((ticker, profile.name))
        print(f"{i}/{n} {ticker} {profile.name} {direction} {prob:.2f} n={horizon}", flush=True)
        out = pd.DataFrame(rows)
        tbl = pa.Table.from_pandas(out, preserve_index=False)
        idx = tbl.schema.get_field_index("date")
        tbl = tbl.set_column(idx, "date", pa.array([d_out] * len(out), type=pa.date32()))
        pq.write_table(tbl, OUT)

    if rows:
        print(f"Wrote {OUT} ({len(rows)} rows)")
    else:
        print("No rows generated")

if __name__ == "__main__":
    main()
