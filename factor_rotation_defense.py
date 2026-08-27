#!/usr/bin/env python3
"""
factor_rotation_defense.py — Defensive factor rotation strategies.

Factors (long-only sleeves):

  Dynamic (computed per month-end, full universe):
    low_vol   — bottom tercile 63d vol over ALL price tickers
    momentum  — top tercile 12-1 momentum (skip last month) over ALL price
                tickers (Value-and-Momentum-Everywhere, Asness/Moskowitz/
                Pedersen 2013 — the diversification half of the value pair,
                ~-0.55 corr with value)

  Fundamental screens (PIT — membership = fundamentals as-of month-end):
    quality   — Buffett pass names (ROE/ROIC >= 15%, D/E <= 1)
    value     — trifecta pass names (EV/EBITDA <= 9, P/B <= 1.5, MCA <= 0.5)
    dual      — quality ∩ value

  Annotation groups — TWO-TABLE GROUP SYSTEM (the single source of named
  groups; groups grow by appending rows, no code changes):

    factor_groups.csv          — catalog: (group, group_type)
                                 group_type ∈ {sector, industry, index,
                                 sleeve, dynamic, custom}
    factor_group_members.csv   — join with as-of dates:
                                 (group, ticker, valid_from, valid_to)
                                 Membership is point-in-time: a ticker
                                 belongs on date d iff valid_from <= d and
                                 (valid_to is null or valid_to > d). This
                                 makes temporal memberships first-class —
                                 e.g. S&P 500 additions/removals populate
                                 valid windows from sp500_changes.parquet.

  The tables are auto-seeded from monitored_stocks annotations on first
  run (defensive_value_index, value_sleeve, growth_tech_index,
  dual_pass_member, sector, industry) + GICS for SP500 names + the S&P
  change history as temporal memberships.

Universe honesty (2026-08 audit):
  - quality/value/dual are rebuilt POINT-IN-TIME from fundamentals.parquet
    as-of each month-end (was: latest_fund() applied to all history — a
    look-ahead bias).
  - low_vol + momentum are computed over ALL price tickers.
  - named groups come from the two-table group system, not hardcoded lists;
    the classic ETF list is the dividend fallback only when no dividend
    group exists.
  - coverage: 142 monitored tickers carry full annotations, 503 SP500 carry
    GICS, 549 carry fundamentals, 551 carry prices.

Rotation signals (monthly, all trailing / point-in-time):
  - Risk-on: prior 21d market vol below median → overweight quality/dual
  - Risk-off: vol above 80th pct or crisis flag → overweight low_vol + dividend
  - Value tilt when value-quality spread (value minus quality 63d return) is
    depressed
  - Momentum overlay (VME 2013): when 12-1 momentum is in the top tercile,
    hold a momentum weight instead of folding its exposure into value.

Usage:
  python factor_rotation_defense.py --save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from threshold_logic import select_regime_from_hmm_file, thresholds_for_regime
except ImportError:
    select_regime_from_hmm_file = None
    thresholds_for_regime = None


DATA_DIR = Path(__file__).parent
PRICES = DATA_DIR / "daily_prices/"
STOCKS = DATA_DIR / "monitored_stocks.parquet"
FUND = DATA_DIR / "fundamentals.parquet"
SP500 = DATA_DIR / "sp500_constituents.parquet"
SP500_CHANGES = DATA_DIR / "sp500_changes.parquet"
GROUPS = DATA_DIR / "factor_groups.parquet"           # catalog: group, group_type
MEMBERS = DATA_DIR / "factor_group_members.parquet"   # join: group, ticker, valid_from, valid_to
OUT_W = DATA_DIR / "factor_rotation_weights.parquet"
OUT_PERF = DATA_DIR / "factor_rotation_performance.parquet"
OUT_SLEEVE = DATA_DIR / "factor_sleeve_returns.parquet"

MOM_SKIP = 21     # 12-1 momentum: skip the most recent month
MOM_WINDOW = 252  # trailing 12 months (trading days)

# the rotation weight schema (sleeves with an allocation)
WEIGHT_SCHEMA = ["quality", "value", "dual", "low_vol", "momentum",
                 "dividend", "defensive_idx"]

_CACHE = {}


def load_panels():
    """Load price/return panels + annotations ONCE. Returns (rets, vol21,
    mkt, cum, mom_score, vol63, ann)."""
    if "panels" in _CACHE:
        return _CACHE["panels"]
    prices = pd.read_parquet(PRICES, columns=["date", "ticker", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().ffill()
    rets = np.log(wide / wide.shift(1)).dropna(how="all")
    mkt = rets.mean(axis=1)
    vol21 = mkt.rolling(21).std() * np.sqrt(252)

    # full-length series, precomputed ONCE (per-month O(1) lookups below)
    cum = (1 + rets).cumprod()
    mom12 = (cum / cum.shift(MOM_WINDOW) - 1)
    mom1 = (cum / cum.shift(MOM_SKIP) - 1)
    mom_score = (mom12 - mom1)  # 12-1 momentum, skip last month
    vol63 = rets.rolling(63).std() * np.sqrt(252)

    # annotations: monitored_stocks flags + GICS for SP500 names
    stocks = pd.read_parquet(STOCKS)
    ann = {}
    for _, r in stocks.iterrows():
        ann[str(r["ticker"]).upper()] = {
            "sector": r.get("sector"),
            "industry": r.get("industry"),
            "subsector": r.get("subsector"),
            "defensive_value_index": bool(r.get("defensive_value_index")),
            "growth_tech_index": bool(r.get("growth_tech_index")),
            "value_sleeve": r.get("value_sleeve"),
            "dual_pass_member": bool(r.get("dual_pass_member")),
            "instrument_type": r.get("instrument_type"),
        }
    try:
        sp = pd.read_parquet(SP500)
        if "gics_sector" in sp.columns:
            for _, r in sp.iterrows():
                tk = str(r["ticker"]).upper()
                if tk not in ann:
                    ann[tk] = {}
                ann[tk]["gics_sector"] = r.get("gics_sector")
    except Exception:
        pass

    _CACHE["panels"] = (rets, vol21, mkt, cum, mom_score, vol63, ann)
    return _CACHE["panels"]


def ensure_groups(ann: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed the two-table group system if absent.

    factor_groups.csv:        (group, group_type) catalog
    factor_group_members.csv: (group, ticker, valid_from, valid_to) join —
                              temporal memberships supported; S&P 500
                              additions/removals populate valid windows.
    Returns (catalog, members).
    """
    if GROUPS.exists() and MEMBERS.exists():
        cat = pd.read_parquet(GROUPS)
        mem = pd.read_parquet(MEMBERS)
        for c in ("valid_from", "valid_to"):
            if c not in mem.columns:
                mem[c] = pd.NaT
        return cat, mem

    rows = []      # (group, ticker)
    types = {}     # group -> group_type
    for tk, a in ann.items():
        if a.get("defensive_value_index"):
            rows.append(("defensive_idx", tk)); types.setdefault("defensive_idx", "index")
        if a.get("value_sleeve") == "defensive_etf":
            rows.append(("dividend", tk)); types.setdefault("dividend", "sleeve")
        if a.get("value_sleeve") == "financial_value":
            rows.append(("financial_value", tk)); types.setdefault("financial_value", "sleeve")
        if a.get("growth_tech_index"):
            rows.append(("growth_tech", tk)); types.setdefault("growth_tech", "index")
        if a.get("dual_pass_member"):
            rows.append(("dual_member", tk)); types.setdefault("dual_member", "index")
        for key, prefix, gtype in (("sector", "sector", "sector"), ("industry", "industry", "industry")):
            v = a.get(key)
            if v:
                rows.append((f"{prefix}_{v}", tk)); types.setdefault(f"{prefix}_{v}", gtype)
        gics = a.get("gics_sector")
        if gics:
            rows.append((f"sector_{gics}", tk)); types.setdefault(f"sector_{gics}", "sector")

    mem = pd.DataFrame(rows, columns=["group", "ticker"]).drop_duplicates()
    mem["valid_from"] = pd.NaT
    mem["valid_to"] = pd.NaT

    # temporal S&P memberships from the change history: contiguous windows
    # per ticker from the sorted event timeline. A ticker is a member on
    # date d iff the LATEST event at or before d is an ADD.
    try:
        sc = pd.read_parquet(SP500_CHANGES)
        if {"event_date", "added", "removed"}.issubset(sc.columns):
            types.setdefault("sp500", "index")
            events = []
            for _, r in sc.iterrows():
                d = pd.Timestamp(r["event_date"])
                if pd.notna(r.get("added")):
                    events.append((d, "add", str(r["added"]).upper()))
                if pd.notna(r.get("removed")):
                    events.append((d, "remove", str(r["removed"]).upper()))
            events.sort(key=lambda e: (e[0], e[1] == "remove"))  # removes after adds same day
            # current SP500 roster closes the windows at the end (constituents
            # table has date_added; those still current are open-ended)
            try:
                sp_cur = pd.read_parquet(SP500)
                cur = set(str(t).upper() for t in sp_cur["ticker"]) if "ticker" in sp_cur.columns else set()
            except Exception:
                cur = set()
            # walk the timeline per ticker -> (valid_from, valid_to) windows
            per_ticker: dict[str, list] = {}
            has_any_add: set[str] = set()
            for d, act, tk in events:
                per_ticker.setdefault(tk, []).append((d, act))
                if act == "add":
                    has_any_add.add(tk)
            sp_mem = []
            for tk, evs in per_ticker.items():
                evs.sort()
                opened = None
                for d, act in evs:
                    if act == "add" and opened is None:
                        opened = d
                    elif act == "remove" and opened is not None:
                        sp_mem.append((tk, opened, d))
                        opened = None
                    elif act == "remove" and opened is None and tk not in has_any_add:
                        # REMOVE WITH NO PRECEDING ADD AND NO ADD ANYWHERE:
                        # the name was an original (pre-log) member whose add
                        # was never recorded — the remove closes a window
                        # that opened before the log began. Without this,
                        # original 1957 members that were later removed
                        # vanish from every historical view (the 2000 as-of
                        # showed 437/500). Only the EARLIEST such remove
                        # opens the pre-log window — a second consecutive
                        # remove (unlogged re-add+remove cycle) must not
                        # claim another from-inception membership.
                        if not any(v[1] == "remove" for v in evs[:evs.index((d, act))]):
                            sp_mem.append((tk, pd.NaT, d))
                if opened is not None:
                    # still a member at the end of the event history
                    sp_mem.append((tk, opened, pd.NaT))
            # current roster not in the event history: open-ended membership
            for tk in cur:
                if tk not in per_ticker or all(v[1] == "remove" for v in per_ticker[tk]):
                    # added before the event history begins; treat as open
                    if tk not in [r[0] for r in sp_mem]:
                        sp_mem.append((tk, pd.NaT, pd.NaT))
            for tk, vf, vt in sp_mem:
                mem = pd.concat([mem, pd.DataFrame([{
                    "group": "sp500", "ticker": tk, "valid_from": vf, "valid_to": vt}])], ignore_index=True)
    except Exception as e:
        print(f"  sp500 changes seed skipped ({e})")

    cat = pd.DataFrame([{"group": g, "group_type": t} for g, t in types.items()])
    # every distinct group in members gets a catalog row
    for g in mem["group"].unique():
        if g not in set(cat["group"]):
            cat = pd.concat([cat, pd.DataFrame([{"group": g, "group_type": "custom"}])], ignore_index=True)

    cat.to_parquet(GROUPS, index=False)
    mem.to_parquet(MEMBERS, index=False)
    print(f"Seeded {GROUPS.name} ({len(cat)} groups) + {MEMBERS.name} ({len(mem)} memberships)")
    return cat, mem


def _members_typed(mem: pd.DataFrame) -> pd.DataFrame:
    """One-time dtype normalization for the members table (cached)."""
    if "members_typed" not in _CACHE:
        m = mem.copy()
        m["valid_from"] = pd.to_datetime(m.get("valid_from"), errors="coerce")
        m["valid_to"] = pd.to_datetime(m.get("valid_to"), errors="coerce")
        _CACHE["members_typed"] = m
    return _CACHE["members_typed"]


def members_asof(mem: pd.DataFrame, d) -> dict[str, list[str]]:
    """Point-in-time group memberships: ticker in group on date d iff
    valid_from <= d and (valid_to null or valid_to > d)."""
    m = _members_typed(mem)
    d = pd.Timestamp(d)
    m = m[m["valid_from"].isna() | (m["valid_from"] <= d)]
    m = m[m["valid_to"].isna() | (m["valid_to"] > d)]
    return {g: list(gdf["ticker"]) for g, gdf in m.groupby("group")}


def _funds_pt():
    """All fundamentals with as_of_date, cached (PIT queries slice this)."""
    if "funds" not in _CACHE:
        f = pd.read_parquet(FUND)
        f["as_of_date"] = pd.to_datetime(f["as_of_date"])
        f = f.sort_values("as_of_date")
        _CACHE["funds"] = f
    return _CACHE["funds"]


def pit_fund(as_of) -> pd.DataFrame:
    """Point-in-time fundamentals: the latest row per ticker with
    as_of_date <= as_of. Returns index=ticker."""
    f = _funds_pt()
    f = f[f["as_of_date"] <= pd.Timestamp(as_of)]
    if f.empty:
        return pd.DataFrame()
    return f.groupby("ticker", as_index=False).tail(1).set_index("ticker")


def build_sleeves(mom_row, vol_row, fund: pd.DataFrame, groups_asof: dict[str, list[str]],
                  cols) -> dict[str, list[str]]:
    """Sleeve membership at ONE point in time, from precomputed rows.

    mom_row / vol_row: the cross-section of the momentum score and 63d vol
    at this month-end (O(1) lookups — no per-month full-panel rebuild).
    fund = PIT fundamentals; groups_asof = PIT group memberships.
    """
    low_vol = []
    v = vol_row.dropna()
    if len(v):
        low_vol = v.nsmallest(max(5, len(v) // 3)).index.tolist()

    momentum = []
    ms = mom_row.dropna()
    if len(ms):
        momentum = ms.nlargest(max(5, len(ms) // 3)).index.tolist()

    buffett, trifecta = [], []
    if not fund.empty:
        ok = fund[(fund.roe >= 0.15) & (fund.roic >= 0.15) & (fund.debt_to_equity <= 1.0)]
        buffett = ok.index.tolist()
        tf = fund[(fund.ev_ebitda <= 9) & (fund.pb_ratio <= 1.5) & (fund.mktcap_to_assets <= 0.5)]
        trifecta = tf.index.tolist()
    dual = list(set(buffett) & set(trifecta))

    sleeves: dict[str, list[str]] = {
        "quality": buffett,
        "value": trifecta,
        "dual": dual,
        "low_vol": low_vol,
        "momentum": momentum,
    }
    for g, members in groups_asof.items():
        members = [t for t in members if t in cols]
        if g in sleeves:
            sleeves[g] = list(dict.fromkeys(sleeves[g] + members))
        else:
            sleeves[g] = members

    if not sleeves.get("dividend"):
        sleeves["dividend"] = [t for t in ["SCHD", "VIG", "XLP", "XLU", "USMV", "SPLV", "VYM"] if t in cols]
    if not sleeves.get("defensive_idx"):
        sleeves["defensive_idx"] = []
    return sleeves


def sleeve_return(rets: pd.DataFrame, members: list[str]) -> pd.Series:
    cols = [c for c in members if c in rets.columns]
    if not cols:
        return pd.Series(0.0, index=rets.index)
    return rets[cols].mean(axis=1)


def run(save: bool = True):
    rets, vol21, mkt, cum, mom_score, vol63, ann = load_panels()
    cat, mem = ensure_groups(ann)
    months = rets.index.to_period("M").unique()
    cols = set(rets.columns)

    weights_rows = []
    port = []
    sleeve_hist = {}

    for m in months:
        end = (m.start_time - pd.Timedelta(days=1))
        hist = vol21.loc[:end].dropna()

        # O(1) cross-section lookups + PIT funds + PIT groups
        fund = pit_fund(end)
        if end in mom_score.index:
            mom_row = mom_score.loc[end]
            vol_row = vol63.loc[end]
        else:
            prior = mom_score.index[mom_score.index <= end]
            mom_row = mom_score.loc[prior[-1]] if len(prior) else mom_score.iloc[0]
            vol_row = vol63.loc[prior[-1]] if len(prior) else vol63.iloc[0]
        groups_asof = members_asof(mem, end)
        sleeves = build_sleeves(mom_row, vol_row, fund, groups_asof, cols)

        days = rets.index[rets.index.to_period("M") == m]
        sret_m = pd.DataFrame({k: sleeve_return(rets.loc[days], v) for k, v in sleeves.items()})
        sleeve_hist[m] = sret_m

        regime = "normal"
        if select_regime_from_hmm_file is not None:
            try:
                regime = select_regime_from_hmm_file(soft_min=0.7)
            except Exception:
                regime = "normal"
        risk_off = False
        risk_on = False

        if len(hist) < 22:
            w = {"quality": 0.15, "value": 0.15, "low_vol": 0.15, "dividend": 0.15, "dual": 0.10, "defensive_idx": 0.10, "momentum": 0.20}
        else:
            v = float(hist.iloc[-1])
            v_med = float(hist.iloc[-126:].median()) if len(hist) >= 60 else float(hist.median())
            v_p80 = float(hist.iloc[-126:].quantile(0.8)) if len(hist) >= 60 else float(hist.quantile(0.8))
            risk_off = v >= v_p80 or regime == "high_vol_stress"
            risk_on = (v <= v_med and regime == "low_vol") or (regime == "low_vol")
            if regime == "uncertain":
                risk_off = False
                risk_on = False

            rtrail = rets.loc[:end]
            q_series = sleeve_return(rtrail, sleeves["quality"])
            v_series = sleeve_return(rtrail, sleeves["value"])
            q = q_series.iloc[-63:].sum() if len(q_series) >= 63 else 0
            val = v_series.iloc[-63:].sum() if len(v_series) >= 63 else 0
            value_cheap = val < q

            mom_strong = False
            ms = mom_row.dropna()
            if len(ms):
                thr = ms.quantile(2 / 3)
                mom_strong = float(ms.median()) > float(thr)

            if risk_off:
                w = {"quality": 0.10, "value": 0.08, "dual": 0.05, "low_vol": 0.28, "momentum": 0.06, "dividend": 0.28, "defensive_idx": 0.15}
            elif risk_on and not value_cheap:
                w = {"quality": 0.22, "value": 0.12, "dual": 0.18, "low_vol": 0.12, "momentum": 0.16, "dividend": 0.08, "defensive_idx": 0.12}
            elif value_cheap and mom_strong:
                # VME core case: value cheap + momentum strong — hold BOTH
                w = {"quality": 0.12, "value": 0.22, "dual": 0.12, "low_vol": 0.12, "momentum": 0.22, "dividend": 0.10, "defensive_idx": 0.10}
            elif value_cheap:
                w = {"quality": 0.14, "value": 0.28, "dual": 0.14, "low_vol": 0.14, "momentum": 0.06, "dividend": 0.14, "defensive_idx": 0.10}
            else:
                w = {"quality": 0.18, "value": 0.18, "dual": 0.10, "low_vol": 0.16, "momentum": 0.14, "dividend": 0.12, "defensive_idx": 0.12}

        for d in days:
            weights_rows.append({"date": d, **w,
                                 "regime": "risk_off" if risk_off else ("risk_on" if risk_on else "neutral"),
                                 "hmm_regime": regime})
            r = sum(w.get(k, 0) * float(sret_m.loc[d, k]) if d in sret_m.index and k in sret_m.columns else 0 for k in w)
            port.append({"date": d, "ret": r})

    wdf = pd.DataFrame(weights_rows)
    pdf = pd.DataFrame(port).set_index("date")["ret"]

    # static benchmark: FULL-HISTORY defensive_idx series (PIT, joined across
    # months) — not just the last month.
    full_sret = pd.concat(sleeve_hist.values())
    full_sret = full_sret[~full_sret.index.duplicated(keep="last")].sort_index()
    static = full_sret.get("defensive_idx")
    if static is None or static.dropna().empty:
        static = full_sret.mean(axis=1)

    def stats(r):
        r = r.dropna()
        if len(r) < 5:
            return {}
        return {
            "ann_ret": float(r.mean() * 252),
            "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": float(r.mean() * 252 / (r.std() * np.sqrt(252))) if r.std() > 0 else np.nan,
            "max_dd": float((np.exp(r.cumsum()) / np.exp(r.cumsum()).cummax() - 1).min()),
        }

    perf = []
    for name in full_sret.columns:
        s = stats(full_sret[name])
        s["strategy"] = name
        perf.append(s)
    for name, series in [("rotation", pdf), ("static_defensive", static)]:
        s = stats(series)
        s["strategy"] = name
        perf.append(s)
    perf_df = pd.DataFrame(perf)
    print("\n=== Factor / rotation performance (PIT membership) ===")
    print(perf_df.to_string(index=False))

    from cv_utils import oos_stats_vs_baseline
    if len(pdf) > 504 and len(static) > 504:
        oos = oos_stats_vs_baseline(pdf.tail(504), static.tail(504))
        oos["strategy"] = "rotation_vs_static_OOS_2y"
        perf_df = pd.concat([perf_df, pd.DataFrame([oos])], ignore_index=True)
        print("\n=== OOS 2y: rotation vs static defensive ===")
        for k, v in oos.items():
            if k != "strategy":
                print(f"  {k}: {v}")

    if save:
        wdf.to_parquet(OUT_W, index=False)
        perf_df.to_parquet(OUT_PERF, index=False)
        full_sret.reset_index().rename(columns={"index": "date"}).to_parquet(OUT_SLEEVE, index=False)
        print(f"Wrote {OUT_W}\nWrote {OUT_PERF}\nWrote {OUT_SLEEVE}")
    return perf_df


def load_group_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two group tables (seeding if absent via annotations)."""
    rets, vol21, mkt, cum, mom_score, vol63, ann = load_panels()
    return ensure_groups(ann)


def manage_group_members(args):
    """Edit the group tables: add/evict/date memberships, create typed
    groups. The temporal schema (valid_from/valid_to) is universal — every
    group supports removal dates; a membership without dates is simply
    always-valid (evict by deleting the row or setting valid_to).

    Commands:
      add-group    GROUP [--type TYPE]          create a typed group
      add          GROUP TICKER [--from D] [--to D]   add a membership
      evict        GROUP TICKER [--on D]        evict (delete row, or set
                                                valid_to=D if --on)
      show         [--group G] [--as-of D]      list memberships (PIT when
                                                --as-of given)
      timeline     TICKER                       all windows for a ticker
    """
    cat, mem = load_group_tables()
    if not GROUPS.exists():
        cat.to_parquet(GROUPS, index=False)
    if not MEMBERS.exists():
        mem.to_parquet(MEMBERS, index=False)

    g = getattr(args, "group", None)
    g = g.upper() if g else None
    tk = getattr(args, "ticker", None)
    tk = tk.upper() if tk else None

    if args.cmd == "add-group":
        if g in set(cat["group"]):
            print(f"group {g} already exists (type {cat.loc[cat['group']==g,'group_type'].iloc[0]})")
            return
        cat = pd.concat([cat, pd.DataFrame([{"group": g, "group_type": args.type or "custom"}])], ignore_index=True)
        cat.to_parquet(GROUPS, index=False)
        print(f"group {g} (type={args.type or 'custom'}) created")
    elif args.cmd == "add":
        if g not in set(cat["group"].str.upper()):
            print(f"group {g} not in catalog — run add-group first")
            return
        row = {"group": g, "ticker": tk, "valid_from": args.frm, "valid_to": args.to}
        mem = pd.concat([mem, pd.DataFrame([row])], ignore_index=True)
        mem.to_parquet(MEMBERS, index=False)
        print(f"{tk} added to {g} [{args.frm or 'open'} -> {args.to or 'open'}]")
    elif args.cmd == "evict":
        m = mem[(mem["group"].str.upper() == g) & (mem["ticker"].str.upper() == tk)]
        if m.empty:
            print(f"no membership {tk} in {g}")
            return
        if args.on:
            # set valid_to on the open-ended window(s); rows already closed stay
            idx = m[m["valid_to"].isna() | (m["valid_to"] == "")].index
            mem.loc[idx, "valid_to"] = args.on
            mem.to_parquet(MEMBERS, index=False)
            print(f"{tk} evicted from {g} effective {args.on}")
        else:
            mem = mem.drop(m.index)
            mem.to_parquet(MEMBERS, index=False)
            print(f"{tk} evicted from {g} (rows removed)")
    elif args.cmd == "show":
        m = mem
        if g:
            m = m[m["group"].str.upper() == g]
        if args.asof:
            m = m.copy()
            m["valid_from"] = pd.to_datetime(m.get("valid_from"), errors="coerce")
            m["valid_to"] = pd.to_datetime(m.get("valid_to"), errors="coerce")
            d = pd.Timestamp(args.asof)
            m = m[m["valid_from"].isna() | (m["valid_from"] <= d)]
            m = m[m["valid_to"].isna() | (m["valid_to"] > d)]
            print(f"members of {g or 'all groups'} as-of {args.asof}: {len(m)}")
        print(m.to_string(index=False) if len(m) else "(none)")
    elif args.cmd == "timeline":
        m = mem[mem["ticker"] == tk]
        if m.empty:
            print(f"no memberships for {tk}")
            return
        for _, r in m.iterrows():
            print(f"{tk} in {r['group']}: [{r['valid_from'] or 'open'} -> {r['valid_to'] or 'open'}]")


def main():
    ap = argparse.ArgumentParser(description="Defensive factor rotation + group-table editor")
    sub = ap.add_subparsers(dest="cmd")

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--save", action="store_true")
    ap_run.set_defaults(func=lambda a: run(save=a.save))

    mg = sub.add_parser("add-group")
    mg.add_argument("--group", required=True)
    mg.add_argument("--type", default="custom", choices=["sector", "industry", "index", "sleeve", "dynamic", "custom"])
    mg.set_defaults(func=manage_group_members)

    ma = sub.add_parser("add")
    ma.add_argument("--group", required=True)
    ma.add_argument("--ticker", required=True)
    ma.add_argument("--from", dest="frm", default=None)
    ma.add_argument("--to", dest="to", default=None)
    ma.set_defaults(func=manage_group_members)

    me = sub.add_parser("evict")
    me.add_argument("--group", required=True)
    me.add_argument("--ticker", required=True)
    me.add_argument("--on", default=None)
    me.set_defaults(func=manage_group_members)

    ms = sub.add_parser("show")
    ms.add_argument("--group", default=None)
    ms.add_argument("--as-of", dest="asof", default=None)
    ms.set_defaults(func=manage_group_members)

    mt = sub.add_parser("timeline")
    mt.add_argument("--ticker", required=True)
    mt.set_defaults(func=manage_group_members)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
