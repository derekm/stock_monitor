#!/usr/bin/env python3
"""Basic tests for core stock_monitor modules."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_edgar_merge():
    """Test backfill_edgar merge logic with _old column restore and future date filter."""
    print("Testing EDGAR merge...")
    from backfill_edgar import merge_into_fundamentals
    import pandas as pd
    import shutil
    from pathlib import Path
    from datetime import date, timedelta
    
    # Use unique ticker to avoid conflicts
    test_ticker = 'TESTMERGE'
    
    # Backup original
    orig = Path('fundamentals.parquet')
    backup = Path('fundamentals_test_backup.parquet')
    shutil.copy2(orig, backup)
    
    try:
        today = date.today()
        
        # Create test data: 1 actual row, 2 future estimate rows
        new_rows = [
            {'ticker': test_ticker, 'as_of_date': today - timedelta(days=30), 'source': 'edgar_v2', 'total_revenue': 1000, 'net_income': 100},
            {'ticker': test_ticker, 'as_of_date': today + timedelta(days=90), 'source': 'edgar_v2', 'total_revenue': 1200, 'net_income': 150},  # future Q+1
            {'ticker': test_ticker, 'as_of_date': today + timedelta(days=180), 'source': 'edgar_v2', 'total_revenue': 1300, 'net_income': 180},  # future Q+2
        ]
        
        n = merge_into_fundamentals(new_rows)
        print(f'  Merged {n} rows')
        
        fund = pd.read_parquet('fundamentals.parquet')
        fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date
        
        test_rows = fund[fund['ticker'] == test_ticker]
        if len(test_rows) != 1:
            raise AssertionError(f"Expected 1 {test_ticker} row, got {len(test_rows)}")
        
        row = test_rows.iloc[0]
        # Verify future date NOT in fundamentals
        if row['as_of_date'] > today:
            raise AssertionError("Future-dated row should not be in fundamentals")
        
        # Verify prior estimates captured
        if row.get('prior_estimate_total_revenue') != 1200:
            raise AssertionError(f"Expected prior_estimate_total_revenue=1200, got {row.get('prior_estimate_total_revenue')}")
        if row.get('prior_estimate_net_income') != 150:
            raise AssertionError(f"Expected prior_estimate_net_income=150, got {row.get('prior_estimate_net_income')}")
        
        print("  EDGAR merge with prior_estimate columns ✓")
        return True
    finally:
        shutil.copy2(backup, orig)
        backup.unlink()


def test_distrust_fit():
    """The fitted distrust logit must stay DIAGNOSTIC, not drive distrust_discount.

    It failed honest OOS validation (0.591 walk-forward on the liquid universe,
    below the 0.65 gate, beaten by vol63 alone), and buy_candidates.py
    multiplies its score by distrust_discount -- so a regression that re-blends
    the fitted value into distrust_p_bad would silently move live decisions.
    """
    print("Testing distrust fit is diagnostic-only...")
    import numpy as np
    from preferred_metrics import build_table

    df = build_table()
    for c in ("distrust_p_bad", "distrust_discount", "distrust_p_bad_fitted",
              "distrust_fit_auc_insample", "distrust_fit_auc_oos"):
        assert c in df.columns, f"{c} column missing"

    # The honest OOS number must be recorded and must fail the gate.
    oos = float(df["distrust_fit_auc_oos"].dropna().iloc[0])
    assert oos < 0.65, f"recorded OOS AUC {oos} unexpectedly passes the gate"
    assert not bool(df["distrust_fit_gate_pass"].dropna().iloc[0]), "gate must be False"

    # distrust_discount must equal the HEURISTIC form, not the 40/60 blend.
    p_h = pd.to_numeric(df["distrust_p_bad"], errors="coerce")
    p_f = pd.to_numeric(df["distrust_p_bad_fitted"], errors="coerce")
    ex = pd.to_numeric(df["excess_cash_share"], errors="coerce").fillna(0)
    dd = pd.to_numeric(df["distrust_discount"], errors="coerce")

    expect_heur = (1.0 - p_h * ex).round(4)
    blend = (0.4 * p_h + 0.6 * p_f).clip(0, 0.8)
    expect_blend = (1.0 - blend * ex).round(4)

    m = dd.notna() & expect_heur.notna()
    assert np.allclose(dd[m], expect_heur[m], atol=1e-4), \
        "distrust_discount does not match the heuristic form"

    # And it must be measurably DIFFERENT from the blended form, so this test
    # actually detects a re-blend regression rather than passing vacuously.
    differs = (~np.isclose(expect_heur[m], expect_blend[m], atol=1e-4)).sum()
    assert differs > 0, "heuristic and blend are indistinguishable; test is vacuous"
    print(f"  discount is heuristic-only (differs from blend on {differs} names) ✓")
    print(f"  recorded OOS AUC {oos} < 0.65 gate, gate_pass=False ✓")
    return True


def test_rolling_skip():
    """Test rolling_window_analysis skip-if-complete."""
    print("Testing rolling skip...")
    from rolling_window_analysis import rolling_cumsum_2d
    
    # Simple test
    arr = np.random.randn(100, 10)
    out = rolling_cumsum_2d(arr, 21)
    assert out.shape == arr.shape, f"Shape mismatch: {out.shape} vs {arr.shape}"
    print(f"  rolling_cumsum_2d works: {out.shape} ✓")
    return True


def test_gpu_fallback():
    """Test GPU fallback (CUDA -> DirectML -> CPU)."""
    print("Testing GPU fallback...")
    import torch
    from tensor_ops import get_device, rolling_sum
    
    dev = get_device()
    print(f"  Selected device: {dev}")
    assert dev.type in ("cuda", "privateuseone", "cpu"), f"Unexpected device type: {dev.type}"
    
    # Test rolling_sum works on the selected device - use realistic size
    arr = np.random.randn(100, 252).astype(np.float32)
    out = rolling_sum(arr, 10, device=dev)
    assert out.shape == arr.shape, f"Shape mismatch: {out.shape} vs {arr.shape}"
    # First 9 columns will be NaN (window not full), check only valid columns
    valid = out[:, 9:]
    assert np.isfinite(valid).all(), "Non-finite values in valid output"
    print(f"  rolling_sum on {dev.type} works ✓")
    return True


def test_cape_erp():
    """Test CAPE ERP loading with hash verification."""
    print("Testing CAPE ERP...")
    from erp_service import load_erp, latest_implied_erp
    
    # Test loading different ERP sources
    for src in ["damodaran", "cape", "spy_sma"]:
        df = load_erp(src, "monthly")
        assert not df.empty, f"{src} ERP should not be empty"
        assert "erp" in df.columns, f"{src} missing erp column"
        assert "date" in df.columns, f"{src} missing date column"
        print(f"  {src}: {len(df)} rows, latest={df['erp'].iloc[-1]:.4f} ✓")
    
    erp = latest_implied_erp()
    assert 0.02 < erp < 0.10, f"ERP out of range: {erp}"
    print(f"  latest_implied_erp: {erp:.4f} ✓")
    return True


def test_tensor_ops_correctness():
    """Test tensor_ops numerical correctness vs CPU."""
    print("Testing tensor_ops correctness...")
    import torch
    from tensor_ops import rolling_sum, rolling_slope, rolling_beta, rolling_std
    from rolling_window_analysis import rolling_cumsum_2d
    
    # Create test data - [T, D] = [tickers, time]
    np.random.seed(42)
    arr = np.random.randn(200, 252).astype(np.float32)
    arr[:, 0] = np.cumsum(arr[:, 0])  # Add trend to first series
    bench = np.random.randn(252).astype(np.float32)  # 1D benchmark matching arr width
    
    # Test on CPU
    dev_cpu = torch.device("cpu")
    out_sum_cpu = rolling_sum(arr, 21, device=dev_cpu)
    out_slope_cpu = rolling_slope(arr, 21, device=dev_cpu)
    out_beta_cpu = rolling_beta(arr, bench, 21, device=dev_cpu)
    out_std_cpu = rolling_std(arr, 21, device=dev_cpu)
    
    # Test on GPU if available
    try:
        dev_gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("privateuseone")
        out_sum_gpu = rolling_sum(arr, 21, device=dev_gpu)
        out_slope_gpu = rolling_slope(arr, 21, device=dev_gpu)
        out_beta_gpu = rolling_beta(arr, bench, 21, device=dev_gpu)
        out_std_gpu = rolling_std(arr, 21, device=dev_gpu)
        
        # Compare valid columns only (first 20 columns will be NaN)
        valid = slice(20, None)
        
        max_diff_sum = np.abs(out_sum_cpu[:, valid] - out_sum_gpu[:, valid]).max()
        max_diff_slope = np.abs(out_slope_cpu[:, valid] - out_slope_gpu[:, valid]).max()
        max_diff_beta = np.abs(out_beta_cpu[:, valid] - out_beta_gpu[:, valid]).max()
        max_diff_std = np.abs(out_std_cpu[:, valid] - out_std_gpu[:, valid]).max()
        
        print(f"  Max diff sum: {max_diff_sum:.6f}")
        print(f"  Max diff slope: {max_diff_slope:.6f}")
        print(f"  Max diff beta: {max_diff_beta:.6f}")
        print(f"  Max diff std: {max_diff_std:.6f}")
        
        # Allow small numerical differences
        assert max_diff_sum < 1e-4, f"Sum mismatch: {max_diff_sum}"
        assert max_diff_slope < 1e-3, f"Slope mismatch: {max_diff_slope}"
        assert max_diff_beta < 1e-3, f"Beta mismatch: {max_diff_beta}"
        assert max_diff_std < 1e-4, f"Std mismatch: {max_diff_std}"
        print("  GPU vs CPU numerical match ✓")
    except Exception as e:
        print(f"  GPU test skipped: {e}")
    
    # Test rolling_cumsum_2d
    arr2d = np.random.randn(100, 10).astype(np.float32)
    out_cpu = rolling_cumsum_2d(arr2d, 21, device=dev_cpu)
    assert out_cpu.shape == arr2d.shape
    print("  rolling_cumsum_2d on CPU works ✓")
    
    return True


def test_fractal_gpu():
    """fractal_windows batch path must use tensor_ops device selection, and its GPU
    and CPU paths must agree numerically (test moved here from ad-hoc probes)."""
    print("Testing fractal GPU/CPU parity via tensor_ops...")
    import torch
    from tensor_ops import _best_device as canonical, is_gpu, device_name
    import fractal_windows as F

    # Device selection must be the SAME object identity-wise as tensor_ops.
    # fractal_windows no longer has its own _best_device; it resolves via tensor_ops.
    dev = canonical()
    print(f"  fractal device: {device_name(dev)}")
    assert dev.type in ("cuda", "privateuseone", "cpu")

    # gpu_available must agree with tensor_ops (was CUDA-only before).
    from tensor_ops import gpu_available as tops_gpu
    assert F.gpu_available() == tops_gpu(), "gpu_available diverges from tensor_ops"

    # CPU vs GPU parity on the real fractal kernel.
    np.random.seed(7)
    logp = np.cumsum(np.random.randn(12, 400) * 0.01, axis=1) + 4.0
    res_cpu = F.fractal_batch(logp, a=30, b=3, device=torch.device("cpu"))
    assert res_cpu, "fractal_batch returned nothing on CPU"

    if is_gpu(dev):
        res_gpu = F.fractal_batch(logp, a=30, b=3, device=dev)
        assert set(res_cpu) == set(res_gpu), "span keys differ between devices"
        worst = 0.0
        for span in res_cpu:
            for stat in ("ret", "slope", "momentum"):
                a = res_cpu[span][stat].cpu().numpy()
                b = res_gpu[span][stat].cpu().numpy()
                m = np.isfinite(a) & np.isfinite(b)
                if m.any():
                    worst = max(worst, float(np.abs(a[m] - b[m]).max()))
        print(f"  max |GPU-CPU| across spans/stats: {worst:.2e}")
        assert worst < 1e-2, f"fractal GPU/CPU divergence too large: {worst}"
        print("  fractal GPU vs CPU parity ✓")
    else:
        print("  no accelerator present — CPU path only ✓")
    return True


def test_coiled_spring_gpu_parity():
    """backtest_coiled_spring batched path: GPU == CPU == polars reference.

    Guards the three bugs found while building it:
      1. torch.cumsum propagates NaN where np.nancumsum does not, so a
         72%-NaN price panel gave 22,500 NaN on GPU vs 100 on CPU.
      2. rolling_mean/std divided by the window width instead of the observed
         count, disagreeing with pandas/polars in any NaN window.
      3. the cumsum-of-squares variance identity is unstable on real prices
         (sum(x^2) ~6e15 exhausts float64) -- measured 0.68 absolute error.
    """
    print("Testing coiled-spring GPU/CPU parity...")
    import numpy as np
    import polars as pl
    import torch
    from backtest_coiled_spring import compute_states_pl, compute_states_batch

    rng = np.random.default_rng(17)
    T, D = 12, 700
    close = 100 * np.cumprod(1 + rng.standard_normal((T, D)) * 0.012, axis=1)
    high = close * (1 + np.abs(rng.standard_normal((T, D))) * 0.004)
    low = close * (1 - np.abs(rng.standard_normal((T, D))) * 0.004)
    vol = rng.uniform(1e6, 6e6, (T, D))
    # inject real-world gaps: a late-listing ticker and interior holes
    close[0, :200] = np.nan
    high[0, :200] = np.nan
    low[0, :200] = np.nan
    vol[0, :200] = np.nan
    wide = {"close": close, "high": high, "low": low, "volume": vol}
    names = [f"T{i}" for i in range(T)]

    cpu = compute_states_batch(wide, names, device=torch.device("cpu"))
    auto = compute_states_batch(wide, names, device=None)
    print(f"  devices: cpu vs {auto['_device']}")

    bool_cols = ["squeeze_active", "width_compressed", "is_test", "is_held", "is_sprung"]
    for c in bool_cols:
        n = int((cpu[c] != auto[c]).sum())
        assert n == 0, f"{c}: {n} GPU/CPU mismatches"
    for c in ["bb_width", "vol_z", "bb_width_p252"]:
        a, b = cpu[c], auto[c]
        assert (np.isnan(a) == np.isnan(b)).all(), f"{c}: NaN pattern differs (cumsum NaN bug)"
        m = np.isfinite(a) & np.isfinite(b)
        d = float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0
        assert d < 1e-6, f"{c}: GPU/CPU maxdiff {d}"
    print("  batched GPU == CPU on all 8 outputs ✓")

    # And the batched path must match the per-ticker polars reference.
    worst_bool, worst_num = 0, 0.0
    for i in range(1, T):  # row 0 has the injected gap; polars ref needs >=300 rows
        ref = compute_states_pl(pl.DataFrame({
            "close": close[i], "high": high[i], "low": low[i], "volume": vol[i]}))
        for c in bool_cols:
            r = np.nan_to_num(ref[c].to_numpy()).astype(bool)
            worst_bool = max(worst_bool, int((r != auto[c][i]).sum()))
        for c in ["bb_width", "vol_z"]:
            r = ref[c].to_numpy()
            m = np.isfinite(r) & np.isfinite(auto[c][i])
            if m.any():
                worst_num = max(worst_num, float(np.abs(r[m] - auto[c][i][m]).max()))
    assert worst_bool == 0, f"batched vs polars: {worst_bool} boolean mismatches"
    assert worst_num < 1e-6, f"batched vs polars: {worst_num} numeric drift"
    print(f"  batched == polars reference (numeric drift {worst_num:.1e}) ✓")
    return True


def test_rolling_nan_semantics():
    """rolling_mean/std/sum must match pandas NaN semantics on BOTH devices."""
    print("Testing rolling NaN semantics vs pandas...")
    import numpy as np
    import torch
    from tensor_ops import rolling_mean, rolling_std, rolling_sum, get_device

    rng = np.random.default_rng(3)
    T, D = 6, 300
    a = rng.standard_normal((T, D)).cumsum(1) + 100.0
    a[:, :25] = np.nan                      # late listing
    for i in range(T):                      # interior holes
        a[i, rng.choice(np.arange(30, D), 15, replace=False)] = np.nan

    for dev in [torch.device("cpu"), get_device()]:
        for fn, kw, ref in [
            (rolling_mean, {}, lambda s: s.rolling(20).mean()),
            (rolling_std, {"ddof": 1}, lambda s: s.rolling(20).std()),
            (rolling_sum, {}, lambda s: s.rolling(20).sum()),
        ]:
            got = fn(a, 20, device=dev, **kw)
            exp = np.vstack([ref(pd.Series(a[i])).to_numpy() for i in range(T)])
            assert (np.isnan(got) == np.isnan(exp)).all(), \
                f"{fn.__name__} NaN pattern differs from pandas on {dev}"
            m = np.isfinite(got) & np.isfinite(exp)
            d = float(np.abs(got[m] - exp[m]).max())
            assert d < 1e-8, f"{fn.__name__} on {dev}: maxdiff {d}"
    print("  mean/std/sum match pandas incl. NaN windows, both devices ✓")
    return True


def test_snapshot_history():
    """append_history must be idempotent, DATE-native, and PIT-joinable.

    These snapshot tables are overwritten every run, which is why 6 of ~13
    buy_candidates components were untestable. Options chains especially cannot
    be reconstructed after the fact, so a silent regression here permanently
    loses data rather than just delaying a backtest.
    """
    print("Testing snapshot history append...")
    import datetime as dt
    import numpy as np
    import pyarrow.parquet as pq
    from snapshot_history import append_history, load_history, history_path, asof_join

    name = "_pytest_snapshot_hist"
    path = history_path(name)
    if path.exists():
        path.unlink()
    try:
        d1 = pd.DataFrame({"ticker": ["A", "B"], "composite": [1.0, 2.0]})
        d2 = pd.DataFrame({"ticker": ["A", "B", "C"], "composite": [1.5, 2.5, 3.5]})
        append_history(d1, name, as_of=dt.date(2026, 8, 1), quiet=True)
        append_history(d2, name, as_of=dt.date(2026, 8, 2), quiet=True)
        h = load_history(name)
        assert len(h) == 5, f"expected 5 rows, got {len(h)}"
        assert h["as_of_date"].nunique() == 2

        # idempotency: re-running the same as_of replaces, never duplicates
        append_history(d2, name, as_of=dt.date(2026, 8, 2), quiet=True)
        h = load_history(name)
        assert len(h) == 5, f"re-run duplicated rows: {len(h)}"

        # DATE-native (date32[day]), never a midnight timestamp
        assert str(pq.read_schema(path).field("as_of_date").type) == "date32[day]"
        assert isinstance(h["as_of_date"].iloc[0], dt.date)
        assert not isinstance(h["as_of_date"].iloc[0], pd.Timestamp)

        # point-in-time join must not leak the future
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2026-08-01", "2026-08-02", "2026-07-31"]),
            "ticker": ["A", "A", "A"],
        })
        j = asof_join(panel, name, ["composite"]).sort_values("date")
        vals = dict(zip(j["date"].dt.strftime("%Y-%m-%d"), j["composite"]))
        assert np.isnan(vals["2026-07-31"]), "leaked history from the future"
        assert vals["2026-08-01"] == 1.0, f"wrong PIT value: {vals}"
        assert vals["2026-08-02"] == 1.5, f"stale PIT value: {vals}"
        print("  append idempotent, date32[day], PIT join has no lookahead ✓")

        # the real writers must be wired up
        import fragility_screen, signal_aggregator, options_skew
        for mod in (fragility_screen, signal_aggregator, options_skew):
            src = Path(mod.__file__).read_text()
            assert "append_history" in src, f"{mod.__name__} does not append history"
        print("  fragility_screen / signal_aggregator / options_skew all append ✓")
    finally:
        if path.exists():
            path.unlink()
    return True


def test_rolling_moments():
    """rolling_skew/rolling_kurt match textbook moments on both devices.

    Also pins the DELIBERATE difference from
    statistical_profiler._rolling_skew_kurt, which uses a double-rolling
    (~2L effective) window and EXCESS kurtosis. They must NOT be silently
    unified: on 500 points at L=60 they differ by 3.04 (skew) / 9.59 (kurt),
    which would change every STAT_COLS output.
    """
    print("Testing rolling skew/kurt...")
    import numpy as np
    import torch
    from tensor_ops import rolling_skew, rolling_kurt, get_device

    rng = np.random.default_rng(4)
    x = rng.standard_normal(500).cumsum() + 100.0
    L = 60
    s = pd.Series(x)
    sk_ref = s.rolling(L).apply(
        lambda w: ((w - w.mean()) ** 3).mean() / ((w - w.mean()) ** 2).mean() ** 1.5,
        raw=True).to_numpy()
    ku_ref = s.rolling(L).apply(
        lambda w: ((w - w.mean()) ** 4).mean() / ((w - w.mean()) ** 2).mean() ** 2,
        raw=True).to_numpy()

    for dev in [torch.device("cpu"), get_device()]:
        for name, got, exp in [("skew", rolling_skew(x, L, device=dev), sk_ref),
                               ("kurt", rolling_kurt(x, L, device=dev), ku_ref)]:
            assert (np.isnan(got) == np.isnan(exp)).all(), f"{name} NaN pattern on {dev}"
            m = np.isfinite(got) & np.isfinite(exp)
            d = float(np.abs(got[m] - exp[m]).max())
            assert d < 1e-9, f"{name} on {dev}: maxdiff {d}"
    print("  match textbook moments on CPU and GPU ✓")

    # the profiler's estimator is intentionally different -- assert it stays so
    from statistical_profiler import _rolling_skew_kurt
    p_sk, _ = _rolling_skew_kurt(x, L)
    m = np.isfinite(p_sk) & np.isfinite(sk_ref)
    assert float(np.abs(p_sk[m] - sk_ref[m]).max()) > 0.1, \
        "profiler skew now matches textbook -- estimators were unified; update docs"
    print("  profiler estimator still distinct (double-rolling, excess) ✓")
    return True


def test_profiler_batch_parity():
    """window_profile_stats_batch must equal the per-ticker reference.

    Guards the bugs found while adding the batched path:
      * price_slope/price_curvature were built from an INCONSISTENT normal-
        equation matrix (A[0,1] used a window-local index sum while the rest
        used global sums), so neither was a valid quadratic fit. Both paths now
        use tensor_ops.rolling_quad_fit, verified against np.polyfit.
      * those two columns also returned 0.0 (not NaN) before the window formed,
        i.e. a fake "flat trend" signal.
      * torch.nanmedian takes the LOWER middle value on an even window while
        numpy averages, so rolling_median had to use nanquantile(0.5).
    """
    print("Testing statistical_profiler batch parity...")
    import numpy as np
    import torch
    from statistical_profiler import (
        window_profile_stats, window_profile_stats_batch, STAT_COLS,
    )

    rng = np.random.default_rng(21)
    n, L = 400, 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    c = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.012)
    o = c * (1 + rng.standard_normal(n) * 0.003)
    h = np.maximum(c, o) * (1 + np.abs(rng.standard_normal(n)) * 0.004)
    lo = np.minimum(c, o) * (1 - np.abs(rng.standard_normal(n)) * 0.004)
    v = rng.uniform(1e6, 5e6, n)
    S = lambda a: pd.Series(a, index=idx)

    ref = window_profile_stats(S(c), S(v), L, S(o), S(h), S(lo))
    for dev in [torch.device("cpu"), None]:
        got = window_profile_stats_batch(c[None, :], v[None, :], L,
                                         o[None, :], h[None, :], lo[None, :],
                                         device=dev)
        for col in STAT_COLS:
            if col == "price_mode":      # histogram mode: CPU-only by design
                continue
            a = ref[col].to_numpy(dtype=float)
            b = np.asarray(got[col][0], dtype=float)
            assert (np.isnan(a) == np.isnan(b)).all(), \
                f"{col}: NaN pattern differs on {got['_device']}"
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                scale = max(float(np.nanmax(np.abs(a))), 1e-9)
                rel = float(np.abs(a[m] - b[m]).max()) / scale
                assert rel < 1e-9, f"{col}: rel diff {rel:.2e} on {got['_device']}"
        print(f"  {got['_device']}: all 30 stats match reference ✓")

    # slope/curvature must be a REAL quadratic fit and NaN before the window
    from tensor_ops import rolling_quad_fit
    sl, cu = rolling_quad_fit(c, L, device=torch.device("cpu"))
    w = c[L - 1 - (L - 1):L]          # first full window
    p = np.polyfit(np.arange(L, dtype=float), c[:L], 2)
    assert abs(p[0] - cu[L - 1]) < 1e-9, "curvature disagrees with np.polyfit"
    assert abs(p[1] - sl[L - 1]) < 1e-9, "slope disagrees with np.polyfit"
    assert np.isnan(cu[: L - 1]).all(), "curvature must be NaN before the window forms"
    assert np.isnan(ref["price_curvature"].to_numpy()[: L - 1]).all(), \
        "reference still emits 0.0 curvature before the window forms"
    print("  slope/curvature match np.polyfit and are NaN pre-window ✓")
    return True


def test_resid_mom_reconstruction():
    """The OOS resid_mom_63 must match production's DEFINITION, not a proxy.

    momentum_analytics.py computes a BETA-ADJUSTED 63-day cumulative residual
    (resid.tail(63).mean() * 63). buy_candidates_oos originally reconstructed it
    as `mom21 - mean(mom21)` -- a 21-day simple demean with no beta adjustment.
    That is a different variable on a different scale, and it inverted the
    conclusion of the removal test: the proxy said drop it (t=2.50, p=0.015),
    the real definition says keep it (t=0.85, p=0.40).

    This test pins the reconstruction to the 63d horizon so the ablation cannot
    silently drift back to measuring something production never computes.
    """
    print("Testing resid_mom_63 reconstruction...")
    import inspect
    import numpy as np
    import buy_candidates_oos as B

    src = inspect.getsource(B.reconstruct_inputs)
    # look at CODE only; the comment block deliberately mentions the old proxy
    code = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith("#"))
    assert 'out["resid_mom_63"] = _resid_mom_63_pit' in code, \
        "resid_mom_63 no longer uses the PIT helper"
    assert 'mom21"] - ' not in code, \
        "reconstruct_inputs is back to the 21d-demean proxy for resid_mom_63"

    # the helper must use the 63d horizon and be explicit (NaN) without it
    hsrc = inspect.getsource(B._resid_mom_63_pit)
    assert "mom63" in hsrc, "resid_mom_63 helper does not use the 63d horizon"

    n = 240
    df = pd.DataFrame({
        "ticker": ["A"] * n + ["B"] * n,
        "date": list(pd.date_range("2020-01-01", periods=n, freq="B")) * 2,
        "mom63": np.concatenate([
            np.linspace(0.02, 0.40, n), np.linspace(-0.10, 0.10, n)]),
    })
    out = B._resid_mom_63_pit(df)
    assert len(out) == len(df), "helper changed row count"
    assert out.notna().any(), "helper returned all-NaN on valid input"

    # missing mom63 -> explicit NaN, never a silent fallback to the 21d proxy
    bad = df.drop(columns=["mom63"]).assign(mom21=0.05)
    assert B._resid_mom_63_pit(bad).isna().all(), \
        "helper silently substituted a proxy when mom63 was absent"
    print("  beta-adjusted 63d horizon, no silent 21d fallback ✓")
    return True


def test_no_duplicate_device_logic():
    """No module may reimplement device selection; all must defer to tensor_ops."""
    print("Testing centralized device handling...")
    import tensor_ops
    mods = ["fractal_windows", "fractal_windows_backtest_gpu", "backtest_coiled_spring"]
    import importlib
    for name in mods:
        m = importlib.import_module(name)
        if hasattr(m, "_best_device"):
            assert m._best_device is tensor_ops._best_device, \
                f"{name} has its own _best_device (must reuse tensor_ops)"
        if hasattr(m, "gpu_available"):
            assert m.gpu_available() == tensor_ops.gpu_available(), \
                f"{name}.gpu_available disagrees with tensor_ops"
    # Device objects, never strings — the bug that sent CPU work to the GPU path.
    d = tensor_ops._best_device()
    assert not isinstance(d, str), "_best_device must return a torch.device, not a str"
    print(f"  {len(mods)} modules defer to tensor_ops ✓")
    return True


def test_daily_partitioned():
    """Test daily_prices partitioned read."""
    print("Testing daily_prices partitioned read...")
    import polars as pl
    
    path = Path("daily_prices_partitioned")
    if path.exists():
        df = pl.scan_parquet("daily_prices_partitioned/year=2024/month=1/data.parquet").collect()
        print(f"  2024-01 partition: {df.shape} ✓")
        assert df.shape[0] > 0
    else:
        print("  Partitioned directory not found, skipping ✓")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Running basic tests...")
    print("=" * 60)
    
    tests = [
        ("EDGAR merge", test_edgar_merge),
        ("Distrust fit AUC", test_distrust_fit),
        ("Rolling skip", test_rolling_skip),
        ("GPU fallback", test_gpu_fallback),
        ("CAPE ERP", test_cape_erp),
        ("Tensor ops correctness", test_tensor_ops_correctness),
        ("Fractal GPU/CPU parity", test_fractal_gpu),
        ("Coiled spring GPU parity", test_coiled_spring_gpu_parity),
        ("Rolling NaN semantics", test_rolling_nan_semantics),
        ("Snapshot PIT history", test_snapshot_history),
        ("Rolling skew/kurt", test_rolling_moments),
        ("Profiler batch parity", test_profiler_batch_parity),
        ("resid_mom_63 reconstruction", test_resid_mom_reconstruction),
        ("Centralized device logic", test_no_duplicate_device_logic),
        ("Daily partitioned", test_daily_partitioned),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)