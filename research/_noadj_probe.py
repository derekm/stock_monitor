import pandas as pd
PRICES = "C:/Users/derek/src/stockmagic/stock_monitor/daily_prices/"
raw = pd.read_parquet(PRICES)
print("rows", len(raw), "cols", list(raw.columns))
no_adj = set()
for tk, sub in raw.groupby("ticker"):
    if "adj_close" not in sub.columns or sub["adj_close"].isna().all():
        no_adj.add(tk); continue
    c = sub["close"]; a = sub["adj_close"]
    m = (c > 0) & (a > 0) & c.notna() & a.notna()
    if m.sum() == 0:
        no_adj.add(tk); continue
    rel = (a[m] - c[m]).abs() / c[m].abs()
    if rel.mean() < 1e-4:
        no_adj.add(tk)
print("total tickers", raw["ticker"].nunique(), "no_adj", len(no_adj))
print("sample no_adj", sorted(list(no_adj))[:20])
