#!/usr/bin/env python3
import pandas as pd

bm = pd.read_parquet('basket_members.parquet')
sr = pd.read_parquet('subindustry_regime.parquet', columns=['basket','date','regime'])

test_tickers = ['AAPL', 'MSFT', 'JPM', 'NVDA', 'XOM']
for tk in test_tickers:
    rows = bm[(bm['ticker'] == tk) & (bm['basket_kind'] == 'gics_sector')]
    if rows.empty:
        print(f'{tk}: NO gics_sector basket')
    else:
        sector_basket = rows.iloc[0]['basket']
        sub = sr[sr['basket'] == sector_basket]
        print(f'{tk}: sector={sector_basket}, regime rows={len(sub)}')
        if len(sub) > 0:
            print(f'  date range: {sub["date"].min()} -> {sub["date"].max()}')
            print(f'  regimes: {sub["regime"].value_counts().to_dict()}')