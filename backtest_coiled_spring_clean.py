#!/usr/bin/env python3
"""Clean coiled spring backtest using proven indicators from coiled_spring.py"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def compute_states_for_ticker(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['ret'] = d['close'].pct_change()
    d['bb_mid'] = d['close'].rolling(20).mean()
    d['bb_std'] = d['close'].rolling(20).std()
    d['bb_upper'] = d['bb_mid'] + 2 * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - 2 * d['bb_std']
    d['bb_width'] = (d['bb_upper'] - d['bb_lower']) / d['bb_mid']
    d['bb_pos'] = (d['close'] - d['bb_lower']) / (d['bb_upper'] - d['bb_lower'])
    tr = pd.concat([
        d['high'] - d['low'],
        (d['high'] - d['close'].shift()).abs(),
        (d['low'] - d['close'].shift()).abs()
    ], axis=1).max(axis=1)
    d['atr'] = tr.rolling(20).mean()
    d['kc_mid'] = d['close'].rolling(20).mean()
    d['squeeze_on'] = (d['bb_upper'] < d['kc_mid'] + 1.5 * d['atr']) & (d['bb_lower'] > d['kc_mid'] - 1.5 * d['atr'])
    d['squeeze_20d'] = d['squeeze_on'].rolling(20).sum()
    d['vol_z'] = (d['volume'] - d['volume'].rolling(20).mean()) / d['volume'].rolling(20).std()
    d['bb_width_p252'] = d['bb_width'].rolling(252).rank(pct=True)
    return d

def main():
    print('=== Clean Coiled Spring Backtest ===')
    px = pd.read_parquet('daily_prices/')
    px['date'] = pd.to_datetime(px['date'])
    tickers = sorted(px['ticker'].unique())
    print(f'Tickers: {len(tickers)}')

    events = []
    for i, t in enumerate(tickers):
        if i % 100 == 0:
            print(f'{i}/{len(tickers)}')
        d = px[px['ticker'] == t].set_index('date').sort_index()
        if len(d) < 300:
            continue
        d = compute_states_for_ticker(d)
        for j in range(252, len(d)):
            row = d.iloc[j]
            if row['squeeze_20d'] >= 10:
                events.append({'ticker': t, 'date': row.name, 'state': 'coiled', 'close': row['close'], 'bb_width': row['bb_width'], 'vol_z': row['vol_z']})
            if pd.notna(row['bb_width_p252']) and row['bb_width_p252'] <= 0.25:
                events.append({'ticker': t, 'date': row.name, 'state': 'tight', 'close': row['close'], 'bb_width': row['bb_width'], 'vol_z': row['vol_z']})
            if row['bb_pos'] < 0 and row['vol_z'] > 1.5:
                events.append({'ticker': t, 'date': row.name, 'state': 'test', 'bb_width': row['bb_width'], 'vol_z': row['vol_z'], 'close': row['close']})

    ev = pd.DataFrame(events)
    print('Total events:', len(ev))
    print(ev['state'].value_counts())
    ev.to_parquet('coiled_spring_full_events.parquet', index=False)
    print('Saved coiled_spring_full_events.parquet')

    # Simple shadow book on 'test' entries
    print('\\nBuilding shadow book on test entries...')
    book = []
    for t in ev['ticker'].unique():
        tests = ev[(ev['ticker'] == t) & (ev['state'] == 'test')].sort_values('date')
        d = px[px['ticker'] == t].set_index('date')['close']
        for _, r in tests.iterrows():
            ed = r['date']
            ep = r['close']
            fut = d[d.index > ed].iloc[:63]
            if len(fut) < 5: continue
            ret = (fut.iloc[-1] / ep) - 1
            book.append({'ticker': t, 'entry': ed, 'ret': ret, 'days': len(fut)})

    b = pd.DataFrame(book)
    print('Shadow book n:', len(b))
    if len(b) > 0:
        print('Mean ret:', round(b['ret'].mean() * 100, 2), '%')
        print('Median ret:', round(b['ret'].median() * 100, 2), '%')
        print('Hit rate (>0):', round((b['ret'] > 0).mean(), 3))

    b.to_parquet('shadow_book_clean.parquet', index=False)
    print('Done')

if __name__ == '__main__':
    main()
