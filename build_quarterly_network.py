#!/usr/bin/env python3
"""
Build quarterly ownership network panels from historical 13F-HR data.

Vectorized implementation for speedy historical backfills.
Outputs:
- quarterly_holdings_panel.parquet: filer × as_of_date × held_ticker × market_value
- quarterly_network_edges.parquet: dated edges with weights
- quarterly_network_metrics.parquet: per-quarter metrics
- quarterly_lookthrough_fundamentals.parquet: per-quarter look-through
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from tqdm import tqdm

# Load historical holdings
print("Loading historical 13F holdings...")
hist = pd.read_parquet('historical_13f_holdings.parquet')
print(f"Loaded {len(hist)} rows, {hist['as_of_date'].nunique()} quarters, {hist['filer_ticker'].nunique()} filers")

# Load CUSIP to ticker map
with open('cusip_ticker_map.json') as f:
    cusip_to_ticker = json.load(f)

# Load CIK map
with open('cik_ticker_map.json') as f:
    cik_map = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}

# Load daily prices for market value conversion
prices = pd.read_parquet('daily_prices.parquet')
prices['date'] = pd.to_datetime(prices['date']).dt.date

# Load fundamentals for look-through
fund = pd.read_parquet('fundamentals.parquet')
fund['as_of_date'] = pd.to_datetime(fund['as_of_date']).dt.date

# Map CUSIP to ticker for held securities
def map_cusip_to_ticker(cusip):
    if pd.isna(cusip):
        return None
    cusip_clean = str(cusip).strip().upper()
    return cusip_to_ticker.get(cusip_clean)

hist['held_ticker'] = hist['held_cusip'].apply(map_cusip_to_ticker)

# Also map nameOfIssuer for fallback
# (could add fuzzy matching here)

# Convert held_value_thousands to market_value (USD)
hist['market_value'] = hist['held_value_thousands'] * 1000

# Keep only rows with identified tickers
hist_identified = hist.dropna(subset=['held_ticker']).copy()
print(f"Identified holdings: {len(hist_identified)} / {len(hist)} ({len(hist_identified)/len(hist)*100:.1f}%)")

# Aggregate to quarterly panel: filer × as_of_date × held_ticker
print("Building quarterly holdings panel...")
quarterly_panel = hist_identified.groupby(
    ['filer_ticker', 'as_of_date', 'held_ticker'],
    as_index=False
).agg({
    'market_value': 'sum',
    'held_shares': 'sum',
    'held_cusip': 'first',
    'filer_cik': 'first',
    'filing_date': 'min'
})

print(f"Quarterly panel: {len(quarterly_panel)} rows")
print(f"Unique filers: {quarterly_panel['filer_ticker'].nunique()}")
print(f"Unique held: {quarterly_panel['held_ticker'].nunique()}")
print(f"Quarters: {quarterly_panel['as_of_date'].nunique()}")

# Save quarterly panel
quarterly_panel.to_parquet('quarterly_holdings_panel.parquet', index=False)
print("Saved quarterly_holdings_panel.parquet")

# ============================================================
# QUARTERLY NETWORK EDGES
# ============================================================
print("\nBuilding quarterly network edges...")
edges_df = quarterly_panel[["as_of_date", "filer_ticker", "held_ticker", "market_value"]].copy()
filer_quarter_totals = edges_df.groupby(["filer_ticker", "as_of_date"])["market_value"].transform("sum")
edges_df["ownership_pct"] = edges_df["market_value"] / filer_quarter_totals.replace(0, np.nan)

print(f"Total quarterly edges: {len(edges_df)}")
edges_df.to_parquet('quarterly_network_edges.parquet', index=False)
print("Saved quarterly_network_edges.parquet")

# ============================================================
# QUARTERLY NETWORK METRICS
# ============================================================
print("\nCalculating quarterly network metrics...")

# Load nodes for attributes
nodes = pd.read_parquet('ownership_network_nodes.parquet')

quarterly_metrics = []

for q in tqdm(quarters, desc="Quarters"):
    q_edges = edges_df[edges_df['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    # Build filer overlap network for this quarter
    filers = q_edges['filer_ticker'].unique()
    held = q_edges['held_ticker'].unique()
    
    # Filer overlap (cosine similarity on holdings)
    filer_holdings = {}
    for f in filers:
        f_data = q_edges[q_edges['filer_ticker'] == f]
        filer_holdings[f] = dict(zip(f_data['held_ticker'], f_data['ownership_pct']))
    
    # Compute pairwise similarities
    filer_list = list(filers)
    n = len(filer_list)
    
    # Build holding vectors for cosine
    all_held = list(set().union(*[set(h.keys()) for h in filer_holdings.values()]))
    held_idx = {h: i for i, h in enumerate(all_held)}
    
    W = np.zeros((n, len(all_held)))
    for i, f in enumerate(filer_list):
        for h, w in filer_holdings[f].items():
            W[i, held_idx[h]] = w
    
    # Normalize
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_norm = np.divide(W, norms, where=norms>0)
    
    # Cosine similarity matrix
    cos_sim = W_norm @ W_norm.T
    np.fill_diagonal(cos_sim, 0)
    
    # Network metrics
    total_value = q_edges['market_value'].sum()
    n_filers = len(filers)
    n_held = len(held)
    
    # Concentration
    filer_values = q_edges.groupby('filer_ticker')['market_value'].sum()
    hhi = (filer_values / total_value).pow(2).sum()
    top5_concentration = filer_values.nlargest(5).sum() / total_value
    
    # Network density
    density = np.sum(cos_sim > 0.1) / (n * (n - 1)) if n > 1 else 0
    
    # Core analysis (SCCs)
    G = nx.Graph()
    G.add_nodes_from(filer_list)
    for i in range(n):
        for j in range(i+1, n):
            if cos_sim[i, j] > 0.3:
                G.add_edge(filer_list[i], filer_list[j], weight=cos_sim[i, j])
    
    sccs = list(nx.connected_components(G))  # Undirected
    core_size = max(len(c) for c in sccs) if sccs else 0
    
    quarterly_metrics.append({
        'as_of_date': q,
        'n_filers': n_filers,
        'n_held': n_held,
        'total_value': total_value,
        'hhi': hhi,
        'top5_concentration': top5_concentration,
        'network_density': density,
        'core_size': core_size,
        'n_components': len(sccs)
    })

metrics_df = pd.DataFrame(quarterly_metrics)
metrics_df.to_parquet('quarterly_network_metrics.parquet', index=False)
print("Saved quarterly_network_metrics.parquet")

# ============================================================
# QUARTERLY LOOK-THROUGH FUNDAMENTALS
# ============================================================
print("\nCalculating quarterly look-through fundamentals...")

# Key fundamental metrics to compute look-through
fund_metrics = ['ev_ebitda', 'roic', 'fcf_margin', 'debt_to_equity', 'interest_coverage']
available_metrics = [m for m in fund_metrics if m in fund.columns]
print(f"Available fundamental metrics: {available_metrics}")

lookthrough_results = []

for q in tqdm(quarters, desc="Quarters"):
    q_edges = edges_df[edges_df['as_of_date'] == q]
    if len(q_edges) == 0:
        continue
    
    # Get fundamental data as of this quarter (or latest prior)
    fund_q = fund[fund['as_of_date'] <= q].drop_duplicates(subset=['ticker', 'as_of_date'], keep='last')
    fund_q = fund_q.set_index('ticker')[available_metrics]
    
    # For each filer, compute weighted average
    for filer in q_edges['filer_ticker'].unique():
        f_edges = q_edges[q_edges['filer_ticker'] == filer]
        total_val = f_edges['market_value'].sum()
        
        if total_val == 0:
            continue
        
        # Weight by market_value
        weights = f_edges.set_index('held_ticker')['market_value'] / total_val
        
        # Align with fundamentals
        common = weights.index.intersection(fund_q.index)
        if len(common) == 0:
            continue
        
        w = weights[common]
        w = w / w.sum()  # renormalize
        
        lt_vals = {}
        for metric in available_metrics:
            vals = fund_q.loc[common, metric]
            # Remove NaN
            valid = vals.dropna()
            if len(valid) > 0:
                w_valid = w[valid.index]
                w_valid = w_valid / w_valid.sum()
                lt_vals[f'lt_{metric}'] = np.sum(w_valid * valid)
        
        if lt_vals:
            lt_vals['filer_ticker'] = filer
            lt_vals['as_of_date'] = q
            lt_vals['holdings_value'] = total_val
            lt_vals['n_holdings_with_fund'] = len(valid)
            lookthrough_results.append(lt_vals)

lt_df = pd.DataFrame(lookthrough_results)
lt_df.to_parquet('quarterly_lookthrough_fundamentals.parquet', index=False)
print("Saved quarterly_lookthrough_fundamentals.parquet")

# ============================================================
# SUMMARY
# ============================================================
print("\n=== SUMMARY ===")
print(f"Quarters: {len(quarters)} ({quarters[0]} to {quarters[-1]})")
print(f"Total quarterly edges: {len(edges_df):,}")
print(f"Quarterly metrics rows: {len(metrics_df)}")
print(f"Look-through rows: {len(lt_df)}")

# Show sample
print("\nQuarterly metrics sample:")
print(metrics_df.head(10).to_string())

print("\nLook-through sample:")
print(lt_df.head(10).to_string())