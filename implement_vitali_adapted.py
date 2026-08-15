#!/usr/bin/env python3
"""
Implement Vitali et al. (2011) techniques adapted for institutional holdings network.

Our network: Filers (13F-HR managers) → Held companies (public equities)
Adaptations:
1. Proportional control (ownership % = control %) instead of threshold
2. Filer-to-filer overlap network (common holdings = indirect connections)
3. Control flow through common holdings
4. Concentration analysis (HHI, top-k) - already done
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import json

# Load our ownership network
edges = pd.read_parquet('ownership_network_edges.parquet')
nodes = pd.read_parquet('ownership_network_nodes.parquet')

print(f"Loaded {len(edges)} edges, {len(nodes)} nodes")

# ============================================================
# 1. BUILD BIPARTITE GRAPH + FILER OVERLAP NETWORK
# ============================================================
print("\n=== BIPARTITE & OVERLAP NETWORKS ===")

# Bipartite graph
B = nx.Graph()
filers = set(edges['filer_ticker'].unique())
held = set(edges['held_ticker'].unique())

for f in filers:
    B.add_node(f, bipartite=0, type='filer')
for h in held:
    B.add_node(h, bipartite=1, type='held')

for _, row in edges.iterrows():
    B.add_edge(row['filer_ticker'], row['held_ticker'], 
               weight=row['market_value'],
               ownership_pct=row['ownership_pct'] if 'ownership_pct' in row else 0)

print(f"Bipartite graph: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

# Project to filer-filer network (common holdings)
# Weight = sum of min(ownership_pct) or Jaccard / cosine similarity
filer_holdings = defaultdict(dict)
for _, row in edges.iterrows():
    filer_holdings[row['filer_ticker']][row['held_ticker']] = row['market_value']

# Build filer overlap network
filer_list = list(filers)
G_filers = nx.Graph()
G_filers.add_nodes_from(filer_list)

for i, f1 in enumerate(filer_list):
    for f2 in filer_list[i+1:]:
        common = set(filer_holdings[f1].keys()) & set(filer_holdings[f2].keys())
        if common:
            # Multiple similarity metrics
            # 1. Overlap value
            overlap_val = sum(min(filer_holdings[f1][h], filer_holdings[f2][h]) for h in common)
            # 2. Cosine similarity
            v1 = np.array([filer_holdings[f1].get(h, 0) for h in common])
            v2 = np.array([filer_holdings[f2].get(h, 0) for h in common])
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0 else 0
            # 3. Jaccard
            union = set(filer_holdings[f1].keys()) | set(filer_holdings[f2].keys())
            jaccard = len(common) / len(union) if union else 0
            
            G_filers.add_edge(f1, f2, 
                             overlap_value=overlap_val,
                             cosine_sim=cos_sim,
                             jaccard=jaccard,
                             n_common=len(common))

print(f"Filer overlap network: {G_filers.number_of_nodes()} nodes, {G_filers.number_of_edges()} edges")

# ============================================================
# 2. PROPORTIONAL NETWORK CONTROL (not threshold)
# ============================================================
print("\n=== PROPORTIONAL NETWORK CONTROL ===")

# In institutional context: control = ownership percentage
# Network control = direct + indirect through common holdings

# Build directed filer->filer control via common holdings
# If filer A and B both hold X, A's control flows to B proportionally

# Direct ownership matrix (filer x held)
filer_idx = {f: i for i, f in enumerate(filer_list)}
held_list = list(held)
held_idx = {h: i for i, h in enumerate(held_list)}

n_filers = len(filer_list)
n_held = len(held_list)

W_fh = np.zeros((n_filers, n_held))  # filer x held ownership %
for _, row in edges.iterrows():
    i = filer_idx[row['filer_ticker']]
    j = held_idx[row['held_ticker']]
    W_fh[i, j] = row['ownership_pct'] if 'ownership_pct' in row else 0

# Normalize rows to sum to 1 (proportional control)
row_sums = W_fh.sum(axis=1, keepdims=True)
W_fh_norm = np.divide(W_fh, row_sums, where=row_sums>0)

# Filer-to-filer control through common holdings
# C_ff = W_fh_norm @ W_fh_norm.T  (co-ownership matrix)
C_ff = W_fh_norm @ W_fh_norm.T
np.fill_diagonal(C_ff, 0)  # remove self

print(f"Filer-filer control matrix: {C_ff.shape}")
print(f"Non-zero entries: {np.sum(C_ff > 0)}")

# Network control: iterative (like PageRank but for control)
alpha = 0.85  # damping
C_net_ff = np.eye(n_filers) * 0.15 + alpha * C_ff
for _ in range(50):
    C_new = np.eye(n_filers) * 0.15 + alpha * C_ff @ C_net_ff
    if np.max(np.abs(C_new - C_net_ff)) < 1e-6:
        break
    C_net_ff = C_new

# Total control per filer
filer_control = C_net_ff.sum(axis=1)
filer_control_df = pd.DataFrame({
    'filer': filer_list,
    'net_control': filer_control,
    'direct_overlap': C_ff.sum(axis=1)
}).sort_values('net_control', ascending=False)

print("\nFiler Network Control (proportional):")
print(filer_control_df.to_string())

# ============================================================
# 3. BOW-TIE ON FILER OVERLAP NETWORK
# ============================================================
print("\n=== BOW-TIE ON FILER OVERLAP NETWORK ===")

# Use cosine similarity > threshold as directed edges for bow-tie
threshold = 0.3
G_filers_directed = nx.DiGraph()
G_filers_directed.add_nodes_from(filer_list)

for u, v, data in G_filers.edges(data=True):
    if data['cosine_sim'] > threshold:
        G_filers_directed.add_edge(u, v, weight=data['cosine_sim'])
        G_filers_directed.add_edge(v, u, weight=data['cosine_sim'])  # symmetric

print(f"Directed filer network (cosine>{threshold}): {G_filers_directed.number_of_nodes()} nodes, {G_filers_directed.number_of_edges()} edges")

# Weakly connected components
weak_cc = list(nx.weakly_connected_components(G_filers_directed))
largest_cc = max(weak_cc, key=len)
print(f"Weakly CCs: {len(weak_cc)}, Largest: {len(largest_cc)}")

# SCCs
G_lcc = G_filers_directed.subgraph(largest_cc).copy()
sccs = list(nx.strongly_connected_components(G_lcc))
sccs_sorted = sorted(sccs, key=len, reverse=True)
core = sccs_sorted[0]
print(f"SCCs: {len(sccs)}, Core: {len(core)} nodes - {sorted(core)}")

# Bow-tie sections
in_sec, out_sec, tnt_sec = set(), set(), set()
for node in G_lcc.nodes():
    if node in core:
        continue
    can_reach_core = any(nx.has_path(G_lcc, node, c) for c in core)
    core_can_reach = any(nx.has_path(G_lcc, c, node) for c in core)
    if can_reach_core and not core_can_reach:
        in_sec.add(node)
    elif core_can_reach and not can_reach_core:
        out_sec.add(node)
    else:
        tnt_sec.add(node)

print(f"IN: {len(in_sec)} - {sorted(in_sec)}")
print(f"SCC: {len(core)} - {sorted(core)}")
print(f"OUT: {len(out_sec)} - {sorted(out_sec)}")
print(f"T&T: {len(tnt_sec)} - {sorted(tnt_sec)}")

# ============================================================
# 4. CONCENTRATION & "SUPER-ENTITY" IN FILER NETWORK
# ============================================================
print("\n=== CONCENTRATION ANALYSIS ===")

# Market value controlled by each filer
filer_values = edges.groupby('filer_ticker')['market_value'].sum()
total_value = filer_values.sum()

# Top control holders by network control * value
filer_control_df = filer_control_df.set_index('filer')
filer_control_df['holdings_value'] = filer_control_df.index.map(filer_values).fillna(0)
filer_control_df['control_value'] = filer_control_df['net_control'] * filer_control_df['holdings_value']

print(f"Total holdings value: ${total_value/1e9:.2f}B")
print("\nTop filers by control value:")
for _, row in filer_control_df.head(10).iterrows():
    print(f"  {row.name:6s}: net_control={row['net_control']:.3f}, holdings=${row['holdings_value']/1e9:.2f}B, control_val=${row['control_value']/1e9:.2f}B")

# Core analysis
if len(core) > 0:
    core_filers = [f for f in core if f in filer_control_df.index]
    core_control = filer_control_df.loc[core_filers]
    print(f"\nCore filers ({len(core_filers)}): {core_filers}")
    print(f"Core share of net control: {core_control['net_control'].sum() / filer_control_df['net_control'].sum() * 100:.1f}%")
    print(f"Core share of control value: {core_control['control_value'].sum() / filer_control_df['control_value'].sum() * 100:.1f}%")

# Top 50% control holders
top_half = filer_control_df.head(len(filer_control_df)//2)
print(f"\nTop 50% filers ({len(top_half)}): {top_half['net_control'].sum() / filer_control_df['net_control'].sum() * 100:.1f}% of network control")

# ============================================================
# 5. HELD COMPANY CENTRALITY (which stocks are most "controlled")
# ============================================================
print("\n=== HELD COMPANY CENTRALITY ===")

# Held company centrality: how many filers hold it, weighted by ownership
held_centrality = edges.groupby('held_ticker').agg(
    n_filers=('filer_ticker', 'count'),
    total_ownership_pct=('ownership_pct', 'sum') if 'ownership_pct' in edges.columns else ('market_value', 'sum'),
    total_value=('market_value', 'sum')
).sort_values('total_value', ascending=False)

print("Top 20 held companies by total value held:")
print(held_centrality.head(20).to_string())

# Betweenness centrality in bipartite graph (which held companies connect filers)
held_betweenness = nx.betweenness_centrality(B, normalized=True)
held_bw = {k: v for k, v in held_betweenness.items() if k in held}
held_bw_sorted = sorted(held_bw.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 held companies by betweenness (connect filers):")
for h, bw in held_bw_sorted[:20]:
    n_f = held_centrality.loc[h, 'n_filers'] if h in held_centrality.index else 0
    val = held_centrality.loc[h, 'total_value'] if h in held_centrality.index else 0
    print(f"  {h:6s}: betweenness={bw:.4f}, n_filers={n_f}, value=${val/1e9:.2f}B")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n=== SAVING RESULTS ===")

# Filer overlap network
filer_overlap_edges = []
for u, v, data in G_filers.edges(data=True):
    filer_overlap_edges.append({
        'filer_1': u, 'filer_2': v,
        'overlap_value': data['overlap_value'],
        'cosine_sim': data['cosine_sim'],
        'jaccard': data['jaccard'],
        'n_common': data['n_common']
    })
pd.DataFrame(filer_overlap_edges).to_parquet('filer_overlap_network.parquet', index=False)
print("Saved filer_overlap_network.parquet")

# Filer network control
filer_control_df.reset_index().to_parquet('filer_network_control.parquet', index=False)
print("Saved filer_network_control.parquet")

# Bow-tie on filers
bowtie_filers = pd.DataFrame({
    'filer': filer_list,
    'bowtie_section': ['SCC' if f in core else 'IN' if f in in_sec else 'OUT' if f in out_sec else 'T&T' for f in filer_list]
})
bowtie_filers.to_parquet('filer_bowtie_decomposition.parquet', index=False)
print("Saved filer_bowtie_decomposition.parquet")

# Held company centrality
held_centrality.reset_index().to_parquet('held_company_centrality.parquet', index=False)
print("Saved held_company_centrality.parquet")

# Summary
summary = {
    'n_filers': n_filers,
    'n_held': n_held,
    'bipartite_edges': B.number_of_edges(),
    'filer_overlap_edges': G_filers.number_of_edges(),
    'filer_network_density': nx.density(G_filers),
    'filer_sccs': len(sccs),
    'core_size': len(core),
    'core_filers': list(core),
    'in_size': len(in_sec),
    'out_size': len(out_sec),
    'tnt_size': len(tnt_sec),
    'total_holdings_value': float(total_value),
    'top_filer_control_share': float(filer_control_df.head(1)['net_control'].values[0] / filer_control_df['net_control'].sum()),
    'top5_filer_control_share': float(filer_control_df.head(5)['net_control'].sum() / filer_control_df['net_control'].sum()),
}

with open('vitali_adapted_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved vitali_adapted_summary.json")

print("\n=== DONE ===")