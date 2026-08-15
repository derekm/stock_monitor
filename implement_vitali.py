#!/usr/bin/env python3
"""
Implement Vitali et al. (2011) "The Network of Global Corporate Control" techniques
on our ownership network.

Techniques:
1. Bow-tie decomposition (IN, SCC, OUT, T&T)
2. Network control with cycle correction
3. Threshold-based control model (50% rule)
4. Core-periphery / "super-entity" identification
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
print(f"Filers: {edges['filer_ticker'].nunique()}")
print(f"Held: {edges['held_ticker'].nunique()}")

# Build directed graph: filer -> held (ownership direction)
G = nx.DiGraph()

# Add nodes with attributes
for _, row in nodes.iterrows():
    G.add_node(row['ticker'], 
               type=row['node_type'],
               sector=row.get('sector', ''),
               industry=row.get('industry', ''),
               market_cap=row.get('market_cap', 0))

# Add edges with ownership weights
# Normalize by filer's total holdings to get ownership percentage
filer_totals = edges.groupby('filer_ticker')['market_value'].sum()

for _, row in edges.iterrows():
    filer = row['filer_ticker']
    held = row['held_ticker']
    value = row['market_value']
    total = filer_totals[filer]
    ownership_pct = value / total if total > 0 else 0
    G.add_edge(filer, held, 
               weight=ownership_pct,
               market_value=value)

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ============================================================
# 1. BOW-TIE DECOMPOSITION
# ============================================================
print("\n=== BOW-TIE DECOMPOSITION ===")

# Find weakly connected components
weak_components = list(nx.weakly_connected_components(G))
largest_cc = max(weak_components, key=len)
print(f"Weakly connected components: {len(weak_components)}")
print(f"Largest CC: {len(largest_cc)} nodes ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")

# Subgraph of largest CC
G_lcc = G.subgraph(largest_cc).copy()

# Find strongly connected components (SCCs)
sccs = list(nx.strongly_connected_components(G_lcc))
sccs_sorted = sorted(sccs, key=len, reverse=True)
print(f"Strongly connected components: {len(sccs)}")
print(f"Largest SCC (core): {len(sccs_sorted[0])} nodes")
print(f"Core nodes: {sorted(sccs_sorted[0])}")

# Bow-tie decomposition
core = sccs_sorted[0]

# IN section: nodes that can reach core but not reachable from core
in_section = set()
# OUT section: nodes reachable from core but cannot reach core
out_section = set()
# T&T: nodes in LCC but not in IN, OUT, or core
tnt_section = set()

for node in G_lcc.nodes():
    if node in core:
        continue
    # Can node reach core?
    can_reach_core = any(nx.has_path(G_lcc, node, c) for c in core)
    # Can core reach node?
    core_can_reach = any(nx.has_path(G_lcc, c, node) for c in core)
    
    if can_reach_core and not core_can_reach:
        in_section.add(node)
    elif core_can_reach and not can_reach_core:
        out_section.add(node)
    elif can_reach_core and core_can_reach:
        # This shouldn't happen if core is maximal SCC, but handle
        tnt_section.add(node)
    else:
        tnt_section.add(node)

print(f"\nBow-tie sections:")
print(f"  IN: {len(in_section)} nodes - {sorted(in_section)}")
print(f"  SCC (core): {len(core)} nodes - {sorted(core)}")
print(f"  OUT: {len(out_section)} nodes - {sorted(out_section)}")
print(f"  T&T: {len(tnt_section)} nodes - {sorted(tnt_section)}")

# ============================================================
# 2. NETWORK CONTROL WITH CYCLE CORRECTION
# ============================================================
print("\n=== NETWORK CONTROL COMPUTATION ===")

# Threshold model: 50% ownership = full control
def threshold_control(ownership_pct, threshold=0.5):
    """Threshold model: >threshold = full control (1), else 0"""
    return 1.0 if ownership_pct > threshold else 0.0

# Build control matrix C where C[i,j] = control of i over j
nodes_list = list(G_lcc.nodes())
n = len(nodes_list)
node_to_idx = {node: i for i, node in enumerate(nodes_list)}

# Direct control matrix
C_direct = np.zeros((n, n))
W_direct = np.zeros((n, n))

for u, v, data in G_lcc.edges(data=True):
    i, j = node_to_idx[u], node_to_idx[v]
    W_direct[i, j] = data['weight']
    C_direct[i, j] = threshold_control(data['weight'])

print(f"Direct ownership matrix: {W_direct.shape}")
print(f"Direct control matrix (threshold 50%): {np.sum(C_direct > 0)} edges with control")

# Network control: C_net = C_direct + C_direct @ C_net
# Solve: C_net = (I - C_direct)^-1 @ C_direct - but need to handle cycles
# Use iterative method with convergence check
C_net = C_direct.copy()
max_iter = 100
tol = 1e-6

for iteration in range(max_iter):
    C_new = C_direct + C_direct @ C_net
    # Threshold again to prevent overestimation
    C_new = np.where(C_new > 0.5, 1.0, 0.0)
    diff = np.max(np.abs(C_new - C_net))
    C_net = C_new
    if diff < tol:
        print(f"Converged after {iteration+1} iterations")
        break
else:
    print(f"Did not converge after {max_iter} iterations, max diff: {diff}")

# Total network control per node (sum of control over all others)
net_control = C_net.sum(axis=1)
control_df = pd.DataFrame({
    'ticker': nodes_list,
    'net_control': net_control,
    'direct_control_out': C_direct.sum(axis=1),
    'is_core': [t in core for t in nodes_list],
    'bowtie_section': ['SCC' if t in core else 'IN' if t in in_section else 'OUT' if t in out_section else 'T&T' for t in nodes_list]
})

control_df = control_df.sort_values('net_control', ascending=False)
print("\nTop 20 by network control:")
print(control_df.head(20).to_string())

# ============================================================
# 3. CONTROL VALUE (weighted by market cap / economic value)
# ============================================================
print("\n=== CONTROL VALUE (ECONOMIC VALUE) ===")

# Get market values for nodes
node_values = {}
for node in G_lcc.nodes():
    # Use total holdings value for filers, market cap for held
    filer_edges = edges[edges['filer_ticker'] == node]
    if len(filer_edges) > 0:
        node_values[node] = filer_edges['market_value'].sum()
    else:
        # Held company - try to get market cap from nodes
        node_data = nodes[nodes['ticker'] == node]
        if len(node_data) > 0 and 'market_cap' in node_data.columns:
            mc = node_data['market_cap'].values[0]
            if pd.notna(mc) and mc > 0:
                node_values[node] = mc
            else:
                node_values[node] = 1e9  # default
        else:
            node_values[node] = 1e9

# Control value: sum of control * value of controlled entity
control_values = []
for i, node in enumerate(nodes_list):
    val = sum(C_net[i, j] * node_values.get(nodes_list[j], 1e9) for j in range(n))
    control_values.append(val)

control_df['control_value'] = control_values
control_df = control_df.sort_values('control_value', ascending=False)

print("\nTop 20 by control value ($):")
for _, row in control_df.head(20).iterrows():
    print(f"  {row['ticker']:6s}: ${row['control_value']/1e9:.2f}B (net_control={row['net_control']:.0f}, section={row['bowtie_section']})")

# ============================================================
# 4. SUPER-ENTITY ANALYSIS
# ============================================================
print("\n=== SUPER-ENTITY ANALYSIS ===")

# Core statistics
core_control = control_df[control_df['is_core']]
print(f"Core nodes: {len(core_control)}")
print(f"Core total network control: {core_control['net_control'].sum():.1f}")
print(f"Core total control value: ${core_control['control_value'].sum()/1e9:.2f}B")

# All network control
total_control = control_df['net_control'].sum()
total_control_value = control_df['control_value'].sum()
print(f"\nTotal network control: {total_control:.1f}")
print(f"Total control value: ${total_control_value/1e9:.2f}B")
print(f"Core share of network control: {core_control['net_control'].sum()/total_control*100:.1f}%")
print(f"Core share of control value: {core_control['control_value'].sum()/total_control_value*100:.1f}%")

# Top control holders
top_50 = control_df.head(50)
print(f"\nTop 50 control holders: {top_50['net_control'].sum():.1f} network control ({top_50['net_control'].sum()/total_control*100:.1f}%)")
print(f"Top 50 control value: ${top_50['control_value'].sum()/1e9:.2f}B ({top_50['control_value'].sum()/total_control_value*100:.1f}%)")

# Financial vs non-financial in top control holders
financial_sectors = ['Financial Services', 'Banks', 'Insurance', 'Asset Management', 'Capital Markets']
top_50_financial = top_50[top_50['ticker'].isin(nodes[nodes['sector'].isin(financial_sectors)]['ticker'])]
print(f"Financial in top 50: {len(top_50_financial)}/{len(top_50)} = {len(top_50_financial)/len(top_50)*100:.1f}%")

# ============================================================
# 5. CYCLE DETECTION
# ============================================================
print("\n=== CYCLE DETECTION ===")

# Find all simple cycles (limited to avoid explosion)
cycles = list(nx.simple_cycles(G_lcc))
print(f"Total simple cycles found: {len(cycles)}")

# Cycle length distribution
cycle_lengths = defaultdict(int)
for cycle in cycles:
    cycle_lengths[len(cycle)] += 1

for length, count in sorted(cycle_lengths.items()):
    print(f"  Length {length}: {count} cycles")

# 2-cycles (mutual cross-shareholdings)
cycles_2 = [c for c in cycles if len(c) == 2]
print(f"\n2-cycles (mutual cross-shareholdings): {len(cycles_2)}")
for c in cycles_2[:10]:
    u, v = c[0], c[1]
    w_uv = G_lcc[u][v]['weight']
    w_vu = G_lcc[v][u]['weight']
    print(f"  {u} <-> {v}: {w_uv:.2%} / {w_vu:.2%}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n=== SAVING RESULTS ===")

# Save bow-tie classification
bowtie_df = control_df[['ticker', 'bowtie_section', 'is_core', 'net_control', 'control_value']].copy()
bowtie_df.to_parquet('bowtie_decomposition.parquet', index=False)
print("Saved bowtie_decomposition.parquet")

# Save network control matrix
np.save('network_control_matrix.npy', C_net)
with open('network_control_nodes.json', 'w') as f:
    json.dump(nodes_list, f)
print("Saved network_control_matrix.npy and network_control_nodes.json")

# Save cycle data
cycles_df = pd.DataFrame([{'cycle': c, 'length': len(c)} for c in cycles])
cycles_df.to_parquet('ownership_cycles.parquet', index=False)
print("Saved ownership_cycles.parquet")

# Summary stats
summary = {
    'total_nodes': int(G.number_of_nodes()),
    'total_edges': int(G.number_of_edges()),
    'lcc_nodes': int(len(largest_cc)),
    'lcc_edges': int(G_lcc.number_of_edges()),
    'num_sccs': len(sccs),
    'core_size': int(len(core)),
    'in_size': int(len(in_section)),
    'out_size': int(len(out_section)),
    'tnt_size': int(len(tnt_section)),
    'total_cycles': len(cycles),
    'cycles_2': len(cycles_2),
    'total_net_control': float(total_control),
    'total_control_value': float(total_control_value),
    'core_net_control_share': float(core_control['net_control'].sum() / total_control),
    'core_control_value_share': float(core_control['control_value'].sum() / total_control_value),
    'top50_net_control_share': float(top_50['net_control'].sum() / total_control),
    'top50_control_value_shade': float(top_50['control_value'].sum() / total_control_value),
}

with open('vitali_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved vitali_analysis_summary.json")

print("\n=== DONE ===")