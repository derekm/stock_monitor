#!/usr/bin/env python3
"""
build_ownership_network.py — Build ownership network from detailed holdings panel.

Input: holdings_panel_enriched.parquet
Output: 
  - ownership_network_edges.parquet (filer_ticker, held_ticker, as_of_date, market_value, shares, concept)
  - ownership_network_nodes.parquet (ticker, sector, market_cap, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def build_ownership_network(holdings_path: str, prices_path: str, fundamentals_path: str, 
                           edges_output: str, nodes_output: str):
    """Build ownership network edges and nodes from holdings panel"""
    
    # Load data
    holdings = pd.read_parquet(holdings_path)
    prices = pd.read_parquet(prices_path)
    fundamentals = pd.read_parquet(fundamentals_path)
    
    # Ensure date types
    holdings["as_of_date"] = pd.to_datetime(holdings["as_of_date"]).dt.date
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    if "as_of_date" in fundamentals.columns:
        fundamentals["as_of_date"] = pd.to_datetime(fundamentals["as_of_date"]).dt.date
    if "date" in fundamentals.columns:
        fundamentals["date"] = pd.to_datetime(fundamentals["date"]).dt.date
    
    # Filter for holdings with identified held_ticker and market_value
    edges_df = holdings[
        (holdings["held_ticker"].notna()) & 
        (holdings["market_value"].notna()) & 
        (holdings["market_value"] > 0)
    ].copy()
    
    print(f"Building edges from {len(edges_df)} holdings with market values")
    
    # Aggregate by (filer, held, date) - sum market values
    edges_agg = edges_df.groupby(["filer_ticker", "held_ticker", "as_of_date"]).agg({
        "market_value": "sum",
        "value": "sum",  # original value (could be shares or USD)
        "value_type": "first",
        "concept": lambda x: ", ".join(x.unique()),
        "held_cik": "first",
    }).reset_index()
    
    edges_agg.columns = ["filer_ticker", "held_ticker", "as_of_date", "market_value", 
                         "original_value", "value_type", "concepts", "held_cik"]
    
    # Add weight (market value as weight for network analysis)
    edges_agg["weight"] = edges_agg["market_value"]
    
    # Save edges
    edges_agg.to_parquet(edges_output, index=False)
    print(f"Saved {len(edges_agg)} edges to {edges_output}")
    
    # Build nodes - all unique tickers (filers + held)
    all_tickers = set(edges_agg["filer_ticker"].unique()) | set(edges_agg["held_ticker"].unique())
    
    # Get latest fundamental data for each ticker
    if "as_of_date" in fundamentals.columns:
        latest_fund = fundamentals.sort_values("as_of_date").groupby("ticker").last().reset_index()
    else:
        latest_fund = fundamentals.groupby("ticker").last().reset_index()
    
    # Get latest market cap from PIT panel
    mcap_path = Path(prices_path).parent / "daily_mcap.parquet"
    if mcap_path.exists():
        latest_prices = pd.read_parquet(mcap_path, columns=["date", "ticker", "market_cap"])
    else:
        latest_prices = prices.sort_values("date").groupby("ticker").last().reset_index()
    latest_prices = latest_prices.sort_values("date").groupby("ticker").last().reset_index()
    market_caps = latest_prices[["ticker", "market_cap"]].rename(columns={"market_cap": "latest_market_cap"})
    
    # Build nodes
    nodes_list = []
    for ticker in all_tickers:
        node = {"ticker": ticker}
        
        # Get sector from fundamentals
        fund_row = latest_fund[latest_fund["ticker"] == ticker]
        if len(fund_row) > 0:
            if "gics_sector" in fund_row.columns:
                node["sector"] = fund_row["gics_sector"].values[0]
            elif "sector" in fund_row.columns:
                node["sector"] = fund_row["sector"].values[0]
            else:
                node["sector"] = None
                
            if "gics_industry" in fund_row.columns:
                node["industry"] = fund_row["gics_industry"].values[0]
            elif "industry" in fund_row.columns:
                node["industry"] = fund_row["industry"].values[0]
            else:
                node["industry"] = None
        else:
            node["sector"] = None
            node["industry"] = None
        
        # Get market cap
        cap_row = market_caps[market_caps["ticker"] == ticker]
        if len(cap_row) > 0 and pd.notna(cap_row["latest_market_cap"].values[0]):
            node["market_cap"] = cap_row["latest_market_cap"].values[0]
        else:
            node["market_cap"] = None
        
        # Node type: filer, held, or both
        is_filer = ticker in edges_agg["filer_ticker"].values
        is_held = ticker in edges_agg["held_ticker"].values
        if is_filer and is_held:
            node["node_type"] = "both"
        elif is_filer:
            node["node_type"] = "filer"
        else:
            node["node_type"] = "held"
        
        nodes_list.append(node)
    
    nodes_df = pd.DataFrame(nodes_list)
    nodes_df.to_parquet(nodes_output, index=False)
    print(f"Saved {len(nodes_df)} nodes to {nodes_output}")
    
    # Print summary
    print(f"\nNetwork Summary:")
    print(f"  Total edges: {len(edges_agg)}")
    print(f"  Total nodes: {len(nodes_df)}")
    print(f"  Filer nodes: {(nodes_df['node_type'] == 'filer').sum()}")
    print(f"  Held nodes: {(nodes_df['node_type'] == 'held').sum()}")
    print(f"  Both nodes: {(nodes_df['node_type'] == 'both').sum()}")
    print(f"  Date range: {edges_agg['as_of_date'].min()} to {edges_agg['as_of_date'].max()}")
    
    return edges_agg, nodes_df


def calculate_network_metrics(edges_path: str, nodes_path: str, 
                             metrics_output: str, lookthrough_output: str):
    """Calculate network metrics and look-through EV/EBITDA"""
    
    edges = pd.read_parquet(edges_path)
    nodes = pd.read_parquet(nodes_path)
    
    # Ensure date type
    edges["as_of_date"] = pd.to_datetime(edges["as_of_date"]).dt.date
    
    # Load fundamentals for EV/EBITDA
    fundamentals = pd.read_parquet("fundamentals.parquet")
    if "as_of_date" in fundamentals.columns:
        fundamentals["as_of_date"] = pd.to_datetime(fundamentals["as_of_date"]).dt.date
    
    # Get EV/EBITDA for held companies
    # EV = market_cap + total_debt - cash
    # We'll use market_cap from prices and debt/cash from fundamentals
    
    # For each date, calculate look-through metrics for each filer
    results = []
    
    for as_of_date in edges["as_of_date"].unique():
        date_edges = edges[edges["as_of_date"] == as_of_date].copy()
        
        for filer in date_edges["filer_ticker"].unique():
            filer_edges = date_edges[date_edges["filer_ticker"] == filer]
            total_portfolio_value = filer_edges["market_value"].sum()
            
            if total_portfolio_value == 0:
                continue
            
            # Portfolio concentration (Herfindahl index)
            weights = filer_edges["market_value"] / total_portfolio_value
            hhi = (weights ** 2).sum()
            
            # Number of holdings
            n_holdings = len(filer_edges)
            
            # Top 5 concentration
            top5_weight = filer_edges.nlargest(5, "market_value")["market_value"].sum() / total_portfolio_value
            
            # Sector exposure
            if "sector" in nodes.columns:
                filer_edges = filer_edges.merge(nodes[["ticker", "sector"]], 
                                               left_on="held_ticker", right_on="ticker", how="left")
                sector_exposure = filer_edges.groupby("sector")["market_value"].sum() / total_portfolio_value
                sector_entropy = - (sector_exposure * np.log(sector_exposure + 1e-10)).sum()
                top_sector = sector_exposure.idxmax() if len(sector_exposure) > 0 else None
                top_sector_weight = sector_exposure.max() if len(sector_exposure) > 0 else 0
            else:
                sector_entropy = None
                top_sector = None
                top_sector_weight = 0
            
            # Look-through EV/EBITDA (weighted average of holdings' EV/EBITDA)
            # Need to get EV/EBITDA for each held ticker as of this date
            held_tickers = filer_edges["held_ticker"].unique()
            
            # Get latest EV/EBITDA for held tickers from fundamentals
            lt_ev_ebitda = None
            lt_revenue_growth = None
            lt_roic = None
            
            ev_ebitda_values = []
            ev_ebitda_weights = []
            
            for held in held_tickers:
                held_weight = filer_edges[filer_edges["held_ticker"] == held]["market_value"].values[0] / total_portfolio_value
                
                # Find EV/EBITDA for held ticker around this date
                held_fund = fundamentals[fundamentals["ticker"] == held]
                if len(held_fund) > 0 and "ev_ebitda" in held_fund.columns:
                    # Get closest date <= as_of_date
                    date_fund = held_fund[held_fund["as_of_date"] <= as_of_date] if "as_of_date" in held_fund.columns else held_fund
                    if len(date_fund) > 0:
                        latest = date_fund.sort_values("as_of_date" if "as_of_date" in date_fund.columns else date_fund.columns[0]).iloc[-1]
                        if "ev_ebitda" in latest and pd.notna(latest["ev_ebitda"]):
                            ev_ebitda_values.append(latest["ev_ebitda"])
                            ev_ebitda_weights.append(held_weight)
                        
                        # Also get revenue growth and ROIC
                        if "revenue_growth" in latest and pd.notna(latest["revenue_growth"]):
                            pass  # We'll compute weighted average later
            
            if ev_ebitda_values and ev_ebitda_weights:
                lt_ev_ebitda = np.average(ev_ebitda_values, weights=ev_ebitda_weights)
            
            results.append({
                "filer_ticker": filer,
                "as_of_date": as_of_date,
                "total_holdings_value": total_portfolio_value,
                "n_holdings": n_holdings,
                "hhi": hhi,
                "top5_concentration": top5_weight,
                "sector_entropy": sector_entropy,
                "top_sector": top_sector,
                "top_sector_weight": top_sector_weight,
                "lookthrough_ev_ebitda": lt_ev_ebitda,
            })
    
    metrics_df = pd.DataFrame(results)
    metrics_df.to_parquet(metrics_output, index=False)
    print(f"Saved network metrics to {metrics_output}: {len(metrics_df)} rows")
    
    # Also create look-through fundamentals panel
    lookthrough_results = []
    
    for as_of_date in edges["as_of_date"].unique():
        date_edges = edges[edges["as_of_date"] == as_of_date].copy()
        
        for filer in date_edges["filer_ticker"].unique():
            filer_edges = date_edges[date_edges["filer_ticker"] == filer]
            total_portfolio_value = filer_edges["market_value"].sum()
            
            if total_portfolio_value == 0:
                continue
            
            # For each fundamental metric, compute weighted average
            held_tickers = filer_edges["held_ticker"].unique()
            
            row = {"filer_ticker": filer, "as_of_date": as_of_date}
            
            # Get weights
            weights_dict = dict(zip(filer_edges["held_ticker"], filer_edges["market_value"] / total_portfolio_value))
            
            # Metrics to compute look-through for (using correct column names from fundamentals)
            metrics = ["ev_ebitda", "roic", "fcf_margin", "debt_to_equity", "interest_coverage"]
            
            for metric in metrics:
                values = []
                weights = []
                for held in held_tickers:
                    held_fund = fundamentals[fundamentals["ticker"] == held]
                    if len(held_fund) > 0 and metric in held_fund.columns:
                        date_fund = held_fund[held_fund["as_of_date"] <= as_of_date] if "as_of_date" in held_fund.columns else held_fund
                        if len(date_fund) > 0:
                            latest = date_fund.sort_values("as_of_date" if "as_of_date" in date_fund.columns else date_fund.columns[0]).iloc[-1]
                            if metric in latest and pd.notna(latest[metric]):
                                values.append(latest[metric])
                                weights.append(weights_dict.get(held, 0))
                
                if values and weights:
                    row[f"lt_{metric}"] = np.average(values, weights=weights)
                else:
                    row[f"lt_{metric}"] = None
            
            lookthrough_results.append(row)
    
    lookthrough_df = pd.DataFrame(lookthrough_results)
    lookthrough_df.to_parquet(lookthrough_output, index=False)
    print(f"Saved look-through fundamentals to {lookthrough_output}: {len(lookthrough_df)} rows")
    
    return metrics_df, lookthrough_df


def main():
    holdings_path = "holdings_panel_enriched.parquet"
    prices_path = "daily_prices.parquet"
    fundamentals_path = "fundamentals.parquet"
    edges_output = "ownership_network_edges.parquet"
    nodes_output = "ownership_network_nodes.parquet"
    metrics_output = "ownership_network_metrics.parquet"
    lookthrough_output = "lookthrough_fundamentals.parquet"
    
    if not Path(holdings_path).exists():
        print(f"Holdings panel not found: {holdings_path}")
        return
    
    # Step 1: Build network
    print("Building ownership network...")
    edges, nodes = build_ownership_network(
        holdings_path, prices_path, fundamentals_path,
        edges_output, nodes_output
    )
    
    # Step 2: Calculate metrics
    print("\nCalculating network metrics and look-through fundamentals...")
    metrics, lookthrough = calculate_network_metrics(
        edges_output, nodes_output,
        metrics_output, lookthrough_output
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()