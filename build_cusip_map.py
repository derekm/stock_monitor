#!/usr/bin/env python3
"""
Build CUSIP to ticker mapping from various sources.
"""

import json
import pandas as pd
from pathlib import Path

def build_cusip_ticker_map():
    """Build CUSIP -> ticker mapping from available data."""
    
    # Load CIK map for ticker reference
    with open("cik_ticker_map.json") as f:
        ticker_to_cik = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}
    
    # Known CUSIP mappings (from 13F data and common knowledge)
    cusip_to_ticker = {
        # Major holdings from BRK 13F
        "037833100": "AAPL",      # Apple Inc
        "025816109": "AXP",       # American Express Co
        "191216100": "KO",        # Coca Cola Co
        "02079K305": "GOOGL",     # Alphabet Inc Class A
        "02079K107": "GOOGL",     # Alphabet Inc Class C
        "060505104": "BAC",       # Bank of America Corp
        "166764100": "CVX",       # Chevron Corp
        "674599105": "OXY",       # Occidental Petroleum Corp
        "H1467J104": "CB",        # Chubb Limited
        "615369105": "MCO",       # Moody's Corp
        "500754106": "KHC",       # Kraft Heinz Co
        "23918K108": "DVA",       # DaVita Inc
        "247361702": "DAL",       # Delta Air Lines Inc
        "829933100": "SIRI",      # Sirius XM Holdings Inc
        "92343E102": "VRSN",      # VeriSign Inc
        "501044101": "KR",        # Kroger Co
        "02005N100": "ALLY",      # Ally Financial Inc
        "526057104": "LEN",       # Lennar Corp Class A
        "530909308": "LLYVK",     # Liberty Live Holdings Inc Series C
        "650111107": "NYT",       # New York Times Co Class A
        "14040H105": "COF",       # Capital One Financial Corp
        "530909100": "LLYVA",     # Liberty Live Holdings Inc Series A
        "546347105": "LPX",       # Louisiana Pacific Corp
        "670346105": "NUE",       # Nucor Corp
        "55616P104": "M",         # Macy's Inc
        "62944T105": "NVR",       # NVR Inc
        "526057302": "LEN.B",     # Lennar Corp Class B
        "472319109": "JEF",       # Jefferies Financial Group Inc
        "23331A109": "DHI",       # D.R. Horton Inc
        
        # Additional common CUSIPs
        "459200101": "IBM",       # International Business Machines
        "594918104": "MSFT",      # Microsoft Corp
        "023135106": "AMZN",      # Amazon.com Inc
        "30303M102": "META",      # Meta Platforms Inc
        "67066G104": "NVDA",      # NVIDIA Corp
        "88160R101": "TSLA",      # Tesla Inc
        "91324P102": "UNH",       # UnitedHealth Group Inc
        "92826C839": "V",         # Visa Inc
        "57636Q104": "MA",        # Mastercard Inc
        "437076102": "HD",        # Home Depot Inc
        "742718109": "PG",        # Procter & Gamble Co
        "713448108": "PEP",       # PepsiCo Inc
        "22160K105": "COST",      # Costco Wholesale Corp
        "254687106": "DIS",       # Walt Disney Co
        "92343V104": "VZ",        # Verizon Communications
        "00724F101": "ADBE",      # Adobe Inc
        "125523100": "CMCSA",     # Comcast Corp
        "67103H107": "ORCL",      # Oracle Corp
        "17275R102": "CSCO",      # Cisco Systems Inc
        "458140100": "INTC",      # Intel Corp
        "007903107": "AMD",       # Advanced Micro Devices
        "747525103": "QCOM",      # Qualcomm Inc
        "882508104": "TXN",       # Texas Instruments Inc
        "11135F101": "AVGO",      # Broadcom Inc
        "478160104": "JNJ",       # Johnson & Johnson
        "717081103": "PFE",       # Pfizer Inc
        "58933Y105": "MRK",       # Merck & Co Inc
        "002824100": "ABT",       # Abbott Laboratories
        "375558103": "GILD",      # Gilead Sciences Inc
        "02079K305": "GOOGL",     # Alphabet Inc Class A
        "02079K107": "GOOGL",     # Alphabet Inc Class C
    }
    
    # Save mapping
    output_path = "cusip_ticker_map.json"
    with open(output_path, "w") as f:
        json.dump(cusip_to_ticker, f, indent=2)
    
    print(f"Saved CUSIP->ticker map with {len(cusip_to_ticker)} entries to {output_path}")
    return cusip_to_ticker


if __name__ == "__main__":
    build_cusip_ticker_map()