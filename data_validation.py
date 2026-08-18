#!/usr/bin/env python3
"""data_validation.py — Validation guards to prevent data errors."""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean price data."""
    original_len = len(df)
    
    # Remove zero/negative/infinite/NaN prices
    df = df[df['adj_close'] > 0]
    df = df[np.isfinite(df['adj_close'])]
    df = df[df['adj_close'].notna()]
    
    # Remove future dates
    today = datetime.now().date()
    df = df[pd.to_datetime(df['date']).dt.date <= today]
    
    if len(df) != original_len:
        print(f"  Price validation: {original_len} -> {len(df)} rows")
    
    return df

def validate_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean fundamentals data."""
    original_len = len(df)
    
    # Remove future dates
    today = datetime.now().date()
    df = df[pd.to_datetime(df['as_of_date']).dt.date <= today]
    
    # Set infinite values to NaN
    for col in ['net_income_quarterly', 'revenue_quarterly', 'free_cash_flow', 'operating_income_quarterly']:
        if col in df.columns:
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                df.loc[inf_mask, col] = np.nan
    
    if len(df) != original_len:
        print(f"  Fundamentals validation: {original_len} -> {len(df)} rows")
    
    return df
