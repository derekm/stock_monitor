#!/usr/bin/env python3
"""
bayer_ir_parser.py — Parse Bayer IR Excel financial reports.

Parses the structured Excel files Bayer publishes at:
- Half-year reports: https://reports.bayer.com/half-year-financial-report-q2-2026/en/_assets/downloads/
- Annual reports: https://reports.bayer.com/annual-report-2025/en/_assets/downloads/

Usage:
    python bayer_ir_parser.py --file entire-bayer-ir226.xlsx --ticker BAYRY
    python bayer_ir_parser.py --file entire-bayer-ar25.xlsx --ticker BAYRY
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class BayerIRParser:
    """Parse Bayer IR Excel workbooks."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.xls = pd.ExcelFile(filepath)
        self.sheets = {}
        for name in self.xls.sheet_names:
            df = pd.read_excel(self.xls, sheet_name=name, header=None)
            self.sheets[name] = df

        # Detect report type
        self.is_annual_report = any('annual' in name.lower() or 'ar25' in name.lower() or 'ar24' in name.lower() 
                                     for name in [str(filepath)])
        self.is_half_year_report = any('half-year' in name.lower() or 'ir226' in name.lower() or 'ir225' in name.lower()
                                        for name in [str(filepath)])

    def parse_key_data(self) -> dict:
        """Parse the main key data sheet (ovw-key-figures or ovw-five-year-summary)."""
        # Find the key data sheet
        key_sheet = None
        for name in self.sheets:
            if 'key' in name.lower() or 'ovw' in name.lower() or 'five-year' in name.lower() or 'summary' in name.lower():
                key_sheet = self.sheets[name]
                break

        if key_sheet is None:
            return {}

        df = key_sheet
        data = {}

        # Find header row (contains '€ million' or year numbers)
        header_row = None
        for i, row in df.iterrows():
            vals = [str(v) for v in row if pd.notna(v)]
            # Match '€ million', '€\xa0million', or just '€' in header
            if any('€ million' in v or '€' in v for v in vals) or any(v in ['2021', '2022', '2023', '2024', '2025', '2026'] for v in vals):
                header_row = i
                break

        if header_row is None:
            return {}

        # Determine format
        header_vals = df.iloc[header_row].tolist()
        # Check if it's annual format (years as columns) or half-year format (Q2/H1 columns) or quarterly (Q1 2025, Q1 2026)
        is_annual_format = False
        is_quarterly_format = False
        for v in header_vals:
            if pd.notna(v):
                v_str = str(v)
                # Check for year in string (e.g., 'Dec. 31, 2025' or '2025' or 'Q1 2025')
                try:
                    year = int(float(v_str))
                    if 2020 <= year <= 2030:
                        is_annual_format = True
                        break
                except (ValueError, TypeError):
                    # Try to extract year from string like 'Dec. 31, 2025' or 'Q1 2025'
                    import re
                    year_match = re.search(r'\b(20\d{2})\b', v_str)
                    if year_match:
                        year = int(year_match.group(1))
                        if 2020 <= year <= 2030:
                            # Check if it's quarterly format
                            if 'Q1' in v_str or 'Q2' in v_str or 'Q3' in v_str or 'Q4' in v_str:
                                is_quarterly_format = True
                            else:
                                is_annual_format = True
                            break
                    continue

        if is_annual_format:
            # Annual format: columns are years
            year_cols = {}
            for j, v in enumerate(header_vals):
                if pd.notna(v):
                    v_str = str(v)
                    try:
                        year = int(float(v_str))
                        if 2020 <= year <= 2030:
                            year_cols[j] = year
                    except (ValueError, TypeError):
                        # Try to extract year
                        import re
                        year_match = re.search(r'\b(20\d{2})\b', v_str)
                        if year_match:
                            year = int(year_match.group(1))
                            if 2020 <= year <= 2030:
                                year_cols[j] = year
                        continue

            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                vals = {}
                for j, year in year_cols.items():
                    if j < len(row) and pd.notna(row[j]):
                        try:
                            vals[f'FY_{year}'] = float(row[j])
                        except (ValueError, TypeError):
                            continue

                if vals:
                    data[label] = vals
        elif is_quarterly_format:
            # Quarterly format: Q1 2025, Q1 2026
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                try:
                    q1_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                    q1_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None

                    data[label] = {
                        'Q1_2025': float(q1_2025) if q1_2025 is not None else None,
                        'Q1_2026': float(q1_2026) if q1_2026 is not None else None,
                    }
                except (ValueError, TypeError):
                    continue
        else:
            # Half-year format: Q2 2025, Q2 2026, H1 2025, H1 2026
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                try:
                    q2_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                    q2_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None
                    h1_2025 = row[5] if len(row) > 5 and pd.notna(row[5]) else None
                    h1_2026 = row[6] if len(row) > 6 and pd.notna(row[6]) else None

                    data[label] = {
                        'Q2_2025': float(q2_2025) if q2_2025 is not None else None,
                        'Q2_2026': float(q2_2026) if q2_2026 is not None else None,
                        'H1_2025': float(h1_2025) if h1_2025 is not None else None,
                        'H1_2026': float(h1_2026) if h1_2026 is not None else None,
                    }
                except (ValueError, TypeError):
                    continue

        return data

    def parse_income_statement(self) -> dict:
        """Parse the condensed consolidated income statement."""
        for name in self.sheets:
            if 'income' in name.lower() and ('cfs' in name.lower() or 'cmr' in name.lower()):
                df = self.sheets[name]
                break
        else:
            return {}

        data = {}
        # Find header row
        header_row = None
        for i, row in df.iterrows():
            vals = [str(v) for v in row if pd.notna(v)]
            if any('€ million' in v for v in vals):
                header_row = i
                break

        if header_row is None:
            return {}

        header_vals = df.iloc[header_row].tolist()
        is_annual_format = False
        for v in header_vals:
            if pd.notna(v):
                try:
                    year = int(float(str(v)))
                    if 2020 <= year <= 2030:
                        is_annual_format = True
                        break
                except (ValueError, TypeError):
                    continue

        if is_annual_format:
            # Annual format: columns like ['Note', 2024, 2025]
            year_cols = {}
            for j, v in enumerate(header_vals):
                if pd.notna(v):
                    try:
                        year = int(float(str(v)))
                        if 2020 <= year <= 2030:
                            year_cols[j] = year
                    except (ValueError, TypeError):
                        continue

            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                vals = {}
                for j, year in year_cols.items():
                    if j < len(row) and pd.notna(row[j]):
                        try:
                            vals[f'FY_{year}'] = float(row[j])
                        except (ValueError, TypeError):
                            continue

                if vals:
                    data[label] = vals
        else:
            # Half-year format
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                try:
                    q2_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                    q2_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None
                    h1_2025 = row[3] if len(row) > 3 and pd.notna(row[3]) else None
                    h1_2026 = row[4] if len(row) > 4 and pd.notna(row[4]) else None

                    data[label] = {
                        'Q2_2025': float(q2_2025) if q2_2025 is not None else None,
                        'Q2_2026': float(q2_2026) if q2_2026 is not None else None,
                        'H1_2025': float(h1_2025) if h1_2025 is not None else None,
                        'H1_2026': float(h1_2026) if h1_2026 is not None else None,
                    }
                except (ValueError, TypeError):
                    continue

        return data

    def parse_balance_sheet(self) -> dict:
        """Parse the condensed consolidated statement of financial position."""
        # Try the detailed sheet first (cfs-financial-position), then summary
        balance_df = None
        for name in self.sheets:
            if name.lower() == 'cfs-financial-position' or name == 'cfs-fin-position' or name == 'cfs-financial-position-cd':
                balance_df = self.sheets[name]
                break
        if balance_df is None:
            for name in self.sheets:
                if 'fin-position' in name.lower() or 'financial position' in name.lower():
                    balance_df = self.sheets[name]
                    break

        if balance_df is None:
            return {}

        data = {}
        current_section = ""
        for i, row in balance_df.iterrows():
            vals = row.tolist()
            label = str(vals[0]) if pd.notna(vals[0]) else ''
            if not label or label.strip() == '' or 'Back to index' in label:
                continue

            label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

            # Track section headers
            if label in ['Noncurrent assets', 'Current assets', 'Total assets',
                         'Equity', 'Noncurrent liabilities', 'Current liabilities',
                         'Total equity and liabilities']:
                current_section = label

            # For financial liabilities, prepend section to distinguish
            storage_label = label
            if label == 'Financial liabilities' and current_section:
                storage_label = f"{current_section} - {label}"

            # Check format
            header_row = None
            for j, row_h in balance_df.iterrows():
                hv = [str(v) for v in row_h if pd.notna(v)]
                if any('€ million' in v or '€' in v for v in hv):
                    header_row = j
                    break

            if header_row is not None and i > header_row:
                header_vals = balance_df.iloc[header_row].tolist()
                is_annual_format = False
                is_quarterly_format = False
                for v in header_vals:
                    if pd.notna(v):
                        v_str = str(v)
                        # Check for quarterly markers FIRST
                        if 'Mar' in v_str or 'Sep' in v_str or 'Q1' in v_str or 'Q3' in v_str:
                            is_quarterly_format = True
                            break
                        # Then check for year
                        try:
                            year = int(float(v_str))
                            if 2020 <= year <= 2030:
                                is_annual_format = True
                                break
                        except (ValueError, TypeError):
                            # Try to extract year from string like 'Dec. 31, 2025'
                            import re
                            year_match = re.search(r'\b(20\d{2})\b', v_str)
                            if year_match:
                                year = int(year_match.group(1))
                                if 2020 <= year <= 2030:
                                    is_annual_format = True
                                    break
                            continue
                
                if is_annual_format:
                    # Annual format: ['Note', 'Dec. 31, 2024', 'Dec. 31, 2025']
                    # Data rows have: [label, Note, 2024_value, 2025_value]
                    if len(vals) >= 4:
                        try:
                            # vals[1] is Note (string like '[14]'), vals[2] = 2024, vals[3] = 2025
                            dec_2024 = vals[2] if pd.notna(vals[2]) else None
                            dec_2025 = vals[3] if pd.notna(vals[3]) else None
                            data[storage_label] = {
                                'Dec_31_2024': float(dec_2024) if dec_2024 is not None else None,
                                'Dec_31_2025': float(dec_2025) if dec_2025 is not None else None,
                            }
                        except (ValueError, TypeError):
                            continue
                elif is_quarterly_format:
                    # Quarterly format: ['€ million', 'Mar. 31, 2025', 'Dec. 31, 2025', 'Mar. 31, 2026']
                    if len(vals) >= 4:
                        try:
                            mar_2025 = vals[1] if pd.notna(vals[1]) else None
                            dec_2025 = vals[2] if pd.notna(vals[2]) else None
                            mar_2026 = vals[3] if pd.notna(vals[3]) else None
                            data[storage_label] = {
                                'Mar_31_2025': float(mar_2025) if mar_2025 is not None else None,
                                'Dec_31_2025': float(dec_2025) if dec_2025 is not None else None,
                                'Mar_31_2026': float(mar_2026) if mar_2026 is not None else None,
                            }
                        except (ValueError, TypeError):
                            continue
                else:
                        # Half-year format: ['€ million', 'June 30, 2025', 'Dec. 31, 2025', 'June 30, 2026']
                        if len(vals) >= 4:
                            try:
                                jun_2025 = vals[1] if pd.notna(vals[1]) else None
                                dec_2025 = vals[2] if pd.notna(vals[2]) else None
                                jun_2026 = vals[3] if pd.notna(vals[3]) else None
                                data[storage_label] = {
                                    'June_30_2025': float(jun_2025) if jun_2025 is not None else None,
                                    'Dec_31_2025': float(dec_2025) if dec_2025 is not None else None,
                                    'June_30_2026': float(jun_2026) if jun_2026 is not None else None,
                                }
                            except (ValueError, TypeError):
                                continue

        return data

    def parse_cash_flow(self) -> dict:
        """Parse the condensed consolidated cash flow statement."""
        for name in self.sheets:
            if 'cash-flow' in name.lower() and ('cfs' in name.lower() or 'cmr' in name.lower()):
                df = self.sheets[name]
                break
        else:
            return {}

        data = {}
        header_row = None
        for i, row in df.iterrows():
            vals = [str(v) for v in row if pd.notna(v)]
            if any('€ million' in v for v in vals):
                header_row = i
                break

        if header_row is None:
            return {}

        header_vals = df.iloc[header_row].tolist()
        is_annual_format = False
        is_quarterly_format = False
        for v in header_vals:
            if pd.notna(v):
                v_str = str(v)
                # Check for year in string (e.g., 'Dec. 31, 2025' or '2025' or 'Mar. 31, 2025')
                try:
                    year = int(float(v_str))
                    if 2020 <= year <= 2030:
                        is_annual_format = True
                        break
                except (ValueError, TypeError):
                    # Try to extract year from string like 'Dec. 31, 2025' or 'Mar. 31, 2025'
                    import re
                    year_match = re.search(r'\b(20\d{2})\b', v_str)
                    if year_match:
                        year = int(year_match.group(1))
                        if 2020 <= year <= 2030:
                            # Check if it's quarterly format
                            if 'Mar' in v_str or 'Sep' in v_str or 'Q1' in v_str or 'Q3' in v_str:
                                is_quarterly_format = True
                            else:
                                is_annual_format = True
                            break
                    continue

        if is_annual_format:
            # Annual format
            year_cols = {}
            for j, v in enumerate(header_vals):
                if pd.notna(v):
                    v_str = str(v)
                    try:
                        year = int(float(v_str))
                        if 2020 <= year <= 2030:
                            year_cols[j] = year
                    except (ValueError, TypeError):
                        # Try to extract year
                        import re
                        year_match = re.search(r'\b(20\d{2})\b', v_str)
                        if year_match:
                            year = int(year_match.group(1))
                            if 2020 <= year <= 2030:
                                year_cols[j] = year
                        continue

            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                vals = {}
                for j, year in year_cols.items():
                    if j < len(row) and pd.notna(row[j]):
                        try:
                            vals[f'FY_{year}'] = float(row[j])
                        except (ValueError, TypeError):
                            continue

                if vals:
                    data[label] = vals
        elif is_quarterly_format:
            # Quarterly format: Q1 2025, Q1 2026
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                try:
                    q1_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                    q1_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None

                    data[label] = {
                        'Q1_2025': float(q1_2025) if q1_2025 is not None else None,
                        'Q1_2026': float(q1_2026) if q1_2026 is not None else None,
                    }
                except (ValueError, TypeError):
                    continue
        else:
            # Half-year format
            for i in range(header_row + 1, len(df)):
                row = df.iloc[i].tolist()
                label = str(row[0]) if pd.notna(row[0]) else ''
                if not label or label.strip() == '':
                    continue

                label = label.strip().replace('\n', ' ').replace('\xa0', ' ')

                try:
                    q2_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                    q2_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None
                    h1_2025 = row[3] if len(row) > 3 and pd.notna(row[3]) else None
                    h1_2026 = row[4] if len(row) > 4 and pd.notna(row[4]) else None

                    data[label] = {
                        'Q2_2025': float(q2_2025) if q2_2025 is not None else None,
                        'Q2_2026': float(q2_2026) if q2_2026 is not None else None,
                        'H1_2025': float(h1_2025) if h1_2025 is not None else None,
                        'H1_2026': float(h1_2026) if h1_2026 is not None else None,
                    }
                except (ValueError, TypeError):
                    continue

        return data

    def extract_fundamentals(self) -> dict:
        """Extract all fundamentals as a unified dict."""
        key_data = self.parse_key_data()
        income = self.parse_income_statement()
        balance = self.parse_balance_sheet()
        cash_flow = self.parse_cash_flow()

        # Also parse core earnings per share sheet for shares outstanding
        core_eps = {}
        for name in self.sheets:
            if 'core-earnings' in name.lower() or 'core earnings' in name.lower():
                df = self.sheets[name]
                header_row = None
                for i, row in df.iterrows():
                    vals = [str(v) for v in row if pd.notna(v)]
                    if any('€ million' in v or '€' in v for v in vals):
                        header_row = i
                        break
                if header_row is not None:
                    header_vals = df.iloc[header_row].tolist()
                    is_annual_format = False
                    is_quarterly_format = False
                    for v in header_vals:
                        if pd.notna(v):
                            v_str = str(v)
                            # Check for quarterly markers FIRST
                            if 'Mar' in v_str or 'Sep' in v_str or 'Q1' in v_str or 'Q3' in v_str:
                                is_quarterly_format = True
                                break
                            # Then check for year
                            try:
                                year = int(float(v_str))
                                if 2020 <= year <= 2030:
                                    is_annual_format = True
                                    break
                            except (ValueError, TypeError):
                                # Try to extract year from string like 'Dec. 31, 2025'
                                import re
                                year_match = re.search(r'\b(20\d{2})\b', v_str)
                                if year_match:
                                    year = int(year_match.group(1))
                                    if 2020 <= year <= 2030:
                                        is_annual_format = True
                                        break
                                continue
                    
                    if is_annual_format:
                        year_cols = {}
                        for j, v in enumerate(header_vals):
                            if pd.notna(v):
                                try:
                                    year = int(float(str(v)))
                                    if 2020 <= year <= 2030:
                                        year_cols[j] = year
                                except (ValueError, TypeError):
                                    continue
                        
                        for i in range(header_row + 1, len(df)):
                            row = df.iloc[i].tolist()
                            label = str(row[0]) if pd.notna(row[0]) else ''
                            if not label or label.strip() == '':
                                continue
                            label = label.strip().replace('\n', ' ').replace('\xa0', ' ')
                            vals = {}
                            for j, year in year_cols.items():
                                if j < len(row) and pd.notna(row[j]):
                                    try:
                                        vals[f'FY_{year}'] = float(row[j])
                                    except (ValueError, TypeError):
                                        continue
                            if vals:
                                core_eps[label] = vals
                    elif is_quarterly_format:
                        # Quarterly format: Q1 2025, Q1 2026
                        for i in range(header_row + 1, len(df)):
                            row = df.iloc[i].tolist()
                            label = str(row[0]) if pd.notna(row[0]) else ''
                            if not label or label.strip() == '':
                                continue
                            label = label.strip().replace('\n', ' ').replace('\xa0', ' ')
                            try:
                                q1_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                                q1_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None

                                core_eps[label] = {
                                    'Q1_2025': float(q1_2025) if q1_2025 is not None else None,
                                    'Q1_2026': float(q1_2026) if q1_2026 is not None else None,
                                }
                            except (ValueError, TypeError):
                                continue
                    else:
                        # Half-year format
                        for i in range(header_row + 1, len(df)):
                            row = df.iloc[i].tolist()
                            label = str(row[0]) if pd.notna(row[0]) else ''
                            if not label or label.strip() == '':
                                continue
                            label = label.strip().replace('\n', ' ').replace('\xa0', ' ')
                            try:
                                q2_2025 = row[1] if len(row) > 1 and pd.notna(row[1]) else None
                                q2_2026 = row[2] if len(row) > 2 and pd.notna(row[2]) else None
                                h1_2025 = row[3] if len(row) > 3 and pd.notna(row[3]) else None
                                h1_2026 = row[4] if len(row) > 4 and pd.notna(row[4]) else None

                                core_eps[label] = {
                                    'Q2_2025': float(q2_2025) if q2_2025 is not None else None,
                                    'Q2_2026': float(q2_2026) if q2_2026 is not None else None,
                                    'H1_2025': float(h1_2025) if h1_2025 is not None else None,
                                    'H1_2026': float(h1_2026) if h1_2026 is not None else None,
                                }
                            except (ValueError, TypeError):
                                continue
                break

        # Merge all
        all_data = {**key_data, **income, **balance, **cash_flow, **core_eps}

        return all_data


def build_fundamentals_row(parser: BayerIRParser, ticker: str) -> dict:
    """Build a fundamentals.parquet-compatible row from Bayer IR data."""
    data = parser.extract_fundamentals()

    # Detect format: annual (FY_2024, FY_2025), half-year (H1_2026), or quarterly (Q1_2026)
    has_annual = any('FY_' in str(v) for dv in data.values() if isinstance(dv, dict) for v in dv.keys())
    has_half_year = any('H1_2026' in str(v) for dv in data.values() if isinstance(dv, dict) for v in dv.keys())
    has_quarterly = any('Q1_2026' in str(v) for dv in data.values() if isinstance(dv, dict) for v in dv.keys())

    # Prefer annual > half-year > quarterly when multiple present
    if has_annual:
        period_suffix = 'FY_2025'
        prev_suffix = 'FY_2024'
        as_of_date = date(2025, 12, 31)
        notes = "Bayer IR Excel Annual Report 2025 (full year); EUR millions -> EUR"
    elif has_half_year:
        period_suffix = 'H1_2026'
        prev_suffix = 'H1_2025'
        as_of_date = date(2026, 6, 30)
        notes = "Bayer IR Excel H1 2026 (6-month); TTM approximated as 2x H1; EUR millions -> EUR"
    elif has_quarterly:
        period_suffix = 'Q1_2026'
        prev_suffix = 'Q1_2025'
        as_of_date = date(2026, 3, 31)
        notes = "Bayer IR Excel Q1 2026 (quarter); TTM approximated as 4x Q1; EUR millions -> EUR"
    else:
        period_suffix = 'H1_2026'
        prev_suffix = 'H1_2025'
        as_of_date = date(2026, 6, 30)
        notes = "Bayer IR Excel (unknown format); EUR millions -> EUR"

    def get_val(key_variants: list[str], suffix: str = period_suffix) -> Optional[float]:
        for k in key_variants:
            for dk, dv in data.items():
                if k.lower() in dk.lower():
                    val = dv.get(suffix)
                    if val is not None:
                        return val
        return None

    def get_bs_val(key_variants: list[str]) -> Optional[float]:
        """Get balance sheet value - try different date suffixes."""
        suffixes = ['Mar_31_2026', 'Dec_31_2025', 'Dec_31_2024', 'June_30_2026', 'Mar_31_2025', 'Dec_31_2025']
        for suffix in suffixes:
            for k in key_variants:
                for dk, dv in data.items():
                    if k.lower() in dk.lower():
                        val = dv.get(suffix)
                        if val is not None:
                            return val
        return None

    # Extract key metrics (values in EUR millions)
    revenue = get_val(['Sales', 'Net sales'])
    net_income = get_val(['Net income', 'of which attributable to Bayer', 'attributable to Bayer AG stockholders'])
    ebitda = get_val(['EBITDA before special items', 'EBITDA1'])
    ebit = get_val(['EBIT before special items', 'EBIT1', 'Core EBIT'])
    ocf = get_val(['Net cash provided by (used in) operating activities'])
    fcf = get_val(['Free cash flow'])
    capex = get_val(['Cash flow-relevant capital expenditures', 'Cash outflows for additions', 'Capital expenditure', 'Cash outflows for additions to property'])
    interest_expense = get_val(['Net interest expense', 'Financial result'])
    rd = get_val(['Research and development expenses'])

    total_assets = get_bs_val(['Total assets'])
    total_equity = get_bs_val(['Equity attributable to Bayer', 'Equity attributable to Bayer AG stockholders', 'Equity'])
    cash = get_bs_val(['Cash and cash equivalents'])

    # Total debt = non-current + current financial liabilities
    total_debt = None
    for dk, dv in data.items():
        if 'financial liabilities' in dk.lower():
            for suffix in ['Dec_31_2025', 'Dec_31_2024', 'June_30_2026']:
                val = dv.get(suffix)
                if val is not None:
                    if total_debt is None:
                        total_debt = 0.0
                    total_debt += val

    goodwill = get_bs_val(['Goodwill'])
    intangible_assets = get_bs_val(['Other intangible assets', 'intangible assets'])
    inventory = get_bs_val(['Inventories'])
    receivables = get_bs_val(['Trade accounts receivable'])
    payables = get_bs_val(['Trade accounts payable'])

    # Per share
    shares = None
    for dk, dv in data.items():
        if 'weighted average number of shares' in dk.lower():
            for suffix in ['FY_2025', 'FY_2024', 'H1_2026', 'H1_2025', 'Q1_2026', 'Q1_2025']:
                val = dv.get(suffix)
                if val is not None:
                    shares = val * 1_000_000  # already in millions
                    break

    # Convert from EUR millions to EUR (multiply by 1e6)
    scale = 1_000_000

    fiscal_year_end_month = 12  # Bayer fiscal year ends Dec 31

    # For annual report, use actual full year values
    # For half-year, TTM = H1 * 2
    # For quarterly, TTM = Q1 * 4
    if has_annual:
        ttm_multiplier = 1.0
    elif has_quarterly:
        ttm_multiplier = 4.0
    else:
        ttm_multiplier = 2.0

    def to_ttm(val: Optional[float]) -> Optional[float]:
        return val * scale * ttm_multiplier if val is not None else None

    def to_abs(val: Optional[float]) -> Optional[float]:
        return val * scale if val is not None else None

    # FCF margin on TTM basis
    rev_ttm = to_ttm(revenue)
    fcf_ttm = to_ttm(fcf)
    fcf_margin = (fcf_ttm / rev_ttm) if (rev_ttm and rev_ttm != 0 and fcf_ttm is not None) else None

    return {
        # Primary keys
        "ticker": ticker,
        "fiscal_year_end": float(fiscal_year_end_month),

        # Income statement
        "revenue_ttm": to_ttm(revenue),
        "revenue_yoy": None,
        "revenue_qoq": None,
        "net_income_ttm": to_ttm(net_income),
        "ebitda_ttm": to_ttm(ebitda),
        "operating_income_ttm": to_ttm(ebit),
        "interest_expense_ttm": to_ttm(interest_expense) if interest_expense else None,

        # Cash flow
        "operating_cash_flow_ttm": to_ttm(ocf),
        "capex_ttm": to_ttm(capex),
        "free_cash_flow": to_ttm(fcf),
        "fcf_margin": fcf_margin,

        # Balance sheet (point-in-time)
        "total_assets": to_abs(total_assets),
        "total_equity": to_abs(total_equity),
        "shareholders_equity": to_abs(total_equity),
        "total_debt": to_abs(total_debt),
        "cash_and_equivalents": to_abs(cash),
        "goodwill": to_abs(goodwill),
        "intangible_assets": to_abs(intangible_assets),
        "inventory": to_abs(inventory),
        "receivables": to_abs(receivables),
        "payables": to_abs(payables),

        # Per share
        "shares_outstanding": shares,

        # Metadata
        "as_of_date": as_of_date,
        "last_updated": pd.Timestamp.now(),
        "source": "bayer_ir_excel",
        "source_rank": 100,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse Bayer IR Excel and output fundamentals")
    parser.add_argument("--file", required=True, help="Path to Bayer IR Excel file")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., BAYRY)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--merge", action="store_true",
                        help="Upsert into fundamentals.parquet (source=bayer_ir_excel, rank 100)")
    args = parser.parse_args()

    bp = BayerIRParser(Path(args.file))
    row = build_fundamentals_row(bp, args.ticker)

    if args.merge:
        fund_path = Path(__file__).parent / "fundamentals.parquet"
        fund = pd.read_parquet(fund_path)
        fund["ticker"] = fund["ticker"].astype(str).str.upper()
        row["ticker"] = str(row["ticker"]).upper()
        row["as_of_date"] = pd.to_datetime(row["as_of_date"]).date()
        if "as_of_date" in fund.columns:
            fund["as_of_date"] = pd.to_datetime(fund["as_of_date"], errors="coerce").dt.date
        mask = (fund["ticker"] == row["ticker"]) & (fund["as_of_date"] == row["as_of_date"])
        if mask.any():
            src = fund.loc[mask, "source"].astype(str) if "source" in fund.columns else pd.Series([""])
            rank = fund.loc[mask, "source_rank"] if "source_rank" in fund.columns else pd.Series([999])
            # keep higher-rank (lower number is better? plan said Bayer 100, companyfacts 110)
            # SOURCE: lower rank number = higher priority in this repo (100 beats 110)
            if rank.min() < 100:
                print(f"skip merge: existing rank {rank.min()} beats 100")
            else:
                fund = fund.loc[~mask]
                fund = pd.concat([fund, pd.DataFrame([row])], ignore_index=True)
                fund.to_parquet(fund_path, index=False)
                print(f"replaced {row['ticker']} {row['as_of_date']}")
        else:
            fund = pd.concat([fund, pd.DataFrame([row])], ignore_index=True)
            fund.to_parquet(fund_path, index=False)
            print(f"appended {row['ticker']} {row['as_of_date']}")
        ni = row.get("net_income_ttm")
        print(f"  NI_ttm={ni} FCF={row.get('free_cash_flow')} equity={row.get('shareholders_equity')}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(row, f, indent=2, default=str)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(row, indent=2, default=str))


if __name__ == "__main__":
    main()