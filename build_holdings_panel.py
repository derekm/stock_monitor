#!/usr/bin/env python3
"""
build_holdings_panel.py — Convert detailed XBRL holdings into a clean holdings panel.

Input: detailed_holdings.parquet (from extract_detailed_holdings.py)
Output: holdings_panel.parquet with columns:
  filer_ticker, as_of_date, held_ticker, held_cik, shares, market_value, concept, source_filing
"""

import json
import re
from pathlib import Path

import pandas as pd

def load_cik_ticker_map():
    """Load bidirectional CIK <-> ticker mapping"""
    with open("cik_ticker_map.json") as f:
        ticker_to_cik = {k.upper(): str(v).zfill(10) for k, v in json.load(f).items()}
    cik_to_ticker = {v: k for k, v in ticker_to_cik.items()}
    return ticker_to_cik, cik_to_ticker

def load_cusip_ticker_map():
    """Load CUSIP -> ticker mapping"""
    with open("cusip_ticker_map.json") as f:
        return json.load(f)

def parse_member_to_ticker(member: str, cik_to_ticker: dict) -> tuple:
    """
    Parse XBRL dimension member to extract ticker/CIK.
    Examples:
      - "brka:TheKraftHeinzCompanyMember" -> look up "Kraft Heinz" ticker
      - "us-gaap:USTreasurySecuritiesMember" -> generic, no specific issuer
      - "dei:LegalEntityAxis": "brka:AmericanExpressCompanyMember" -> American Express
    """
    # Remove namespace prefix
    if ":" in member:
        member = member.split(":", 1)[1]
    
    # Remove "Member" suffix
    if member.endswith("Member"):
        member = member[:-6]
    
    # Handle common patterns
    # brka:TheKraftHeinzCompanyMember -> TheKraftHeinzCompany
    # brka:OccidentalPetroleumCorporationMember -> OccidentalPetroleumCorporation
    # brka:AmericanExpressCompanyMember -> AmericanExpressCompany
    
    # Try to match known company names to tickers
    # This is a simplified mapping - in practice would need a more comprehensive lookup
    name_to_ticker = {
        "TheKraftHeinzCompany": "KHC",
        "KraftHeinzCompany": "KHC",
        "OccidentalPetroleumCorporation": "OXY",
        "AmericanExpressCompany": "AXP",
        "BerkadiaCommercialMortgage": None,  # Private
        "JefferiesFinancialGroupInc": "JEF",
        "BankOfAmericaCorporation": "BAC",
        "AppleInc": "AAPL",
        "CocaColaCompany": "KO",
        "ChevronCorporation": "CVX",
        "Moody'sCorporation": "MCO",
        "MoodySCorporation": "MCO",
        "USGovernmentCorporationsAndAgencies": None,  # Government
        "USTreasuryAndGovernment": None,
        "USTreasurySecurities": None,
        "MortgageBackedSecurities": None,
        "CorporateDebtSecurities": None,
        "MoneyMarketFunds": None,
        "MutualFund": None,
        "AssetBackedSecurities": None,
        "USGovernmentAgenciesDebtSecurities": None,
        "ForeignGovernmentDebtSecurities": None,
        "BankTimeDeposits": None,
        "CommercialPaper": None,
        "CertificateOfDeposit": None,
        "EquitySecurities": None,
        "PreferredStock": None,
        "CommonStock": None,
        "CommonClassC": None,
        # Additional from discovered data
        "GlobalEOnlineLtd": "GLBE",
        "Altera": None,  # Acquired by Intel
        "GrabEquitySecurities": "GRAB",
        "AuroraEquitySecurities": "AUR",
        "JasperNewCoLimitedNewCo": None,
        "RobloxChinaHoldingCorp": None,
        "IndiaJointVenture": None,
        "SpaceX": None,  # Private
        "NewlyFormedCompany": None,
        "DeliveryHero": None,
        "OtherEquityMethodInvestments": None,
        "AEMember": None,
        "TheKraftHeinzCompanyAndOccidentalPetroleumCorporation": None,
        "Fubo": "FUBO",
        "AffirmHoldingsInc": "AFRM",
        "OpenAIGlobalLlc": None,  # Private
        "DataCenterCampusInLouisiana": None,
        "BerkadiaCommercialMortgage": None,
        "OtherEquitySecurities": None,
        "KlaviyoInc": "KVYO",
        "PrivateInvestment": None,
        "Banamex": None,  # Part of Citi
        "DataCenterCampusInTexas": None,
        "DidiEquitySecurities": None,
        "TikTokUSDSJointVentureLLC": None,
        "CCEP": "CCEP",
        "EquityMethodInvesteeInLatinAmerica": None,
        "JointVentureInLatinAmerica": None,
        "VISA": "V",
        "SingleInvestee": None,
        "SilverLakePartners": None,
        "DepartmentOfCommerce": None,
        "ConsolidatedSecuritization": None,
        "EquityInterestHeldByFuboShareholders": None,
        "VariableInterestEntityPrimaryBeneficiary": None,
        "RiskRetentionFinancingFacility": None,
        "EquitySecuritiesWithoutReadilyDeterminableFairValue": None,
        "CommonStock": None,
        "CommonClassB": None,
        "CommonClassC": None,
        "CommonClassB2": None,
        "CommonClassB3": None,
        "NonMarketableEquitySecurities": None,
        "MarketableEquitySecurities": None,
        "SecuredDebt": None,
        "RelatedParty": None,
        "ScenarioForecast": None,
        "SubsequentEvent": None,
    }
    
    # Try direct mapping first
    if member in name_to_ticker:
        return name_to_ticker[member], None
    
    # Try to find partial match
    for name, ticker in name_to_ticker.items():
        if name in member or member in name:
            return ticker, None
    
    # Try to look up by CIK if it looks like a CIK
    if member.isdigit() and len(member) >= 5:
        cik = member.zfill(10)
        if cik in cik_to_ticker:
            return cik_to_ticker[cik], cik
    
    return None, None

def extract_issuer_from_dimensions(dimensions_json: str, cik_to_ticker: dict) -> tuple:
    """Extract issuer ticker/CIK from dimensions JSON"""
    try:
        dims = json.loads(dimensions_json)
    except (json.JSONDecodeError, TypeError):
        return None, None
    
    # Priority order for issuer identification
    issuer_dimensions = [
        "us-gaap:EquityMethodInvestmentNonconsolidatedInvesteeAxis",
        "srt:ScheduleOfEquityMethodInvestmentEquityMethodInvesteeNameAxis",
        "dei:LegalEntityAxis",
        "srt:CounterpartyNameAxis",
        "us-gaap:MajorEquityInvestmentsAxis",
        "us-gaap:EquityMethodInvestmentsAxis",
        "us-gaap:InvestmentsByTypeAxis",
    ]
    
    for dim in issuer_dimensions:
        if dim in dims:
            member = dims[dim]
            ticker, cik = parse_member_to_ticker(member, cik_to_ticker)
            if ticker:
                return ticker, cik
    
    # Check for 13F-HR specific fields in dimensions
    if 'nameOfIssuer' in dims and dims['nameOfIssuer']:
        # Try to map the issuer name to ticker
        issuer_name = dims['nameOfIssuer']
        # Try direct lookup from a mapping (would need to be built)
        # For now, try to match using CUSIP if available
        if 'cusip' in dims and dims['cusip']:
            # We'd need a CUSIP->ticker mapping
            pass
    
    return None, None

def process_holdings(detailed_path: str, output_path: str):
    """Convert detailed holdings to clean panel"""
    df = pd.read_parquet(detailed_path)
    
    # Load CIK mapping
    ticker_to_cik, cik_to_ticker = load_cik_ticker_map()
    # Load CUSIP mapping
    cusip_to_ticker = load_cusip_ticker_map()
    
    # Handle 13F-HR rows which have issuer info in dimensions JSON
    if 'source_form_type' in df.columns:
        mask_13f = df['source_form_type'] == '13F-HR'
        if mask_13f.any():
            print(f"Extracting issuer information from {mask_13f.sum()} 13F-HR rows...")
            # Parse dimensions JSON for 13F-HR rows to extract CUSIP and name
            def extract_13f_issuer(dimensions_json):
                try:
                    dims = json.loads(dimensions_json)
                except:
                    return None, None
                # Try CUSIP first
                if 'cusip' in dims and dims['cusip']:
                    cusip = dims['cusip'].strip()
                    if cusip in cusip_to_ticker:
                        return cusip_to_ticker[cusip], None
                # Try nameOfIssuer
                if 'nameOfIssuer' in dims and dims['nameOfIssuer']:
                    name = dims['nameOfIssuer'].strip()
                    # Could do fuzzy matching here
                    pass
                return None, None
            
            issuers_13f = df.loc[mask_13f, 'dimensions'].apply(extract_13f_issuer)
            df.loc[mask_13f, 'held_ticker'] = [x[0] for x in issuers_13f]
            df.loc[mask_13f, 'held_cik'] = [x[1] for x in issuers_13f]
            print(f"  Identified issuers from 13F-HR: {df.loc[mask_13f, 'held_ticker'].notna().sum()}")
    
    # Extract issuer information from ALL rows (including XBRL dimensions)
    print("Extracting issuer information from XBRL dimensions...")
    issuers = df["dimensions"].apply(lambda x: extract_issuer_from_dimensions(x, cik_to_ticker))
    df["held_ticker"] = df["held_ticker"].combine_first(pd.Series([x[0] for x in issuers], index=df.index))
    df["held_cik"] = df["held_cik"].combine_first(pd.Series([x[1] for x in issuers], index=df.index))
    
    print(f"Rows with identified issuers from XBRL dimensions: {df['held_ticker'].notna().sum()} / {len(df)}")
    
    # Now filter for concepts that represent actual holdings (not gains/losses, not percentages)
    holding_concepts = [
        "EquityMethodInvestments",
        "EquityMethodInvestmentsFairValueDisclosure",
        "AvailableForSaleSecuritiesDebtSecurities",
        "AvailableForSaleSecuritiesDebtMaturities",
        "HeldToMaturitySecurities",
        "TradingSecurities",
        "ShortTermInvestments",
        "LongTermInvestments",
        "Investments",
        "MarketableSecurities",
        "MarketableSecuritiesCurrent",
        "MarketableSecuritiesNoncurrent",
        "InvestmentOwnedBalanceShares",
        "InvestmentOwnedPercentOfCommonSharesOutstanding",
        "EquitySecuritiesFvNi",
        "EquitySecuritiesFvNiCurrentAndNoncurrent",
        "DebtSecuritiesAvailableForSale",
        "DebtSecuritiesHeldToMaturity",
        "OtherInvestments",
        "OtherLongTermInvestments",
        "InvestmentsExcludingTrading",
        "CashCashEquivalentsAndShortTermInvestments",
        "EquitySecuritiesFvNiAndWithoutReadilyDeterminableFairValue",
        "EquitySecuritiesFvNiCost",
        "EquitySecuritiesFvNiNoncurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterest",
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent",
        "DebtSecuritiesAvailableForSaleExcludingAccruedInterestNoncurrent",
        "TradingSecuritiesDebt",
        "InvestmentsFairValueDisclosure",
        "RestrictedCashEquivalentsAndShortTermInvestmentsFairValueDisclosure",
        "UnrestrictedCashEquivalentsAndShortTermInvestmentsFairValueDisclosure",
    ]
    
    # Filter rows
    mask = df["concept"].str.contains("|".join(holding_concepts), case=False, na=False)
    # Exclude gain/loss/percentage concepts
    exclude_patterns = [
        "GainLoss", "UnrealizedGain", "UnrealizedLoss", "AccumulatedGain", "AccumulatedLoss",
        "Percentage", "WeightedAverageYield", "Maturity", "MeasurementInput", "AmortizedCost",
        "Amortization", "Accretion", "RealizedGain", "RealizedLoss", "DividendsOrDistributions",
        "OwnershipPercentage", "DifferenceBetween", "OtherThanTemporaryImpairment",
        "IncomeLossFromEquityMethod", "ExcessOfCarrying", "ExcessOfFairValue",
        "ProceedsFrom", "PaymentsToAcquire", "AccumulatedUnrealized", "AdjustmentNetOfTax",
        "ReclassificationAdjustment", "CashAndSecuritiesPledged", "FairValueOfSecuritiesReceived",
        "SecuritiesBorrowed", "SecuritiesLoaned", "SecuritiesPurchasedUnder", "SecuritiesSoldUnder",
        "FederalFunds", "InterestIncome", "InterestExpense", "InvestmentIncome", "InvestmentBanking",
        "PrincipalInvestment", "IncreaseDecreaseIn", "RepaymentsOf", "ProceedsFromIssuance",
        "SecuritiesReserve", "Collateral", "OciDebtSecurities", "ParticipatingSecurities",
        "RestrictedInvestments", "AlternativeInvestment", "QualifiedAffordableHousing",
        "CashEquivalentsAndMarketableSecuritiesCost", "CashEquivalentsAndMarketableSecurities",
        "DebtSecuritiesAndLoansReceivableFairValueDisclosure", "DebtSecuritiesAndLoansReceivableMeasurementInput",
        "CashCashEquivalentsAndMarketableSecurities", "CashCashEquivalentsAndMarketableSecuritiesCost",
        "CashEquivalentsAndMarketableSecuritiesAccumulatedGrossUnrealizedGainBeforeTax",
        "CashEquivalentsAndMarketableSecuritiesAccumulatedGrossUnrealizedLossBeforeTax",
        "DebtSecuritiesAvailableForSaleMaturityAllocatedAndSingleMaturityDateRollingAfter",
        "DebtSecuritiesAvailableForSaleMaturityAllocatedAndSingleMaturityDateRollingAfterOneThroughFive",
        "DebtSecuritiesAvailableForSaleMaturityAllocatedAndSingleMaturityDateRollingAfterTen",
        "EquitySecuritiesFVNIAccumulatedGrossUnrealizedGainBeforeTax",
        "EquitySecuritiesFVNIAccumulatedGrossUnrealizedLossBeforeTax",
        "EquityMethodInvestmentSummarizedFinancialInformationNetIncomeLoss",
        "EquityMethodInvestmentDifferenceBetweenCarryingAmountAndUnderlyingEquity",
        "EquityMethodInvestmentOtherThanTemporaryImpairment",
        "GainLossOnInvestmentsAndGainLossOnDerivativeInstrumentsTax",
        "IncomeLossFromEquityMethodInvestmentsTax",
        "InvestmentIncomeInterestDividendAndOther",
        "OtherComprehensiveIncomeLossAvailableForSaleSecuritiesAdjustmentNetOfTax",
        "OtherComprehensiveIncomeLossReclassificationAdjustmentFromAOCIForSaleOfSecuritiesNetOfTax",
        "OtherComprehensiveIncomeUnrealizedHoldingGainLossOnSecuritiesArisingDuringPeriodNetOfTax",
        "PaymentsToAcquireAvailableForSaleSecuritiesDebt",
        "ProceedsFromMaturitiesPrepaymentsAndCallsOfAvailableForSaleSecurities",
        "ProceedsFromSaleOfAvailableForSaleSecuritiesDebt",
        "TreasuryStockValueAcquiredCostMethod",
        "OtherAssetsCurrent", "OtherAssetsNoncurrent", "OtherAssets",
    ]
    exclude_mask = df["concept"].str.contains("|".join(exclude_patterns), case=False, na=False)
    df = df[mask & ~exclude_mask].copy()
    
    print(f"After filtering for holding concepts: {len(df)} rows")
    print(f"Rows with identified issuers: {df['held_ticker'].notna().sum()}")
    
    # For rows without specific issuer, use the concept as a category
    df["holding_category"] = df["concept"]
    
    # Convert value based on unit
    def convert_value(row):
        if row["unit"] in ["iso4217:USD", "USD"]:
            return row["value"], "market_value"
        elif row["unit"] in ["xbrli:shares", "shares"]:
            return row["value"], "shares"
        elif row["unit"] in ["xbrli:pure", "pure"]:
            return row["value"], "ratio"
        else:
            return row["value"], row["unit"]
    
    df[["converted_value", "value_type"]] = df.apply(convert_value, axis=1, result_type="expand")
    
    # Select and rename columns for panel
    panel = df[["filer_ticker", "as_of_date", "held_ticker", "held_cik", 
                "converted_value", "value_type", "concept", "holding_category",
                "form", "accession", "period_start", "period_end", "dimensions"]].copy()
    
    panel.columns = ["filer_ticker", "as_of_date", "held_ticker", "held_cik",
                     "value", "value_type", "concept", "holding_category",
                     "form", "accession", "period_start", "period_end", "dimensions"]
    
    # Ensure date types
    panel["as_of_date"] = pd.to_datetime(panel["as_of_date"]).dt.date
    if "filing_date" in panel.columns:
        panel["filing_date"] = pd.to_datetime(panel["filing_date"], errors="coerce").dt.date
    if "report_date" in panel.columns:
        panel["report_date"] = pd.to_datetime(panel["report_date"], errors="coerce").dt.date
    if "period_start" in panel.columns:
        panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce").dt.date
    if "period_end" in panel.columns:
        panel["period_end"] = pd.to_datetime(panel["period_end"], errors="coerce").dt.date
    
    # Sort
    panel = panel.sort_values(["filer_ticker", "as_of_date", "held_ticker"]).reset_index(drop=True)
    
    # Save
    panel.to_parquet(output_path, index=False)
    print(f"Saved panel to {output_path}: {len(panel)} rows")
    print(f"  Filers: {panel['filer_ticker'].nunique()}")
    print(f"  Held tickers identified: {panel['held_ticker'].notna().sum()} / {len(panel)}")
    print(f"  Unique held tickers: {panel['held_ticker'].dropna().nunique()}")
    print(f"  Value types: {panel['value_type'].value_counts().to_dict()}")
    
    return panel

def enrich_with_prices(holdings_panel_path: str, prices_path: str, output_path: str):
    """Enrich holdings panel with market prices for held securities"""
    holdings = pd.read_parquet(holdings_panel_path)
    prices = pd.read_parquet(prices_path)
    
    # Ensure date columns are correct type
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    holdings["as_of_date"] = pd.to_datetime(holdings["as_of_date"]).dt.date
    
    # Only process rows where we have shares and a held_ticker
    shares_mask = (holdings["value_type"] == "shares") & (holdings["held_ticker"].notna())
    shares_rows = holdings[shares_mask].copy()
    
    if len(shares_rows) == 0:
        print("No share-based holdings to enrich with prices")
        # Still need to set market_value for USD holdings
        holdings["market_value"] = None
        holdings["price_date"] = None
        holdings["price_used"] = None
        usd_mask = holdings["value_type"] == "market_value"
        holdings.loc[usd_mask, "market_value"] = holdings.loc[usd_mask, "value"]
        holdings.loc[usd_mask, "price_date"] = holdings.loc[usd_mask, "as_of_date"]
        holdings.loc[usd_mask, "price_used"] = 1.0
        
        holdings.to_parquet(output_path, index=False)
        print(f"Saved enriched panel to {output_path}: {len(holdings)} rows")
        print(f"  Rows with market_value: {holdings['market_value'].notna().sum()}")
        return holdings
    
    # Merge with prices on held_ticker and as_of_date
    # Use the closest price date <= as_of_date
    enriched_rows = []
    
    for _, row in shares_rows.iterrows():
        ticker = row["held_ticker"]
        as_of = row["as_of_date"]
        shares = row["value"]
        
        # Find price on or before as_of_date
        price_data = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)]
        if len(price_data) > 0:
            closest_price = price_data.loc[price_data["date"].idxmax()]
            market_value = shares * closest_price["adj_close"]
            row_copy = row.copy()
            row_copy["market_value"] = market_value
            row_copy["price_date"] = closest_price["date"]
            row_copy["price_used"] = closest_price["adj_close"]
            enriched_rows.append(row_copy)
    
    if enriched_rows:
        enriched_df = pd.DataFrame(enriched_rows)
        # Merge back
        holdings = holdings.merge(
            enriched_df[["filer_ticker", "as_of_date", "held_ticker", "concept", "market_value", "price_date", "price_used"]],
            on=["filer_ticker", "as_of_date", "held_ticker", "concept"],
            how="left"
        )
        print(f"Enriched {len(enriched_rows)} share-based holdings with market prices")
    else:
        holdings["market_value"] = None
        holdings["price_date"] = None
        holdings["price_used"] = None
    
    # For USD-denominated holdings, market_value = value
    usd_mask = holdings["value_type"] == "market_value"
    holdings.loc[usd_mask, "market_value"] = holdings.loc[usd_mask, "value"]
    holdings.loc[usd_mask, "price_date"] = holdings.loc[usd_mask, "as_of_date"]
    holdings.loc[usd_mask, "price_used"] = 1.0
    
    # Save
    holdings.to_parquet(output_path, index=False)
    print(f"Saved enriched panel to {output_path}: {len(holdings)} rows")
    print(f"  Rows with market_value: {holdings['market_value'].notna().sum()}")
    
    return holdings

def main():
    detailed_path = "detailed_holdings.parquet"
    panel_path = "holdings_panel.parquet"
    enriched_path = "holdings_panel_enriched.parquet"
    
    if not Path(detailed_path).exists():
        print(f"Input file not found: {detailed_path}")
        return
    
    # Step 1: Build clean panel
    panel = process_holdings(detailed_path, panel_path)
    
    # Step 2: Enrich with prices
    if Path("daily_prices/").exists():
        enrich_with_prices(panel_path, "daily_prices/", enriched_path)
    else:
        print("daily_prices/ not found, skipping price enrichment")

if __name__ == "__main__":
    main()