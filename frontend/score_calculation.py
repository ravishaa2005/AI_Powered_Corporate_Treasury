import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with required financial columns for at least 16 quarters per company and returns
    a DataFrame with calculated composite scores: Risk, Growth, Profitability, Stability, Sector, Overall.
    """
    required_cols = ['FiscalDate', 'Company', 'Sector', 'Revenue', 'Expenses', 'EBITDA', 'Operating Margin %',
                     'Depreciation', 'Interest', 'PBT', 'Tax', 'Net Profit', 'EPS', 'GST',
                     'CorpTax%', 'Inflation%', 'RepoRate%', 'USDINR_Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required but not present in the input data.")

    # -- Date/time and sorting
    df = df.copy()
    df['FiscalDate'] = pd.to_datetime(df['FiscalDate'], errors='coerce')
    df = df.drop_duplicates()
    df = df.sort_values(by=["Company", "FiscalDate"]).reset_index(drop=True)

    # --- Growth Metrics ---
    df["Revenue_Growth"] = df.groupby("Company")["Revenue"].pct_change()
    df["Expense_Growth"] = df.groupby("Company")["Expenses"].pct_change()
    df["Profit_Growth"] = df.groupby("Company")["Net Profit"].pct_change()
    df["Expense_to_Revenue"] = df["Expenses"] / df["Revenue"]
    df[["Revenue_Growth", "Expense_Growth", "Profit_Growth"]] = df[["Revenue_Growth", "Expense_Growth", "Profit_Growth"]].fillna(0)

    # --- Standardization (z-score per company) ---
    z_features = ["Revenue_Growth", "Expense_Growth", "Profit_Growth", "Expense_to_Revenue", "GST", "Inflation%", "RepoRate%", "USDINR_Close"]
    for col in z_features:
        df[col + "_z"] = df.groupby("Company")[col].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0)

    # --- Risk Score (z-based, lower is safer) ---
    df["Risk_Score"] = (
        -df["Revenue_Growth_z"] +
        df["Expense_Growth_z"] +
        -df["Profit_Growth_z"] +
        df["Expense_to_Revenue_z"] +
        df["GST_z"] +
        df["Inflation%_z"] +
        df["RepoRate%_z"] +
        df["USDINR_Close_z"]
    )

    # --- Growth/Profitability/Stability/Sector/Overall Composite Metrics ---
    # Feature engineering:
    df["Year"] = df["FiscalDate"].dt.year
    df["Revenue_QoQ_Growth"] = df.groupby("Company")["Revenue"].pct_change(1) * 100
    df["Revenue_YoY_Growth"] = df.groupby("Company")["Revenue"].pct_change(4) * 100
    if "EPS" in df.columns:
        df["EPS_QoQ_Growth"] = df.groupby("Company")["EPS"].pct_change(1) * 100
        df["EPS_YoY_Growth"] = df.groupby("Company")["EPS"].pct_change(4) * 100
    else:
        df["EPS_QoQ_Growth"] = None
        df["EPS_YoY_Growth"] = None
    df[["Revenue_QoQ_Growth", "Revenue_YoY_Growth", "EPS_QoQ_Growth", "EPS_YoY_Growth"]] = df[["Revenue_QoQ_Growth", "Revenue_YoY_Growth", "EPS_QoQ_Growth", "EPS_YoY_Growth"]].fillna(0)

    # EBITDA/Net Profit/Interest Ratios
    df["EBITDA_Margin_%"] = (df["EBITDA"] / df["Revenue"].replace(0, np.nan)) * 100
    df["Net_Profit_Margin_%"] = (df["Net Profit"] / df["Revenue"].replace(0, np.nan)) * 100
    df["Interest_Coverage"] = df["EBITDA"] / df["Interest"].replace(0, np.nan)
    df["PBT_Interest_Ratio"] = df["PBT"] / df["Interest"].replace(0, np.nan)
    df["Debt_Proxy_%"] = (df["Interest"] / df["Revenue"].replace(0, np.nan)) * 100

    # --- Stability (volatility) ---
    df["Revenue_QoQ_Volatility"] = df.groupby("Company")["Revenue_QoQ_Growth"].rolling(window=4, min_periods=2).std().reset_index(level=0, drop=True)
    df["EPS_QoQ_Volatility"] = df.groupby("Company")["EPS_QoQ_Growth"].rolling(window=4, min_periods=2).std().reset_index(level=0, drop=True)
    df.fillna({"Revenue_QoQ_Volatility": 0, "EPS_QoQ_Volatility": 0}, inplace=True)

    # --- Sector-relative metrics ---
    df["Sector_Median_EBITDA_Margin"] = df.groupby(["Sector", "FiscalDate"])["EBITDA_Margin_%"].transform("median")
    df["EBITDA_Margin_Gap_vs_Sector"] = df["EBITDA_Margin_%"] - df["Sector_Median_EBITDA_Margin"]
    df["Sector_Median_Revenue_QoQ_Growth"] = df.groupby(["Sector", "FiscalDate"])["Revenue_QoQ_Growth"].transform("median")
    df["Revenue_Growth_Gap_vs_Sector"] = df["Revenue_QoQ_Growth"] - df["Sector_Median_Revenue_QoQ_Growth"]
    df.fillna(0, inplace=True)

    # --- MinMax scaling (0-100) ---
    growth_metrics = ["Revenue_QoQ_Growth", "Revenue_YoY_Growth", "EPS_QoQ_Growth", "EPS_YoY_Growth"]
    profitability_metrics = ["EBITDA_Margin_%", "Net_Profit_Margin_%", "Operating Margin %"]
    risk_metrics = ["Interest_Coverage", "PBT_Interest_Ratio", "Debt_Proxy_%"]
    stability_metrics = ["Revenue_QoQ_Volatility", "EPS_QoQ_Volatility"]
    sector_metrics = ["EBITDA_Margin_Gap_vs_Sector", "Revenue_Growth_Gap_vs_Sector"]
    all_metrics = growth_metrics + profitability_metrics + risk_metrics + stability_metrics + sector_metrics

    scaler = MinMaxScaler(feature_range=(0, 100))
    df[all_metrics] = df[all_metrics].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = scaler.fit_transform(df[all_metrics])
    scaled_df = pd.DataFrame(scaled, columns=[m+"_scaled" for m in all_metrics], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    for m in ["Debt_Proxy_%", "Revenue_QoQ_Volatility", "EPS_QoQ_Volatility"]:
        df[m+"_scaled_inv"] = 100 - df[m+"_scaled"]

    # --- Score formulas ---
    df["Growth_Score"] = (
        0.3 * df["Revenue_QoQ_Growth_scaled"] +
        0.2 * df["Revenue_YoY_Growth_scaled"] +
        0.3 * df["EPS_QoQ_Growth_scaled"] +
        0.2 * df["EPS_YoY_Growth_scaled"]
    )
    df["Profitability_Score"] = (
        0.35 * df["EBITDA_Margin_%_scaled"] +
        0.40 * df["Net_Profit_Margin_%_scaled"] +
        0.25 * df["Operating Margin %_scaled"]
    )
    df["Risk_Score"] = (
        0.5 * df["Debt_Proxy_%_scaled_inv"] +
        0.3 * df["Interest_Coverage_scaled"] +
        0.2 * df["PBT_Interest_Ratio_scaled"]
    )
    df["Stability_Score"] = (
        0.5 * df["Revenue_QoQ_Volatility_scaled_inv"] +
        0.5 * df["EPS_QoQ_Volatility_scaled_inv"]
    )
    df["Sector_Score"] = (
        0.5 * df["EBITDA_Margin_Gap_vs_Sector_scaled"] +
        0.5 * df["Revenue_Growth_Gap_vs_Sector_scaled"]
    )
    df["Overall_Score"] = df[["Growth_Score", "Profitability_Score", "Risk_Score", "Stability_Score", "Sector_Score"]].mean(axis=1)

    # --- Final selection ---
    drop_cols = ["Revenue_Growth", "Expense_Growth", "Profit_Growth", "Expense_to_Revenue", "Revenue_Growth_z", "Expense_Growth_z", "Profit_Growth_z", "Expense_to_Revenue_z", "GST_z", "Inflation%_z", "RepoRate%_z", "USDINR_Close_z", "Year", "Revenue_QoQ_Growth", "Revenue_YoY_Growth", "EPS_QoQ_Growth", "EPS_YoY_Growth", "EBITDA_Margin_%", "Net_Profit_Margin_%", "Interest_Coverage", "PBT_Interest_Ratio", "Debt_Proxy_%", "Revenue_QoQ_Volatility", "EPS_QoQ_Volatility", "Sector_Median_EBITDA_Margin", "EBITDA_Margin_Gap_vs_Sector", "Sector_Median_Revenue_QoQ_Growth", "Revenue_Growth_Gap_vs_Sector", "Revenue_QoQ_Growth_scaled", "Revenue_YoY_Growth_scaled", "EPS_QoQ_Growth_scaled", "EPS_YoY_Growth_scaled", "EBITDA_Margin_%_scaled", "Net_Profit_Margin_%_scaled", "Operating Margin %_scaled", "Interest_Coverage_scaled", "PBT_Interest_Ratio_scaled", "Debt_Proxy_%_scaled", "Revenue_QoQ_Volatility_scaled", "EPS_QoQ_Volatility_scaled", "EBITDA_Margin_Gap_vs_Sector_scaled", "Revenue_Growth_Gap_vs_Sector_scaled", "Debt_Proxy_%_scaled_inv", "Revenue_QoQ_Volatility_scaled_inv", "EPS_QoQ_Volatility_scaled_inv"]

    out_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return out_df

