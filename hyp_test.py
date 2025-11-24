import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ==================================
# LOAD MERGED DATA
# ==================================

df = pd.read_csv("csv_files/merged_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Next day return for predictive correlation tests
df["Next_Return"] = df["Daily_Return"].shift(-1)

# Binary variable: 1 if today's return is positive, else 0
df["Price_Up"] = (df["Daily_Return"] > 0).astype(int)

# Remove rows with shift-created NaN
df = df.dropna()

print(f"N = {len(df)} days")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# ==================================
# 1. CORRELATION TESTS
# ==================================

print("=== Correlation tests (Pearson) ===")
print("Testing behavioral variables against next-day returns or volatility.\n")

corr_tests = [
    ("avg_sentiment", "Next_Return", "Sentiment vs Next Return"),
    ("sentiment_spread", "Next_Return", "Sentiment Spread vs Next Return"),
    ("bull_ratio", "Next_Return", "Bull Ratio vs Next Return"),
    ("bear_ratio", "Next_Return", "Bear Ratio vs Next Return"),
    ("fng_value", "Next_Return", "Fear & Greed vs Next Return"),
    ("total_vol", "Volatility_7d", "Tweet Volume vs Volatility_7d"),
]

for x, y, label in corr_tests:
    df_test = df[[x, y]].dropna()
    if len(df_test) < 3:
        continue
    r, p = stats.pearsonr(df_test[x], df_test[y])
    print(f"{label:35s}  r = {r:7.3f},  p = {p:.4f}")
print()

# ==================================
# 2. MEAN DIFFERENCE TESTS
# ==================================

print("=== Mean difference tests (Up days vs Down days) ===\n")

up = df[df["Price_Up"] == 1]
down = df[df["Price_Up"] == 0]

mean_tests = [
    ("avg_sentiment", "Average Sentiment"),
    ("sentiment_spread", "Sentiment Spread"),
    ("bull_ratio", "Bull Ratio"),
    ("bear_ratio", "Bear Ratio"),
    ("fng_value", "Fear & Greed Index"),
]

for var, label in mean_tests:
    up_vals = up[var].dropna()
    down_vals = down[var].dropna()

    if len(up_vals) < 3 or len(down_vals) < 3:
        continue

    t_stat, t_p = stats.ttest_ind(up_vals, down_vals, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(up_vals, down_vals, alternative="two-sided")

    print(f"{label}:")
    print(f"  Up days   mean = {up_vals.mean():.4f} (n={len(up_vals)})")
    print(f"  Down days mean = {down_vals.mean():.4f} (n={len(down_vals)})")
    print(f"  T-test p-value       = {t_p:.4f}")
    print(f"  Mann-Whitney p-value = {u_p:.4f}\n")
