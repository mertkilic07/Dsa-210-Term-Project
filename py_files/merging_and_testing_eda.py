import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8")
sns.set(font_scale=0.9)

# ==================================
# 1. LOAD DATASETS
# ==================================
# All datasets are loaded individually to preserve their original structure.
# Date columns are converted into datetime objects to enable alignment, filtering, and visualizations.

btc = pd.read_csv("../csv_files/bitcoin_adjusted.csv")
btc["Date"] = pd.to_datetime(btc["Date"])

tweets = pd.read_csv("../csv_files/processed_tweets_final.csv")
tweets["Date"] = pd.to_datetime(tweets["Date"])

fng = pd.read_csv("../csv_files/CryptoGreedFear.csv")
fng["Date"] = pd.to_datetime(fng["date"])

# ==============================
# Google Trends – raw monthly data
# ==============================

btc_web = pd.read_csv("../csv_files/Bitcoin_web_trends.csv", header=None, names=["Date", "btc_web"])
btc_web["Date"] = pd.to_datetime(btc_web["Date"])

btc_yt = pd.read_csv("../csv_files/Bitcoin_yt_trends.csv", header=None, names=["Date", "btc_yt"])
btc_yt["Date"] = pd.to_datetime(btc_yt["Date"])

crypto_web = pd.read_csv("../csv_files/Crypto_web_trends.csv", header=None, names=["Date", "crypto_web"])
crypto_web["Date"] = pd.to_datetime(crypto_web["Date"])

crypto_yt = pd.read_csv("../csv_files/Crypto_yt_trends.csv", header=None, names=["Date", "crypto_yt"])
crypto_yt["Date"] = pd.to_datetime(crypto_yt["Date"])

# ==============================
# Convert monthly → daily (forward fill)
# Each monthly value is carried forward to all days in that month
# ==============================

def expand_monthly_to_daily(df, value_col):
    """Takes a dataframe with a monthly Date column and expands it to daily frequency."""
    df = df.set_index("Date").resample("D").ffill()
    return df.reset_index()

btc_web = expand_monthly_to_daily(btc_web, "btc_web")
btc_yt = expand_monthly_to_daily(btc_yt, "btc_yt")
crypto_web = expand_monthly_to_daily(crypto_web, "crypto_web")
crypto_yt = expand_monthly_to_daily(crypto_yt, "crypto_yt")



# ==================================
# 1.5 ALIGN DATE RANGES (COMMON TIME PERIOD)
# ==================================
# Since each dataset may cover a different time span, all datasets are aligned to a common date range.
# This ensures consistent and meaningful comparisons across all time series.

datasets = [btc, tweets, fng, btc_web, btc_yt, crypto_web, crypto_yt]

min_common_date = max(df["Date"].min() for df in datasets)
max_common_date = min(df["Date"].max() for df in datasets)

print("Common date range:", min_common_date.date(), "to", max_common_date.date())

btc = btc[(btc["Date"] >= min_common_date) & (btc["Date"] <= max_common_date)]
tweets = tweets[(tweets["Date"] >= min_common_date) & (tweets["Date"] <= max_common_date)]
fng = fng[(fng["Date"] >= min_common_date) & (fng["Date"] <= max_common_date)]
btc_web = btc_web[(btc_web["Date"] >= min_common_date) & (btc_web["Date"] <= max_common_date)]
btc_yt = btc_yt[(btc_yt["Date"] >= min_common_date) & (btc_yt["Date"] <= max_common_date)]
crypto_web = crypto_web[(crypto_web["Date"] >= min_common_date) & (crypto_web["Date"] <= max_common_date)]
crypto_yt = crypto_yt[(crypto_yt["Date"] >= min_common_date) & (crypto_yt["Date"] <= max_common_date)]

print("All datasets loaded and aligned successfully.")


# ==================================
# 2. QUICK SUMMARY FUNCTION
# ==================================
# This helper function prints a compact summary for each dataset:
# shape, columns, sample rows, and descriptive statistics for numeric variables.

def quick_summary(name, df):
    print(f"\n\n========== {name} ==========")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())
    print("\nDescribe (numeric):")
    print(df.describe().T)


quick_summary("BTC Adjusted", btc)
quick_summary("Tweets (Daily Sentiment)", tweets)
quick_summary("Fear & Greed Index", fng)
quick_summary("BTC Web Trends", btc_web)
quick_summary("BTC YouTube Trends", btc_yt)
quick_summary("Crypto Web Trends", crypto_web)
quick_summary("Crypto YouTube Trends", crypto_yt)


# ==================================
# 3. TIME SERIES VISUALIZATIONS
# ==================================
# These plots help understand the overall movement and behavior of each dataset over time.
# Time-series behavior is essential for identifying patterns, trends, and potential correlations.

plt.figure(figsize=(12, 4))
plt.plot(btc["Date"], btc["Close"])
plt.title("Bitcoin Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.show()

if "Daily_Return" in btc.columns:
    plt.figure(figsize=(12, 4))
    plt.plot(btc["Date"], btc["Daily_Return"])
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Bitcoin Daily Return")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tweets["Date"], tweets["total_vol"])
plt.title("Daily Tweet Volume")
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tweets["Date"], tweets["avg_sentiment"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Daily Average Tweet Sentiment (VADER)")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(tweets["Date"], tweets["bull_ratio"], label="Bull Ratio")
plt.plot(tweets["Date"], tweets["bear_ratio"], label="Bear Ratio")
plt.title("Bull vs Bear Tweet Ratios")
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(fng["Date"], fng["fng_value"])
plt.title("Fear & Greed Index Over Time")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(btc_web["Date"], btc_web["btc_web"], label="Bitcoin Web")
plt.plot(btc_yt["Date"], btc_yt["btc_yt"], label="Bitcoin YouTube")
plt.title("Bitcoin Search Trends (Web vs YouTube)")
plt.xlabel("Date")
plt.ylabel("Trend Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(crypto_web["Date"], crypto_web["crypto_web"], label="Crypto Web")
plt.plot(crypto_yt["Date"], crypto_yt["crypto_yt"], label="Crypto YouTube")
plt.title("Crypto Search Trends (Web vs YouTube)")
plt.xlabel("Date")
plt.ylabel("Trend Score")
plt.legend()
plt.tight_layout()
plt.show()


# ==================================
# 4. DISTRIBUTIONS (HISTOGRAMS)
# ==================================
# Distribution plots help observe the overall shape, skewness, and volatility of key variables.

if "Daily_Return" in btc.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(btc["Daily_Return"].dropna(), bins=40)
    plt.title("Distribution of Bitcoin Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(6, 4))
plt.hist(tweets["avg_sentiment"].dropna(), bins=40)
plt.title("Distribution of Daily Average Sentiment")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(fng["fng_value"].dropna(), bins=40)
plt.title("Distribution of Fear & Greed Index")
plt.xlabel("Index")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ==================================
# 5. MERGE ALL DATASETS & CORRELATION ANALYSIS
# ==================================
# All datasets are merged on the Date column to form a unified table.
# This allows correlation analysis between price behavior, sentiment, and behavioral indicators.

df = btc[["Date", "Close", "Daily_Return", "Trend_Score"]].copy()

df = df.merge(
    tweets[
        [
            "Date",
            "avg_sentiment",
            "total_vol",
            "bull_ratio",
            "bear_ratio",
            "sentiment_spread",
            "sentiment_momentum",
        ]
    ],
    on="Date",
    how="left",
)

df = df.merge(fng[["Date", "fng_value"]], on="Date", how="left")
df = df.merge(btc_web, on="Date", how="left")
df = df.merge(btc_yt, on="Date", how="left")
df = df.merge(crypto_web, on="Date", how="left")
df = df.merge(crypto_yt, on="Date", how="left")

# Compute correlation matrix only on numeric columns
corr = df.select_dtypes("number").corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix: Price, Sentiment, and Behavioral Indicators")
plt.tight_layout()
plt.show()
