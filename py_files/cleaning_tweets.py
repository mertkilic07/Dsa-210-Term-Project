import pandas as pd
import numpy as np
import os
import time
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already available
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ==============================
# CONFIGURATION
# ==============================
INPUT_FILE = "../Bitcoin_tweets.csv"
OUTPUT_FILE = "../csv_files/processed_tweets_final.csv"
CHUNK_SIZE = 50000  # Number of rows processed per chunk

# ==============================
# KEYWORD DICTIONARIES
# ==============================
# Bullish terms indicating optimism or accumulation
BULL_KEYWORDS = (
    r'\b(?:'
    r'buy|bought|long|moon|rocket|pump|hodl|bull|bullish|breakout|ath|'
    r'gem|accumul|support|wagmi|lfg|diamond\s?hand|btfd|rally|'
    r'green|surge|parabolic|buy.{0,5}dip|dip.{0,5}buy'
    r')\b'
)

# Bearish terms indicating fear, selling, or negative outlook
BEAR_KEYWORDS = (
    r'\b(?:'
    r'sell|sold|short|dump|crash|drop|bear|bearish|scam|'
    r'rug|rekt|ngmi|panic|bleed|ban|hack|'
    r'bubble|top|plummet|fud|death\s?cross|reject|weak|'
    r'exit|scared|afraid|worry'
    r')\b'
)

# ==============================
# CHECK INPUT
# ==============================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")

# ==============================
# PROCESS IN CHUNKS
# ==============================
start_time = time.time()
daily_chunks = []
total_rows = 0
total_spam_removed = 0

chunk_iter = pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    lineterminator="\n",
    encoding="utf-8",
    on_bad_lines="skip"
)

for chunk in chunk_iter:
    # Standardize column names
    chunk.columns = chunk.columns.str.lower().str.strip()

    # Identify essential columns
    date_col = next((c for c in ["date", "timestamp", "created_at"] if c in chunk.columns), None)
    text_col = next((c for c in ["text", "tweet", "content"] if c in chunk.columns), None)
    likes_col = next((c for c in ["likes", "favorites", "favorite_count"] if c in chunk.columns), None)

    # If key columns missing, skip chunk
    if date_col is None or text_col is None:
        continue

    # Convert date column → datetime.date
    chunk["dt"] = pd.to_datetime(chunk[date_col], errors="coerce").dt.date
    chunk = chunk.dropna(subset=["dt"]).copy()

    # Remove spam: duplicated text inside the chunk
    initial_len = len(chunk)
    chunk = chunk.drop_duplicates(subset=[text_col])
    total_spam_removed += (initial_len - len(chunk))

    # Clean and lowercase tweets
    chunk["clean_text"] = chunk[text_col].astype(str).str.lower()

    # Weight tweets using log-scaled likes (if exists)
    if likes_col is not None:
        likes_numeric = pd.to_numeric(chunk[likes_col], errors="coerce").fillna(0)
        chunk["weight"] = np.log1p(likes_numeric) + 1.0
    else:
        chunk["weight"] = 1.0

    # Sentiment score using VADER
    chunk["sentiment"] = chunk["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    # Weighted sentiment score
    chunk["weighted_sentiment"] = chunk["sentiment"] * chunk["weight"]

    # Bullish / Bearish keyword detection
    chunk["is_bullish"] = chunk["clean_text"].str.contains(BULL_KEYWORDS, regex=True).astype(int)
    chunk["is_bearish"] = chunk["clean_text"].str.contains(BEAR_KEYWORDS, regex=True).astype(int)

    # Daily aggregation (per chunk)
    daily = chunk.groupby("dt").agg(
        is_bullish=("is_bullish", "sum"),
        is_bearish=("is_bearish", "sum"),
        weighted_sentiment=("weighted_sentiment", "sum"),
        weight=("weight", "sum"),
        total_vol=("clean_text", "count")
    )

    daily_chunks.append(daily)
    total_rows += initial_len

# ==============================
# COMBINE DAILY CHUNKS
# ==============================
if not daily_chunks:
    raise RuntimeError("No valid data processed. Check column names in the input file.")

full_daily = pd.concat(daily_chunks)

# Aggregate over date (merging chunks containing same day)
daily_final = full_daily.groupby(full_daily.index).agg(
    is_bullish=("is_bullish", "sum"),
    is_bearish=("is_bearish", "sum"),
    weighted_sentiment=("weighted_sentiment", "sum"),
    weight=("weight", "sum"),
    total_vol=("total_vol", "sum")
)

# ==============================
# FEATURE ENGINEERING
# ==============================

# Weighted average sentiment
daily_final["avg_sentiment"] = daily_final["weighted_sentiment"] / daily_final["weight"]

# Bullish/Bearish tweet ratios
daily_final["bull_ratio"] = daily_final["is_bullish"] / daily_final["total_vol"]
daily_final["bear_ratio"] = daily_final["is_bearish"] / daily_final["total_vol"]

# Sentiment spread (bullish dominance indicator)
daily_final["sentiment_spread"] = daily_final["bull_ratio"] - daily_final["bear_ratio"]

# Day-to-day sentiment momentum signal
daily_final["sentiment_momentum"] = daily_final["sentiment_spread"].diff()

# Replace any NaN values
daily_final = daily_final.fillna(0.0)

# Convert index to datetime and name it
daily_final.index = pd.to_datetime(daily_final.index)
daily_final.index.name = "Date"

# ==============================
# SAVE OUTPUT
# ==============================
daily_final.to_csv(OUTPUT_FILE)
