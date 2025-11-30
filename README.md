# 📊 **Understanding Bitcoin Price Movements Through Behavioral Data**

## 🧩 1. Project Overview

This project investigates whether **behavioral indicators** can help explain or anticipate **Bitcoin’s short-term price movements**.  
Instead of focusing solely on financial metrics, the analysis incorporates external human-driven factors such as:

- Social media sentiment and activity  
- Public attention (Google Trends)  
- Investor psychology (Fear & Greed Index)  

The aim is to determine whether collective behavior — expressed online or through search activity — meaningfully relates to Bitcoin’s daily price direction.


---

## 🗂️ 2. Data Overview & Collection Methods

The project uses four main datasets representing different behavioral and financial dimensions.

| Dataset | Description | Source | Collection Method |
|--------|-------------|--------|-------------------|
| **Bitcoin Price Data** | Open, high, low, close, volume values. | Yahoo Finance | Downloaded using the `yfinance` Python library. |
| **Bitcoin Tweets Data** | Millions of tweets containing the keyword “Bitcoin”, used for sentiment and volume analysis. | Kaggle | Downloaded as CSV. Processed using chunk-based cleaning due to file size. |
| **Google Trends Data** | Daily search interest for "bitcoin" and "buy crypto" from Web Search & YouTube Search. | Google Trends | Exported as CSV via Google Trends interface. |
| **Crypto Fear & Greed Index** | Market psychology indicator (0 = extreme fear, 100 = extreme greed). | Yahoo Finance | Downloaded using the `yfinance` Python library. |

All datasets roughly cover the period **2023–2025**, though exact ranges vary, so final analysis uses the **common overlapping time window**.


---

## 🎯 3. Hypotheses

**Null Hypothesis (H₀):**  
Behavioral indicators (tweet sentiment, search interest, Fear & Greed Index) have **no significant effect** on Bitcoin’s short-term price direction.

**Alternative Hypothesis (H₁):**  
Behavioral indicators have a **significant effect** on Bitcoin’s short-term price direction.


---

## 💪 4. Motivation

I chose this topic because cryptocurrencies behave differently from traditional financial assets:

- Bitcoin’s value is heavily influenced by **human behavior and emotion**, not intrinsic fundamentals.  
- As someone who followed crypto markets actively, I noticed that **social media narratives** strongly shape market mood.  
- My own negative investment experiences fueled curiosity about how much investors are influenced by **sentiment, hype, and online behavior**.  
- Bitcoin was selected because it is the **dominant cryptocurrency** and sets the tone for the entire crypto market.

This project aims to test whether these behavioral factors actually align with short-term market outcomes.


---

## ⚙️ 5. Data Preparation & Analysis Plan

After cleaning each dataset (removal of empty values, inconsistent formatting, duplicated entries), the following analysis steps are applied:

### 5.1 Exploratory Data Analysis (EDA)

- Summary statistics and distribution analysis  
- Time-series visualization  
- Correlation heatmaps  
- Behavioral indicator trend inspection  

### 5.2 Statistical Hypothesis Testing

- Pearson correlation tests  
- Independent sample mean-difference tests (Up vs Down days)


---

## ✅ 6. Work Completed So Far (Data Cleaning, Feature Engineering & EDA Summary)

### 6.1 Tweet Cleaning and Feature Engineering

The raw tweet dataset was extremely large, so a **chunk-based processing pipeline** was used with support from an AI assistant.

Steps performed:

- Removed duplicated text entries (spam/bots)  
- Applied **VADER sentiment analysis** to compute a compound sentiment score  
- Detected bullish/bearish tweets via keyword dictionaries  
- Weighted sentiment obtained

 Daily features created:

- `total_vol` — daily tweet volume  
- `avg_sentiment` — weighted daily sentiment score  
- `bull_ratio`, `bear_ratio` — ratio of bullish/bearish tweets  
- `sentiment_spread` — `bull_ratio - bear_ratio`  
- `sentiment_momentum` — day-to-day change in sentiment_spread  
  - indicates whether market tone is shifting toward optimism or pessimism

The final processed tweet dataset is saved as **`processed_tweets_final.csv`** and provides a compact behavioral summary of millions of tweets.

---

### 6.2 Bitcoin Price Feature Engineering

Using `yfinance` Bitcoin price data, several market-relevant features were added to better capture short-term dynamics:

- `Daily_Return` — daily percentage return  
- `Log_Return` — log-transformed return  
- `Volatility_7d` — 7-day rolling volatility of returns  
- `Momentum_3d` — 3-day momentum indicator  
- `MA_7` — 7-day moving average  
- `Trend_Score` — simple trend indicator  
- `Next_Day_Return` — following day’s return  
- `Target_Direction` — binary label (1 = next-day return positive, 0 = non-positive)

These engineered features allow direct comparison between **market structure** and **behavioral signals**.

---

### 6.3 Google Trends Processing

- Google Trends sometimes provides **weekly or interval-based** scores, which creates flat segments in the original time series.  
- To align these with daily Bitcoin data, search scores were **interpolated to daily frequency**.  
- This produces smoother and analysis-ready daily series for:
  - `btc_web`, `btc_yt` (Bitcoin web and YouTube interest)  
  - `crypto_web`, `crypto_yt` (general crypto interest)

---

### 6.4 EDA Findings

Through exploratory visualizations and summary statistics:

- Tweet sentiment is generally **consistently positive**, reflecting an optimism bias in crypto discussions.  
- Bullish signals (bull_ratio) frequently exceed bearish ones (bear_ratio).  
- Google Trends interest shows spikes around market events but does not clearly track short-term returns.  
- The Crypto Fear & Greed Index moves between fear and greed phases but exhibits **weak alignment with daily returns**.  
- A combined correlation heatmap reveals **weak linear relationships** between behavioral features and price-based features such as returns and volatility.

---

### 6.5 Hypothesis Testing Results

Two groups of statistical tests were applied to evaluate the effect of behavioral variables on Bitcoin’s short-term price direction.

#### 6.5.1 Correlation Tests (Pearson)

Each behavioral indicator was tested against **next-day returns** and volatility.  
For all tested pairs:

- p-values > 0.05  
- → **Fail to reject H₀** (no evidence of linear relationship at the 5% level)

This result is consistent with the high noise and unpredictability of Bitcoin in the short term.

#### 6.5.2 Mean Difference Tests (Up Days vs Down Days)

Independent sample tests were used to compare behavioral features on:

- days when Bitcoin price increased  
- days when Bitcoin price decreased  

Most variables (average sentiment, bull_ratio, bear_ratio, Fear & Greed Index) showed:

- no statistically significant difference between up days and down days  
- → **Fail to reject H₀** for their mean equality

Only **sentiment_spread** showed mixed evidence (significant in t-test but not in Mann–Whitney), suggesting a weak and unstable signal.

**Overall conclusion so far:**  
Behavioral indicators exhibit **weak or inconsistent influence** on Bitcoin’s day-to-day price direction.  
In addition, there appears to be a persistent **positive sentiment bias** in crypto-related discussions, where users remain optimistic even during price declines. This bias helps explain the lack of strong predictive relationships.


---

## 📈 7. Possible Outcomes

After combining all behavioral and price-based features, the analysis may lead to one or more of the following outcomes:

- Behavioral indicators such as tweet sentiment, search interest, and Fear & Greed Index may show **limited but interpretable patterns** with Bitcoin’s price direction.  
- Some signals may work only around **specific events or high-volatility periods**, rather than consistently over time.  
- Machine learning models (in future steps) may uncover **nonlinear or interaction-based effects** not visible in simple correlations.  
- Alternatively, the results may support the idea that **short-term Bitcoin movements are largely sentiment-insensitive and highly noisy**, which is itself an important finding.


---

## ⚠️ 8. Limitations

- Tweet data contains **noise, bots, and repeated content**, which may still influence results despite cleaning.  
- Google Trends data is an **indirect proxy** for attention and sometimes originally weekly, requiring interpolation.  
- The Fear & Greed Index is a **high-level aggregate indicator** and may not fully capture intraday or short-term shifts in sentiment.  
- Macro news, regulation announcements, and other fundamental drivers are **not explicitly modeled** in this phase of the project.


---

## 🔮 9. Future Work

- Extend the analysis to other cryptocurrencies (e.g., Ethereum, BNB, Dogecoin) for comparison.  
- Add **Reddit and news sentiment** as additional behavioral signals.  
- Explore **nonlinear machine learning models** (Random Forest, XGBoost, Neural Networks) to capture more complex relationships.  
- Study **longer horizons** (e.g., weekly or monthly returns) where behavioral effects may be stronger than in daily data.


---

## ⏱️ 10. Project Timeline

| Date | Task Description |
|------|------------------|
| **31 October** | Submit project proposal (`README.md`) with project idea, data sources, and collection plan. |
| **28 November** | Complete data collection, data cleaning, feature engineering, EDA, and hypothesis testing. |
| **02 January** | Apply machine learning methods on the engineered dataset to test predictive relationships. |
| **09 January (23:59)** | Final project submission. |
