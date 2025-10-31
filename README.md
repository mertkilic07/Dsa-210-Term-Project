# Dsa 210 Term Project

# 📊 **Understanding Bitcoin Price Movements Through Behavioral Data**

## 🧩 **1. Project Overview**

This project aims to examine the relationship between **behavioral factors** and **Bitcoin’s short-term price direction**.  
Specifically, it investigates whether indicators such as **social media activity**, **public attention**, and **investor psychology** can help explain or predict Bitcoin’s daily price changes.  

To explore this relationship, the project integrates multiple datasets capturing different behavioral dimensions — including **tweet activity and sentiment**, **Google Trends web and YouTube search interest**, and the **Crypto Fear & Greed Index** — alongside Bitcoin price data.  
The analysis focuses on identifying whether patterns in collective investor sentiment and public engagement correspond to measurable movements in Bitcoin’s market behavior.


## 🗂️ **2. Data Overview & Collection Ways & Formats**

The project uses four main datasets representing different behavioral and market dimensions.  
Each dataset is publicly available and collected through open-source tools or verified data repositories.

| Dataset | Description | Source | Collection Method |
|----------|--------------|---------|-------------------|
| **Bitcoin Price Data** | Daily open, close, high, low, and volume values of Bitcoin (BTC/USD).| **Yahoo Finance** | Collected using the **`yfinance`** Python library, which retrieves historical market data directly from Yahoo Finance. |
| **Bitcoin Tweets Data** | Tweets containing the keyword and hashtag “Bitcoin”, . Used to measure daily tweet activity and sentiment levels. | **Kaggle Dataset** | Downloaded as a **CSV file** from Kaggle’s public “Bitcoin Tweets” dataset. |
| **Google Trends Data** | Daily search interest scores for the keywords **“buy crypto”** and **“bitcoin”**, collected from both **Web Search** and **YouTube Search** categories. These metrics represent public attention and user interest toward cryptocurrency topics. | **Google Trends** | Exported as **CSV files** using the Google Trends interface for both platforms (Web and YouTube). |
| **Crypto Fear & Greed Index** | Daily investor psychology scores (0 = Extreme Fear, 100 = Extreme Greed). Reflects general market sentiment and investor confidence. | **Kaggle Dataset** | Downloaded as a **CSV file** from Kaggle’s public dataset repository containing historical Fear & Greed Index values. |

All datasets approximately cover the period between **January 2023 and January 2025**.  
However, since each dataset has a different update frequency and data availability range, the **exact start and end dates may be adjusted** depending on data consistency and project needs.

## 🎯 **3. Hypotheses**

-**Null Hypothesis (H₀):**  
 Social media activity, public attention, and investor psychology have **no significant effect** on Bitcoin’s short-term price direction.  

-**Alternative Hypothesis (H₁):**  
 Social media activity, public attention, and investor psychology have a **significant effect** on Bitcoin’s short-term price direction.

