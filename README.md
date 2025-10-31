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


 
 ## 🧭 **4. Motivation**

My motivation for choosing this topic comes from both **personal experience** and **analytical curiosity**.

- 🔹 I actively follow the cryptocurrency market in my daily life, which makes it a naturally engaging field to study from a data-driven perspective.  
- 🔹 Unlike traditional stocks, cryptocurrencies **lack intrinsic or physical value**; their prices are largely driven by **human behavior, emotions, and collective psychology**.  
- 🔹 My own experience as a retail investor — often making **emotional or poorly timed decisions** that negatively affected my portfolio — motivated me to better understand how **social media activity, public attention, and investor psychology** influence market outcomes.  
- 🔹 I selected **Bitcoin** specifically because it is the **dominant cryptocurrency** and often acts as a **benchmark for the overall crypto market**, reflecting broader market sentiment and trends.  

Through this project, I aim to explore whether **behavioral indicators such as sentiment, attention, and investor psychology** can help explain or anticipate **Bitcoin’s price direction**.



## ⚙️ **5. Data Preparation and Analysis Plan**

Once all datasets are cleaned — meaning **empty columns and inconsistent values are removed** — they will be prepared for analysis.  
After obtaining a clean and usable dataset, the following methods will be applied to explore and model the relationship between behavioral indicators and Bitcoin’s price direction.

### 🔹 **1. Exploratory Data Analysis (EDA)**
- Perform descriptive statistics and correlation analysis to explore relationships between variables.  
- Use **visualization techniques** (line plots, scatter plots, heatmaps, histograms) to examine trends and patterns in behavioral indicators versus Bitcoin’s price movements.  

### 🔹 **2. Modeling and Machine Learning**
- Apply **regression and classification models** (e.g., Logistic Regression, Random Forest) to test the relationship between behavioral indicators and Bitcoin’s price direction.  
- Evaluate model performance using metrics such as **accuracy**, **R²**, and **feature importance** to identify the most influential behavioral signals.

The goal is to determine whether behavioral and attention-based indicators can help **explain or predict Bitcoin’s short-term price direction**.



