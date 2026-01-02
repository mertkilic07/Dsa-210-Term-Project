# ======================================================================
# ML.py - Machine Learning Code for Bitcoin Price Direction Prediction
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA LOADING
# ============================================================
ml_df = pd.read_csv("../csv_files/merged_data.csv", parse_dates=["Date"])
ml_df = ml_df.sort_values("Date").set_index("Date")

print(f"Original dataset shape: {ml_df.shape}")
print(f"Date range: {ml_df.index.min().date()} to {ml_df.index.max().date()}")
print(f"\nColumns: {list(ml_df.columns)}")


# ============================================================
# 2. MISSING VALUE HANDLING
# ============================================================
# Fear & Greed Index: interpolate (it's a continuous indicator)
if "fng_value" in ml_df.columns:
    ml_df["fng_value"] = ml_df["fng_value"].interpolate(method="linear").ffill().bfill()

# Tweet-related columns: fill with 0 (no tweets = no activity)
tweet_cols = ["avg_sentiment", "bull_ratio", "bear_ratio",
              "sentiment_spread", "sentiment_momentum", "total_vol"]
for col in tweet_cols:
    if col in ml_df.columns:
        ml_df[col] = ml_df[col].fillna(0)

# Remaining numeric columns: fill with column mean
numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
ml_df[numeric_cols] = ml_df[numeric_cols].fillna(ml_df[numeric_cols].mean())

print("Missing values handled.")
print(f"Remaining NaN count: {ml_df.isna().sum().sum()}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# --- 3.1 Lag Features ---
lag_base_cols = [
    "Daily_Return", "Volatility_7d", "Trend_Score",
    "avg_sentiment", "bull_ratio", "bear_ratio",
    "sentiment_spread", "sentiment_momentum", "total_vol",
    "fng_value", "btc_web", "btc_yt", "crypto_web", "crypto_yt"
]
lag_base_cols = [col for col in lag_base_cols if col in ml_df.columns]
lag_periods = [1, 2, 3, 7]

for col in lag_base_cols:
    for lag in lag_periods:
        ml_df[f"{col}_lag{lag}"] = ml_df[col].shift(lag)

print(f"Created {len(lag_base_cols) * len(lag_periods)} lag features.")


# --- 3.2 Rolling Statistics ---
rolling_cols = ["Daily_Return", "Volatility_7d", "Trend_Score",
                "avg_sentiment", "sentiment_spread"]
rolling_cols = [col for col in rolling_cols if col in ml_df.columns]

for col in rolling_cols:
    ml_df[f"{col}_roll7_mean"] = ml_df[col].rolling(window=7).mean()
    ml_df[f"{col}_roll7_std"] = ml_df[col].rolling(window=7).std()
    ml_df[f"{col}_roll14_mean"] = ml_df[col].rolling(window=14).mean()
    ml_df[f"{col}_roll14_std"] = ml_df[col].rolling(window=14).std()

print(f"Created {len(rolling_cols) * 4} rolling features.")


# --- 3.3 Regime Features ---
if "Volatility_7d" in ml_df.columns:
    vol_median_30 = ml_df["Volatility_7d"].rolling(window=30).median()
    ml_df["Regime_HighVol"] = (ml_df["Volatility_7d"] > vol_median_30).astype(int)
    print("Created regime feature: 'Regime_HighVol'")


# --- 3.4 Interaction Features ---
if "sentiment_spread" in ml_df.columns and "total_vol" in ml_df.columns:
    ml_df["SentSpread_x_LogVol"] = ml_df["sentiment_spread"] * np.log1p(ml_df["total_vol"])
    print("Created: 'SentSpread_x_LogVol'")

if "Trend_Score" in ml_df.columns and "Volatility_7d" in ml_df.columns:
    ml_df["Trend_x_Vol"] = ml_df["Trend_Score"] * ml_df["Volatility_7d"]
    print("Created: 'Trend_x_Vol'")

print(f"\nTotal features after engineering: {ml_df.shape[1]}")


# --- 3.5 Feature Engineering Summary ---
print("=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

lag_features = [col for col in ml_df.columns if '_lag' in col]
rolling_features = [col for col in ml_df.columns if '_roll' in col]
regime_features = [col for col in ml_df.columns if 'Regime' in col]
interaction_features = [col for col in ml_df.columns if '_x_' in col]

print(f"\nLag Features:         {len(lag_features)}")
print(f"Rolling Features:     {len(rolling_features)}")
print(f"Regime Features:      {len(regime_features)}")
print(f"Interaction Features: {len(interaction_features)}")
print(f"\nTotal Columns:        {ml_df.shape[1]}")


# ============================================================
# 4. TARGET VARIABLE AND TRAIN-TEST SPLIT
# ============================================================
ml_df["Target"] = (ml_df["Daily_Return"].shift(-1) > 0).astype(int)
ml_df_clean = ml_df.dropna()

print(f"\nDataset after cleaning: {ml_df_clean.shape[0]} rows, {ml_df_clean.shape[1]} columns")
print(f"\nTarget distribution:")
print(ml_df_clean["Target"].value_counts())
print(f"\nUp days:   {ml_df_clean['Target'].sum()} ({100*ml_df_clean['Target'].mean():.1f}%)")
print(f"Down days: {(ml_df_clean['Target']==0).sum()} ({100*(1-ml_df_clean['Target'].mean()):.1f}%)")

# Select features
exclude_cols = ["Target"]
feature_cols = [col for col in ml_df_clean.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols]

print(f"\nNumber of features: {len(feature_cols)}")

X = ml_df_clean[feature_cols].copy()
y = ml_df_clean["Target"].copy()

# Train-test split (time-ordered)
test_ratio = 0.20
split_idx = int(len(X) * (1 - test_ratio))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nTraining set: {len(X_train)} samples ({100*(1-test_ratio):.0f}%)")
print(f"Test set:     {len(X_test)} samples ({100*test_ratio:.0f}%)")
print(f"\nTraining period: {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"Test period:     {X_test.index.min().date()} to {X_test.index.max().date()}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized (mean=0, std=1)")


# ============================================================
# 5. LOGISTIC REGRESSION MODEL
# ============================================================
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION PERFORMANCE")
print("=" * 60)

lr_model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["Down", "Up"]))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
            annot_kws={"size": 16})
plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()

# Feature Coefficients
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False).head(15)

plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
plt.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors)
plt.yticks(range(len(coef_df)), coef_df['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Logistic Regression Coefficients\n(Green = Positive for "Up", Red = Negative)', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 15 Features by Coefficient Magnitude:")
print(coef_df.to_string(index=False))


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================
print("\n" + "=" * 60)
print("RANDOM FOREST PERFORMANCE")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=8,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Down", "Up"]))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
            annot_kws={"size": 16})
plt.title("Confusion Matrix - Random Forest", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df,
            palette='viridis', hue='Feature', legend=False)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importances - Random Forest', fontsize=14)
plt.tight_layout()
plt.show()

print("Top 15 Features by Importance:")
print(importance_df.to_string(index=False))


# ============================================================
# 7. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr)
    ],
    'Random Forest': [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf)
    ]
})

print(comparison_df.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(comparison_df))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], comparison_df['Logistic Regression'],
               width, label='Logistic Regression', color='steelblue')
bars2 = ax.bar([i + width/2 for i in x], comparison_df['Random Forest'],
               width, label='Random Forest', color='forestgreen')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Metric'])
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
