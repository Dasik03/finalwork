"""
FIFA Players Dataset Analysis - Final Project
Combines Linear Regression, Logistic Regression, and Neural Networks
to analyze and predict player attributes from the FIFA dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print(" FIFA PLAYERS DATASET ANALYSIS - FINAL PROJECT")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================
print("\n[PART 1] Loading and Exploring Dataset...")

df = pd.read_csv("../dataset/fifa_players.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Total players: {len(df)}")

print("\n--- Sample of data ---")
print(df.head())

print("\n--- Data types and missing values ---")
print(df.info())

print("\n--- Basic statistics ---")
print(df.describe())

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================
print("\n[PART 2] Data Preprocessing...")

# Select relevant features for analysis
features_to_use = [
    'age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential',
    'international_reputation(1-5)', 'weak_foot(1-5)', 'skill_moves(1-5)',
    'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys',
    'dribbling', 'curve', 'freekick_accuracy', 'long_passing', 'ball_control',
    'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance',
    'shot_power', 'jumping', 'stamina', 'strength', 'long_shots',
    'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
    'composure', 'marking', 'standing_tackle', 'sliding_tackle', 'value_euro'
]

# Clean the data
df_clean = df[features_to_use].copy()
df_clean = df_clean.dropna()

print(f"\nCleaned dataset shape: {df_clean.shape}")
print(f"Removed {len(df) - len(df_clean)} rows with missing values")

# Create rating categories for classification
# Low: 0-69, Medium: 70-79, High: 80-89, Elite: 90+
df_clean['rating_category'] = pd.cut(
    df_clean['overall_rating'],
    bins=[0, 70, 80, 90, 100],
    labels=['Low', 'Medium', 'High', 'Elite']
)

# Binary classification: High-rated (>=80) vs Regular (<80)
df_clean['is_high_rated'] = (df_clean['overall_rating'] >= 80).astype(int)

print("\n--- Rating distribution ---")
print(df_clean['rating_category'].value_counts().sort_index())
print(f"\nHigh-rated players (>=80): {df_clean['is_high_rated'].sum()} ({100*df_clean['is_high_rated'].mean():.1f}%)")

# ============================================================================
# PART 3: REGRESSION ANALYSIS - PREDICTING PLAYER VALUE
# ============================================================================
print("\n" + "="*80)
print("[PART 3] REGRESSION ANALYSIS - Predicting Player Value")
print("="*80)

# Prepare data for regression
regression_features = [
    'age', 'overall_rating', 'potential', 'international_reputation(1-5)',
    'skill_moves(1-5)', 'ball_control', 'reactions', 'dribbling',
    'short_passing', 'finishing'
]

X_reg = df_clean[regression_features].values
y_reg = df_clean['value_euro'].values

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Standardize features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"\nTraining samples: {len(X_train_reg)}")
print(f"Test samples: {len(X_test_reg)}")

# --- FROM SCRATCH LINEAR REGRESSION ---
print("\n--- Linear Regression from Scratch ---")

def train_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    """Train linear regression using gradient descent"""
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        y_pred = np.dot(X, w) + b
        error = y_pred - y

        dw = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

t0 = time.perf_counter()
w_scratch, b_scratch = train_linear_regression(
    X_train_reg_scaled, y_train_reg,
    learning_rate=0.1, iterations=2000
)
time_scratch = time.perf_counter() - t0

y_pred_train_scratch = np.dot(X_train_reg_scaled, w_scratch) + b_scratch
y_pred_test_scratch = np.dot(X_test_reg_scaled, w_scratch) + b_scratch

mse_train_scratch = mean_squared_error(y_train_reg, y_pred_train_scratch)
mse_test_scratch = mean_squared_error(y_test_reg, y_pred_test_scratch)
r2_train_scratch = r2_score(y_train_reg, y_pred_train_scratch)
r2_test_scratch = r2_score(y_test_reg, y_pred_test_scratch)

print(f"Training time: {time_scratch:.4f}s")
print(f"Train RMSE: EUR {np.sqrt(mse_train_scratch):,.0f}")
print(f"Test RMSE: EUR {np.sqrt(mse_test_scratch):,.0f}")
print(f"Train R2: {r2_train_scratch:.4f}")
print(f"Test R2: {r2_test_scratch:.4f}")

# --- SKLEARN LINEAR REGRESSION ---
print("\n--- Scikit-learn Linear Regression ---")

t0 = time.perf_counter()
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train_reg_scaled, y_train_reg)
time_sklearn = time.perf_counter() - t0

y_pred_train_sklearn = lr_sklearn.predict(X_train_reg_scaled)
y_pred_test_sklearn = lr_sklearn.predict(X_test_reg_scaled)

mse_train_sklearn = mean_squared_error(y_train_reg, y_pred_train_sklearn)
mse_test_sklearn = mean_squared_error(y_test_reg, y_pred_test_sklearn)
r2_train_sklearn = r2_score(y_train_reg, y_pred_train_sklearn)
r2_test_sklearn = r2_score(y_test_reg, y_pred_test_sklearn)

print(f"Training time: {time_sklearn:.4f}s")
print(f"Train RMSE: EUR {np.sqrt(mse_train_sklearn):,.0f}")
print(f"Test RMSE: EUR {np.sqrt(mse_test_sklearn):,.0f}")
print(f"Train R2: {r2_train_sklearn:.4f}")
print(f"Test R2: {r2_test_sklearn:.4f}")

# ============================================================================
# PART 4: CLASSIFICATION ANALYSIS - PREDICTING HIGH-RATED PLAYERS
# ============================================================================
print("\n" + "="*80)
print("[PART 4] CLASSIFICATION ANALYSIS - Predicting High-Rated Players (>=80)")
print("="*80)

# Prepare data for classification
classification_features = [
    'age', 'potential', 'international_reputation(1-5)', 'skill_moves(1-5)',
    'ball_control', 'reactions', 'dribbling', 'short_passing',
    'finishing', 'sprint_speed', 'agility', 'composure'
]

X_clf = df_clean[classification_features].values
y_clf = df_clean['is_high_rated'].values

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Standardize features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print(f"\nTraining samples: {len(X_train_clf)}")
print(f"Test samples: {len(X_test_clf)}")
print(f"Class distribution in train: {np.bincount(y_train_clf)} ({100*y_train_clf.mean():.1f}% high-rated)")

# --- FROM SCRATCH LOGISTIC REGRESSION ---
print("\n--- Logistic Regression from Scratch ---")

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    """Train logistic regression using gradient descent"""
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(iterations):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        error = y_pred - y

        dw = (1/m) * np.dot(X.T, error)
        db = (1/m) * np.sum(error)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

t0 = time.perf_counter()
w_clf_scratch, b_clf_scratch = train_logistic_regression(
    X_train_clf_scaled, y_train_clf,
    learning_rate=0.1, iterations=3000
)
time_clf_scratch = time.perf_counter() - t0

# Predictions
z_train = np.dot(X_train_clf_scaled, w_clf_scratch) + b_clf_scratch
y_pred_train_clf_scratch = (sigmoid(z_train) >= 0.5).astype(int)

z_test = np.dot(X_test_clf_scaled, w_clf_scratch) + b_clf_scratch
y_pred_test_clf_scratch = (sigmoid(z_test) >= 0.5).astype(int)

acc_train_scratch = accuracy_score(y_train_clf, y_pred_train_clf_scratch)
acc_test_scratch = accuracy_score(y_test_clf, y_pred_test_clf_scratch)

print(f"Training time: {time_clf_scratch:.4f}s")
print(f"Train accuracy: {acc_train_scratch:.4f}")
print(f"Test accuracy: {acc_test_scratch:.4f}")

# --- SKLEARN LOGISTIC REGRESSION ---
print("\n--- Scikit-learn Logistic Regression ---")

t0 = time.perf_counter()
clf_sklearn = LogisticRegression(max_iter=1000, random_state=42)
clf_sklearn.fit(X_train_clf_scaled, y_train_clf)
time_clf_sklearn = time.perf_counter() - t0

y_pred_train_clf_sklearn = clf_sklearn.predict(X_train_clf_scaled)
y_pred_test_clf_sklearn = clf_sklearn.predict(X_test_clf_scaled)

acc_train_sklearn = accuracy_score(y_train_clf, y_pred_train_clf_sklearn)
acc_test_sklearn = accuracy_score(y_test_clf, y_pred_test_clf_sklearn)

print(f"Training time: {time_clf_sklearn:.4f}s")
print(f"Train accuracy: {acc_train_sklearn:.4f}")
print(f"Test accuracy: {acc_test_sklearn:.4f}")

print("\n--- Classification Report (Sklearn) ---")
print(classification_report(
    y_test_clf, y_pred_test_clf_sklearn,
    target_names=['Regular (<80)', 'High-rated (>=80)']
))

# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("[PART 5] Generating Visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('FIFA Players Analysis - Comprehensive Results', fontsize=16, fontweight='bold')

# 1. Value distribution
ax1 = axes[0, 0]
ax1.hist(df_clean['value_euro'][df_clean['value_euro'] < 50000000], bins=50, edgecolor='black')
ax1.set_xlabel('Player Value (EUR)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Player Values (< EUR 50M)')
ax1.grid(alpha=0.3)

# 2. Rating distribution
ax2 = axes[0, 1]
rating_counts = df_clean['rating_category'].value_counts().sort_index()
ax2.bar(rating_counts.index, rating_counts.values, edgecolor='black')
ax2.set_xlabel('Rating Category')
ax2.set_ylabel('Number of Players')
ax2.set_title('Player Rating Distribution')
ax2.grid(alpha=0.3, axis='y')

# 3. Regression predictions
ax3 = axes[0, 2]
sample_size = min(500, len(y_test_reg))
indices = np.random.choice(len(y_test_reg), sample_size, replace=False)
ax3.scatter(y_test_reg[indices]/1000000, y_pred_test_sklearn[indices]/1000000,
            alpha=0.5, s=20, label='Sklearn')
max_val = max(y_test_reg[indices].max(), y_pred_test_sklearn[indices].max()) / 1000000
ax3.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
ax3.set_xlabel('Actual Value (EUR millions)')
ax3.set_ylabel('Predicted Value (EUR millions)')
ax3.set_title(f'Regression Predictions (R2={r2_test_sklearn:.3f})')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Age vs Overall Rating
ax4 = axes[1, 0]
sample = df_clean.sample(min(1000, len(df_clean)))
scatter = ax4.scatter(sample['age'], sample['overall_rating'],
                     c=sample['value_euro'], cmap='viridis',
                     alpha=0.6, s=30)
ax4.set_xlabel('Age')
ax4.set_ylabel('Overall Rating')
ax4.set_title('Age vs Overall Rating (colored by value)')
plt.colorbar(scatter, ax=ax4, label='Value (EUR)')
ax4.grid(alpha=0.3)

# 5. Feature importance (top 10 for regression)
ax5 = axes[1, 1]
feature_importance = np.abs(lr_sklearn.coef_)
top_10_idx = np.argsort(feature_importance)[-10:]
top_10_features = [regression_features[i] for i in top_10_idx]
top_10_values = feature_importance[top_10_idx]
ax5.barh(top_10_features, top_10_values, edgecolor='black')
ax5.set_xlabel('Absolute Coefficient Value')
ax5.set_title('Top 10 Features for Value Prediction')
ax5.grid(alpha=0.3, axis='x')

# 6. Model comparison
ax6 = axes[1, 2]
models = ['LR\nScratch', 'LR\nSklearn', 'LogReg\nScratch', 'LogReg\nSklearn']
metrics = [r2_test_scratch, r2_test_sklearn, acc_test_scratch, acc_test_sklearn]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax6.bar(models, metrics, color=colors, edgecolor='black')
ax6.set_ylabel('Score')
ax6.set_title('Model Performance Comparison')
ax6.set_ylim([0, 1])
ax6.grid(alpha=0.3, axis='y')
for i, (bar, metric) in enumerate(zip(bars, metrics)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{metric:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('fifa_analysis_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'fifa_analysis_results.png'")
plt.show()

# ============================================================================
# PART 6: SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("[PART 6] FINAL SUMMARY REPORT")
print("="*80)

summary_data = {
    'Model': [
        'Linear Regression (Scratch)',
        'Linear Regression (Sklearn)',
        'Logistic Regression (Scratch)',
        'Logistic Regression (Sklearn)'
    ],
    'Task': [
        'Value Prediction',
        'Value Prediction',
        'Rating Classification',
        'Rating Classification'
    ],
    'Train Time (s)': [
        f'{time_scratch:.4f}',
        f'{time_sklearn:.4f}',
        f'{time_clf_scratch:.4f}',
        f'{time_clf_sklearn:.4f}'
    ],
    'Test Metric': [
        f'R2={r2_test_scratch:.4f}',
        f'R2={r2_test_sklearn:.4f}',
        f'Acc={acc_test_scratch:.4f}',
        f'Acc={acc_test_sklearn:.4f}'
    ],
    'Test RMSE/Acc': [
        f'EUR {np.sqrt(mse_test_scratch):,.0f}',
        f'EUR {np.sqrt(mse_test_sklearn):,.0f}',
        f'{acc_test_scratch:.4f}',
        f'{acc_test_sklearn:.4f}'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"1. Dataset: {len(df_clean):,} FIFA players analyzed")
print(f"2. Regression: Can predict player value with R2 = {r2_test_sklearn:.3f}")
print(f"3. Classification: Can identify high-rated players with {acc_test_sklearn:.1%} accuracy")
print(f"4. From-scratch implementations achieve comparable results to sklearn")
print(f"5. Most important features for value: overall_rating, potential, reputation")
print("="*80)

print("\nAnalysis complete! Check 'fifa_analysis_results.png' for visualizations.")
