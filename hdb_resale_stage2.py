import os
import time
import numpy as np
import pandas as pd
SHOW_PLOTS = False   

import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, confusion_matrix)


os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = "figures"                       # plots are saved here for the report
os.makedirs(FIG_DIR, exist_ok=True)


def save_and_show(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# reading the data
df = pd.read_csv("resale_flat_prices_2017_onwards.csv")
print("Raw data:", df.shape)

# renaming text column / data $ getting new column
df["year"] = df["month"].str.slice(0, 4).astype(int)
df["storey"] = df["storey_range"].str.extract(r"(\d+) TO (\d+)").astype(float).mean(axis=1)
df["lease_left"] = (df["remaining_lease"].str.extract(r"(\d+)\s*year")[0].astype(float)
                    + df["remaining_lease"].str.extract(r"(\d+)\s*month")[0].astype(float).fillna(0) / 12)
df["flat_age"] = df["year"] - df["lease_commence_date"]

# drop irrelevant columns
df = df.drop(columns=["block", "street_name", "month", "storey_range", "remaining_lease"])
df = df.dropna()
print("After cleaning:", df.shape)



print(df.head())

# Distribution of the target
plt.figure(figsize=(8, 5))
sns.histplot(df["resale_price"], bins=50, kde=True, color="skyblue")
plt.title("Distribution of HDB Resale Price")
plt.xlabel("Resale price (SGD)")
plt.ylabel("Number of flats")
save_and_show("fig1_price_distribution.png")

# Floor area vs price
plt.figure(figsize=(8, 5))
plt.scatter(df["floor_area_sqm"], df["resale_price"], s=2, alpha=0.1)
plt.title("Floor Area vs Resale Price")
plt.xlabel("Floor area (sqm)")
plt.ylabel("Resale price (SGD)")
save_and_show("fig2_floor_area.png")

# Price by town, sorted by median so the trend is readable
plt.figure(figsize=(12, 5))
town_order = df.groupby("town")["resale_price"].median().sort_values().index
sns.boxplot(x="town", y="resale_price", data=df, order=town_order)
plt.title("Resale Price by Town")
plt.xlabel("Town")
plt.ylabel("Resale price (SGD)")
plt.xticks(rotation=90)
save_and_show("fig3_town.png")

# Price by flat type
plt.figure(figsize=(8, 5))
sns.boxplot(x="flat_type", y="resale_price", data=df, order=sorted(df["flat_type"].unique()))
plt.title("Resale Price by Flat Type")
plt.xlabel("Flat type")
plt.ylabel("Resale price (SGD)")
plt.xticks(rotation=45)
save_and_show("fig4_flat_type.png")

# Price by flat model, sorted by median
plt.figure(figsize=(12, 5))
model_order = df.groupby("flat_model")["resale_price"].median().sort_values().index
sns.boxplot(x="flat_model", y="resale_price", data=df, order=model_order)
plt.title("Resale Price by Flat Model")
plt.xlabel("Flat model")
plt.ylabel("Resale price (SGD)")
plt.xticks(rotation=90)
save_and_show("fig5_flat_model.png")

# Pearson correlation -> which features are worth using
num = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(method="pearson"), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Pearson Correlation Matrix")
save_and_show("fig6_correlation.png")

print("\nCorrelation with resale_price:")
print(num.corr()["resale_price"].sort_values(ascending=False))



# features 
num_features = ["floor_area_sqm", "year", "storey", "lease_commence_date"]
cat_features = ["town", "flat_type", "flat_model"]

X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True)
y = df["resale_price"]
print("\nFeature matrix:", X.shape)


X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# 20% test set
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=42) # 20% validation set, 60% training set
print("Train / val / test sizes:", len(X_train), len(X_val), len(X_test))


def evaluate(y_true, y_pred):
    """MSE, RMSE and MAE are in SGD (MSE in SGD^2). R2 is unitless."""
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


# method 1 - linear regression

t0 = time.time()
model = LinearRegression()
model.fit(X_train, y_train) # train on training set
print(f"\nLinear regression trained in {time.time() - t0:.1f}s")

coef = pd.Series(model.coef_, index=X.columns)
print("\nLinear regression numeric coefficients (SGD per unit):")
print(coef[num_features].round(1))
print("\nTop 10 linear regression coefficients (relative to dropped category):")
print(coef.sort_values(key=abs, ascending=False).head(10).round(0))


# method 2 - random forest

TUNE = False        
best_leaf = 5       # used when TUNE is False
if TUNE:
    tune_mse = {}
    for leaf in [1, 5, 10]:
        m = RandomForestRegressor(n_estimators=50, min_samples_leaf=leaf,
                                  n_jobs=-1, random_state=42).fit(X_train, y_train)   # 50 trees, smaller forest
        tune_mse[leaf] = mean_squared_error(y_val, m.predict(X_val))   # comparing with validation set
        print(f"min_samples_leaf={leaf}: validation MSE = {tune_mse[leaf]:,.0f}")
    best_leaf = min(tune_mse, key=tune_mse.get) # best leaf - lowest validation mse
    print("Chosen min_samples_leaf:", best_leaf)

t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=best_leaf,
                           n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
print(f"\nRandom forest trained in {time.time() - t0:.1f}s")

importances = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 random forest feature importances:")
print(importances.sort_values(ascending=False).head(10).round(4))

plt.figure(figsize=(8, 5))
importances.sort_values().tail(10).plot(kind="barh", color="steelblue")
plt.title("Random Forest: Top 10 Feature Importances")
plt.xlabel("Importance")
save_and_show("fig10_rf_feature_importance.png")

# comparison table
models = {"Linear Regression": model, "Random Forest": rf}
preds = {name: {"train": m.predict(X_train), "val": m.predict(X_val)}
         for name, m in models.items()}

rows = {}
for name in models:
    rows[(name, "train")] = evaluate(y_train, preds[name]["train"])
    rows[(name, "validation")] = evaluate(y_val, preds[name]["val"])
comparison = pd.DataFrame(rows).T
print("\n=== Training vs validation errors ===")
print(comparison.round(3))
comparison.round(3).to_csv(os.path.join(FIG_DIR, "table1_comparison.csv")) 


# validation-set diagnostics for both models 

mean_price = y.mean()
print(f"\nMean resale price: {mean_price:,.0f} SGD")
for name in models:
    mae_val = comparison.loc[(name, "validation"), "MAE"]
    print(f"{name}: validation MAE = {mae_val / mean_price:.1%} of mean price")

# Predicted vs actual, put together
lo, hi = y_val.min(), y_val.max()
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, name in zip(axes, models):
    ax.scatter(y_val, preds[name]["val"], s=2, alpha=0.1)
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_title(f"{name} (validation set)")
    ax.set_xlabel("Actual price (SGD)")
axes[0].set_ylabel("Predicted price (SGD)")
save_and_show("fig7_pred_vs_actual_both.png")

# residuals vs predicted
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, name in zip(axes, models):
    ax.scatter(preds[name]["val"], y_val - preds[name]["val"], s=2, alpha=0.1)
    ax.axhline(0, color="r", ls="--")
    ax.set_title(f"{name}: residuals (validation set)")
    ax.set_xlabel("Predicted price (SGD)")
axes[0].set_ylabel("Actual - predicted (SGD)")
save_and_show("fig8_residuals_both.png")

# mean residual by actual price band (the over/underestimation numbers)
bands = pd.cut(y_val, [0, 400_000, 600_000, 800_000, 1_000_000, np.inf])
for name in models:
    res = y_val - preds[name]["val"]
    print(f"\n{name}: mean residual by actual price band (validation):")
    print(res.groupby(bands, observed=True).agg(["mean", "count"]).round(0))

# price-tier confusion matrices on validation (same tiers as stage 1:
#     terciles of the validation prices, about SGD 428,000 and 590,000)
labels = ["Low", "Mid", "High"]
actual_tier, edges = pd.qcut(y_val, 3, labels=labels, retbins=True)
print("\nTier boundaries (SGD):", [round(e) for e in edges])
edges[0], edges[-1] = -np.inf, np.inf
for name in models:
    pred_tier = pd.cut(preds[name]["val"], bins=edges, labels=labels)
    cm = confusion_matrix(actual_tier, pred_tier, labels=labels)
    print(f"\n{name}: confusion matrix (price tiers, validation):")
    print(cm)
    print(f"{name}: tier accuracy = {np.diag(cm).sum() / cm.sum():.3f}")
    if name == "Random Forest":   # linear regression matrix was already made in stage 1
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"{name}: Price Tier Confusion Matrix (validation set)")
        plt.xlabel("Predicted tier")
        plt.ylabel("Actual tier")
        save_and_show("fig9_confusion_random.png")

# calculating lower valuation MSE model
val_mse = {name: comparison.loc[(name, "validation"), "MSE"] for name in models}
final_name = min(val_mse, key=val_mse.get)
final_model = models[final_name]
print(f"\nFinal model (lowest validation MSE): {final_name}")

# test set being evaluated 
test_results = evaluate(y_test, final_model.predict(X_test))
print("\n=== Test errors (final model only) ===")
print({k: round(v, 3) for k, v in test_results.items()})
print(f"Test MAE = {test_results['MAE'] / mean_price:.1%} of mean price")