"""Predicting HDB resale flat prices with Linear Regression (Stage 2)."""

# Libraries need for project
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, confusion_matrix)

FIG_DIR = "figures"                       # plots are saved here for the report
os.makedirs(FIG_DIR, exist_ok=True)       # under /figures


def save_and_show(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.show()


# Read the data
df = pd.read_csv("resale_flat_prices_2017_onwards.csv")
print("Raw data:", df.shape)
print(df.head())

# Renaming text column / data & getting new column
df["year"] = df["month"].str.slice(0, 4).astype(int)                       # "2017-01" -> 2017
df["storey"] = df["storey_range"].str.extract(r"(\d+) TO (\d+)").astype(float).mean(axis=1)  # "10 TO 12" -> 11
df["lease_left"] = (df["remaining_lease"].str.extract(r"(\d+)\s*year")[0].astype(float)
                    + df["remaining_lease"].str.extract(r"(\d+)\s*month")[0].astype(float).fillna(0) / 12)
df["flat_age"] = df["year"] - df["lease_commence_date"]                    # how old the flat is

# Drop irrelevant columns
df = df.drop(columns=["block", "street_name", "month", "storey_range", "remaining_lease"])
df = df.dropna()
print("After cleaning:", df.shape)

# Visualisation
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

# Price by flat model, sorted by median (justifies keeping flat_model)
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

# What the numbers say:
#   floor_area_sqm (0.56) is the strongest single predictor -> keep
#   year           (0.41) prices drift upward over time     -> keep
#   storey         (0.34) higher floors cost more           -> keep
#   lease_commence_date (0.38), lease_left (0.30) and flat_age (-0.30) all describe
#   the flat's age (corr 0.98 / -1.00 with each other) -> keep only the strongest,
#   lease_commence_date, and drop the other two as redundant
# The boxplots show town, flat_type and flat_model separate the price ranges strongly,
# so we keep them too and one-hot encode them.

# Select features
num_features = ["floor_area_sqm", "year", "storey", "lease_commence_date"]
cat_features = ["town", "flat_type", "flat_model"]

X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True)
y = df["resale_price"]
print("\nFeature matrix:", X.shape)

# Split: 60% train / 20% validation / 20% test
X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=42)
print("Train / val / test sizes:", len(X_train), len(X_val), len(X_test))

# Train on the training set (minimises squared error)
model = LinearRegression()
model.fit(X_train, y_train)


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": np.sqrt(mse),          # RMSE / MAE are in SGD
            "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


# Training vs validation error (used to compare with the Random Forest)
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
results = pd.DataFrame({"train": evaluate(y_train, y_train_pred),
                        "validation": evaluate(y_val, y_val_pred)}).T
print("\nLinear regression errors:")
print(results.round(3))

# Which features push the price the most
coef = pd.Series(model.coef_, index=X.columns)
print("\nNumeric coefficients (SGD per unit):")
print(coef[num_features].round(1))
print("\nTop 10 coefficients (SGD, relative to the dropped category):")
print(coef.sort_values(key=abs, ascending=False).head(10).round(0))

# Predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_val_pred, s=2, alpha=0.1)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
plt.title("Linear Regression: Predicted vs Actual (validation set)")
plt.xlabel("Actual price (SGD)")
plt.ylabel("Predicted price (SGD)")
save_and_show("fig7_lr_pred_vs_actual.png")

# Residuals vs predicted
residuals = y_val - y_val_pred
plt.figure(figsize=(7, 4.5))
plt.scatter(y_val_pred, residuals, s=2, alpha=0.1)
plt.axhline(0, color="r", ls="--")
plt.title("Linear Regression: Residuals vs Predicted (validation set)")
plt.xlabel("Predicted price (SGD)")
plt.ylabel("Actual - predicted (SGD)")
save_and_show("fig8_lr_residuals.png")

bands = pd.cut(y_val, [0, 400_000, 600_000, 800_000, 1_000_000, np.inf])
print("\nMean residual by actual price band (validation):")
print(residuals.groupby(bands, observed=True).agg(["mean", "count"]).round(0))

# Confusion matrix of price tiers
labels = ["Low", "Mid", "High"]
actual_tier, edges = pd.qcut(y_val, 3, labels=labels, retbins=True)
print("\nTier boundaries (SGD):", [round(e) for e in edges])
edges[0], edges[-1] = -np.inf, np.inf        # catch predictions outside the range
pred_tier = pd.cut(y_val_pred, bins=edges, labels=labels)

cm = confusion_matrix(actual_tier, pred_tier, labels=labels)
print("Confusion matrix (price tiers, validation):")
print(cm)
print(f"Tier accuracy: {np.diag(cm).sum() / cm.sum():.3f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Linear Regression: Price Tiers (validation set)")
plt.xlabel("Predicted tier")
plt.ylabel("Actual tier")
save_and_show("fig9_lr_confusion.png")

# Test set: Use this if linear regression is the final chosen method.
test_results = evaluate(y_test, model.predict(X_test))
print("\nTest errors (final model only):")
print({k: round(v, 3) for k, v in test_results.items()})
