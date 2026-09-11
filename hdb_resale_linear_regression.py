"""Predicting HDB resale flat prices with Linear Regression."""

# Libraries need for project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, confusion_matrix)

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
plt.show()

# Price by flat type
plt.figure(figsize=(8, 5))
sns.boxplot(x="flat_type", y="resale_price", data=df, order=sorted(df["flat_type"].unique()))
plt.title("Resale Price by Flat Type")
plt.xlabel("Flat type")
plt.ylabel("Resale price (SGD)")
plt.xticks(rotation=45)
plt.show()

# Price by town, sorted by median so the trend is readable
plt.figure(figsize=(12, 5))
town_order = df.groupby("town")["resale_price"].median().sort_values().index
sns.boxplot(x="town", y="resale_price", data=df, order=town_order)
plt.title("Resale Price by Town")
plt.xlabel("Town")
plt.ylabel("Resale price (SGD)")
plt.xticks(rotation=90)
plt.show()

# Floor area vs price
plt.figure(figsize=(8, 5))
plt.scatter(df["floor_area_sqm"], df["resale_price"], s=2, alpha=0.1)
plt.title("Floor Area vs Resale Price")
plt.xlabel("Floor area (sqm)")
plt.ylabel("Resale price (SGD)")
plt.show()

# Pearson correlation -> which features are worth using
num = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(method="pearson"), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Pearson Correlation Matrix")
plt.show()

print("\nCorrelation with resale_price:")
print(num.corr()["resale_price"].sort_values(ascending=False))

# What the numbers say:
#   floor_area_sqm (0.56) is the strongest single predictor -> keep
#   year           (0.41) prices drift upward over time     -> keep
#   storey         (0.34) higher floors cost more           -> keep
#   lease_left     (0.30) / flat_age (-0.30) are the same thing (corr -1.00) -> keep only lease_left
#   lease_commence_date is 0.98 correlated with lease_left  -> drop, it is redundant
# The boxplots show town and flat_type separate the price ranges strongly,
# so we keep them too and one-hot encode them.

# Select features & spliting training and testing data
num_features = ["floor_area_sqm", "year", "storey", "lease_left"]
cat_features = ["town", "flat_type", "flat_model"]

X = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True)
y = df["resale_price"]
print("\nFeature matrix:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Which features push the price the most
coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 10 coefficients (SGD per unit):")
print(coef.head(10).round(0))

# Results: MSE and confusion matrix
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE : {mse:,.0f}")
print(f"RMSE: {np.sqrt(mse):,.0f}")          # average error in dollars
print(f"MAE : {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R2  : {r2_score(y_test, y_pred):.3f}")

# Predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=2, alpha=0.1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.title("Predicted vs Actual Price")
plt.xlabel("Actual price (SGD)")
plt.ylabel("Predicted price (SGD)")
plt.show()

# confusion matrix
labels = ["Low", "Mid", "High"]
actual_tier, edges = pd.qcut(y_test, 3, labels=labels, retbins=True)
edges[0], edges[-1] = -np.inf, np.inf        # catch predictions outside the test range
pred_tier = pd.cut(y_pred, bins=edges, labels=labels)

cm = confusion_matrix(actual_tier, pred_tier, labels=labels)
print("\nConfusion matrix (price tiers):")
print(cm)
print(f"Tier accuracy: {np.diag(cm).sum() / cm.sum():.3f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (price tiers)")
plt.xlabel("Predicted tier")
plt.ylabel("Actual tier")
plt.show()
