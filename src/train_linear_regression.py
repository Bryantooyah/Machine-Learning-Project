"""Linear regression baseline for HDB resale flat prices (Jan-2017 onwards).

Predicts `resale_price` from flat attributes using a scikit-learn Pipeline:

    raw CSV -> feature engineering -> ColumnTransformer -> LinearRegression

Everything is wired end to end so you can start swapping pieces immediately.
The places you are most likely to change are marked "CUSTOMISE".

Usage:
    python src/train_linear_regression.py
    python src/train_linear_regression.py --sample 20000        # fast iteration
    python src/train_linear_regression.py --split time          # forward-in-time test set
    python src/train_linear_regression.py --log-target          # model log(price)
    python src/train_linear_regression.py --cv 5 --plots --save-model
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "resale_flat_prices_2017_onwards.csv"

RANDOM_STATE = 42

# CUSTOMISE: which engineered columns feed the model.
NUMERIC_FEATURES = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_years",
    "flat_age_years",
    "months_since_start",
]
CATEGORICAL_FEATURES = [
    "town",
    "flat_type",
    "flat_model",
]
TARGET = "resale_price"


# --------------------------------------------------------------------------
# 1. Load + feature engineering
# --------------------------------------------------------------------------

def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw data.gov.sg CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {csv_path.name}")
    return df


def parse_storey_midpoint(storey_range: pd.Series) -> pd.Series:
    """'10 TO 12' -> 11.0 (the bucket midpoint)."""
    bounds = storey_range.str.extract(r"(\d+)\s*TO\s*(\d+)").astype(float)
    return bounds.mean(axis=1)


def parse_remaining_lease(remaining_lease: pd.Series) -> pd.Series:
    """'61 years 04 months' -> 61.333 years. Handles the 'NN years' form too."""
    years = remaining_lease.str.extract(r"(\d+)\s*year", flags=re.IGNORECASE)[0].astype(float)
    months = (
        remaining_lease.str.extract(r"(\d+)\s*month", flags=re.IGNORECASE)[0]
        .astype(float)
        .fillna(0.0)
    )
    return years + months / 12.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw columns into model-ready numeric/categorical features.

    CUSTOMISE: add your own features here (e.g. price per sqm of the town in
    the prior month, distance to MRT, block-level aggregates).
    """
    out = df.copy()

    # 'month' is a 'YYYY-MM' string -> a real timestamp we can do arithmetic on.
    out["transaction_date"] = pd.to_datetime(out["month"], format="%Y-%m")
    out["transaction_year"] = out["transaction_date"].dt.year
    out["transaction_month"] = out["transaction_date"].dt.month

    # Linear time trend: months elapsed since the first transaction in the data.
    # This lets the model absorb the general market drift over 2017 onwards.
    start = out["transaction_date"].min()
    out["months_since_start"] = (
        (out["transaction_date"].dt.year - start.year) * 12
        + (out["transaction_date"].dt.month - start.month)
    )

    out["storey_mid"] = parse_storey_midpoint(out["storey_range"])
    out["remaining_lease_years"] = parse_remaining_lease(out["remaining_lease"])
    out["flat_age_years"] = out["transaction_year"] - out["lease_commence_date"]

    # Drop rows where an engineered feature or the target failed to parse.
    needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    before = len(out)
    out = out.dropna(subset=needed)
    if before != len(out):
        print(f"Dropped {before - len(out):,} rows with missing/unparseable values")

    return out


# --------------------------------------------------------------------------
# 2. Model
# --------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Preprocessing + linear regression as a single estimator.

    Keeping preprocessing inside the Pipeline means the scaler and encoder are
    fitted on training folds only -- no leakage into validation/test.

    CUSTOMISE: swap LinearRegression for Ridge/Lasso/ElasticNet, or add
    PolynomialFeatures to the numeric branch.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def split_data(df: pd.DataFrame, how: str, test_size: float):
    """Random split (default) or a forward-in-time split.

    A time split is the honest test if you care about predicting *future*
    prices: train on the earlier months, test on the most recent ones.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    if how == "time":
        df_sorted = df.sort_values("transaction_date")
        cutoff = int(len(df_sorted) * (1 - test_size))
        train_idx = df_sorted.index[:cutoff]
        test_idx = df_sorted.index[cutoff:]
        boundary = df_sorted["transaction_date"].iloc[cutoff].strftime("%Y-%m")
        print(f"Time split at {boundary} (test = transactions from then on)")
        return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------

def report_metrics(name: str, y_true, y_pred) -> dict:
    metrics = {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        # Mean absolute percentage error -- easier to read than raw dollars.
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }
    print(
        f"{name:<8} RMSE ${metrics['rmse']:>10,.0f}   "
        f"MAE ${metrics['mae']:>10,.0f}   "
        f"R2 {metrics['r2']:>6.4f}   "
        f"MAPE {metrics['mape']:>5.2f}%"
    )
    return metrics


def show_coefficients(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Largest-magnitude coefficients, in dollars per (scaled) unit."""
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_

    table = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} coefficients (intercept = {pipeline.named_steps['model'].intercept_:,.0f}):")
    print(table.head(top_n).to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    return table


def make_plots(y_test, y_pred, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # write files without needing a display
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(y_test, y_pred, s=4, alpha=0.15, edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set(xlabel="Actual price", ylabel="Predicted price", title="Predicted vs actual")

    axes[1].scatter(y_pred, residuals, s=4, alpha=0.15, edgecolors="none")
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].set(xlabel="Predicted price", ylabel="Residual", title="Residuals vs fitted")

    axes[2].hist(residuals, bins=80)
    axes[2].axvline(0, color="r", ls="--", lw=1)
    axes[2].set(xlabel="Residual", ylabel="Count", title="Residual distribution")

    fig.tight_layout()
    path = out_dir / "diagnostics.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved plots to {path}")


# --------------------------------------------------------------------------
# 4. Entry point
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to the raw CSV")
    parser.add_argument("--sample", type=int, default=None, help="Use a random N-row subset (fast iteration)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction (default 0.2)")
    parser.add_argument("--split", choices=["random", "time"], default="random", help="Hold-out strategy")
    parser.add_argument("--log-target", action="store_true", help="Fit on log(price); metrics still in dollars")
    parser.add_argument("--cv", type=int, default=0, help="K-fold CV on the training set (0 = skip)")
    parser.add_argument("--plots", action="store_true", help="Write diagnostic plots to outputs/")
    parser.add_argument("--save-model", action="store_true", help="Persist the fitted pipeline to outputs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found at {args.csv}\nRun: python scripts/download_data.py")

    # --- data ------------------------------------------------------------
    df = load_raw(args.csv)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=RANDOM_STATE)
        print(f"Sampled {len(df):,} rows")

    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df, args.split, args.test_size)
    print(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows")

    # --- fit -------------------------------------------------------------
    pipeline = build_pipeline()

    # Prices are right-skewed; fitting log(price) often straightens out the
    # residual fan and turns the model into a multiplicative one.
    y_train_fit = np.log(y_train) if args.log_target else y_train

    print("\nFitting LinearRegression ...")
    pipeline.fit(X_train, y_train_fit)

    def predict(X):
        pred = pipeline.predict(X)
        return np.exp(pred) if args.log_target else pred

    # --- evaluate --------------------------------------------------------
    print()
    report_metrics("train", y_train, predict(X_train))
    test_metrics = report_metrics("test", y_test, predict(X_test))

    if args.cv > 0:
        cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            pipeline, X_train, y_train_fit, cv=cv, scoring="r2", n_jobs=-1
        )
        print(f"\n{args.cv}-fold CV R2: {scores.mean():.4f} +/- {scores.std():.4f}  {np.round(scores, 4)}")

    show_coefficients(pipeline)

    # A naive reference point: always predict the training mean. If the model
    # is not comfortably better than this, something is wrong.
    baseline_rmse = root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))
    print(
        f"\nBaseline (predict train mean) RMSE ${baseline_rmse:,.0f} "
        f"-> model is {(1 - test_metrics['rmse'] / baseline_rmse) * 100:.1f}% better"
    )

    # --- artefacts -------------------------------------------------------
    out_dir = PROJECT_ROOT / "outputs"
    if args.plots:
        make_plots(y_test, predict(X_test), out_dir)

    if args.save_model:
        import joblib

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "linear_regression.joblib"
        joblib.dump(pipeline, path)
        print(f"Saved fitted pipeline to {path}")


if __name__ == "__main__":
    main()
