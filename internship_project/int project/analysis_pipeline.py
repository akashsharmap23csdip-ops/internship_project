from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder


DROP_COLUMNS = [
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Country",
    "City",
    "State",
    "Postal Code",
    "Product ID",
    "Product Name",
]

REQUIRED_COLUMNS = {
    "Order Date",
    "Ship Date",
    "Sales",
    "Ship Mode",
    "Segment",
    "Region",
    "Category",
    "Sub-Category",
}

OPTIONAL_COLUMNS_DEFAULTS = {
    "Quantity": 1,
    "Discount": 0.0,
    "Profit": 0.0,
}


def _read_csv(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path, encoding="latin-1", errors="ignore")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_list}.")


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    df = df.dropna(subset=["Order Date", "Ship Date", "Sales"])

    df_ml = df.drop(columns=DROP_COLUMNS, errors="ignore")

    df_ml["Sales"] = pd.to_numeric(df_ml["Sales"], errors="coerce")
    df_ml["Quantity"] = pd.to_numeric(df_ml["Quantity"], errors="coerce")
    df_ml["Discount"] = pd.to_numeric(df_ml["Discount"], errors="coerce")
    df_ml["Profit"] = pd.to_numeric(df_ml["Profit"], errors="coerce")

    df_ml["Order_Year"] = df_ml["Order Date"].dt.year
    df_ml["Order_Month"] = df_ml["Order Date"].dt.month
    df_ml["Order_DayOfWeek"] = df_ml["Order Date"].dt.dayofweek
    df_ml["Order_Quarter"] = df_ml["Order Date"].dt.quarter
    df_ml["Shipping_Days"] = (df_ml["Ship Date"] - df_ml["Order Date"]).dt.days

    df_ml = df_ml.drop(columns=["Order Date", "Ship Date"])

    cat_cols = df_ml.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df_ml[cat_cols] = df_ml[cat_cols].fillna("Unknown")

    for col in df_ml.columns:
        if df_ml[col].dtype.kind in {"i", "u", "f"}:
            if df_ml[col].isna().any():
                df_ml[col] = df_ml[col].fillna(df_ml[col].median())

    le_dict: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        df_ml[col] = encoder.fit_transform(df_ml[col])
        le_dict[col] = encoder

    df_ml = df_ml.dropna(subset=["Sales"])
    return df_ml


def _dataset_summary(df: pd.DataFrame) -> dict[str, str | int]:
    date_range = ""
    if "Order Date" in df.columns:
        dates = pd.to_datetime(df["Order Date"], errors="coerce")
        min_date = dates.min()
        max_date = dates.max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = f"{min_date.year}-{max_date.year}"

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_range": date_range,
    }


def _setup_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("viridis")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 12


def _save_and_close(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_eda_plots(df: pd.DataFrame, df_ml: pd.DataFrame, eda_dir: Path) -> None:
    _setup_plot_style()

    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["Sales"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].set_title("Sales Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sales ($)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(np.log1p(df["Sales"]), bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
    axes[1].set_title("Log-Transformed Sales Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Log(Sales)")
    axes[1].set_ylabel("Frequency")
    _save_and_close(fig, eda_dir / "sales_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    cat_sales.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Total Sales by Category", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Total Sales ($)")
    axes[0].tick_params(axis="x", rotation=0)

    cat_avg = df.groupby("Category")["Sales"].mean().sort_values(ascending=False)
    cat_avg.plot(kind="bar", ax=axes[1], color=colors, edgecolor="white")
    axes[1].set_title("Average Sales by Category", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Avg Sales ($)")
    axes[1].tick_params(axis="x", rotation=0)
    _save_and_close(fig, eda_dir / "sales_by_category.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=True)
    region_sales.plot(
        kind="barh",
        ax=ax,
        color=["#FF9800", "#E91E63", "#9C27B0", "#2196F3"],
    )
    ax.set_title("Total Sales by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    _save_and_close(fig, eda_dir / "sales_by_region.png")

    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Sales"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly.plot(ax=ax, color="#2196F3", linewidth=2, marker="o", markersize=3)
    ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    ax.set_xlabel("Month")
    _save_and_close(fig, eda_dir / "monthly_sales_trend.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df_ml.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save_and_close(fig, eda_dir / "correlation_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df["Discount"], df["Sales"], alpha=0.3, c="#E91E63", s=10)
    axes[0].set_title("Sales vs Discount", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Discount")
    axes[0].set_ylabel("Sales ($)")

    axes[1].scatter(df["Profit"], df["Sales"], alpha=0.3, c="#4CAF50", s=10)
    axes[1].set_title("Sales vs Profit", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Profit ($)")
    axes[1].set_ylabel("Sales ($)")
    _save_and_close(fig, eda_dir / "scatter_plots.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(
        column="Sales",
        by="Segment",
        ax=ax,
        patch_artist=True,
        boxprops=dict(facecolor="#42A5F5", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_title("Sales Distribution by Segment", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sales ($)")
    plt.suptitle("")
    _save_and_close(fig, eda_dir / "sales_by_segment.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    top_sub = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).tail(10)
    top_sub.plot(kind="barh", ax=ax, color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
    ax.set_title("Top 10 Sub-Categories by Total Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    _save_and_close(fig, eda_dir / "top_subcategories.png")

    quarterly = df.groupby([df["Order Date"].dt.year, df["Order Date"].dt.quarter])["Sales"].sum()
    quarterly.index = [f"{year}-Q{quarter}" for year, quarter in quarterly.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    quarterly.plot(kind="bar", ax=ax, color="#26A69A", edgecolor="white", alpha=0.85)
    ax.set_title("Quarterly Sales Trend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    plt.xticks(rotation=45)
    _save_and_close(fig, eda_dir / "quarterly_sales.png")


def _train_models(df_ml: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, object]]:
    X = df_ml.drop(columns=["Sales"])
    y = df_ml["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = math.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = math.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    tuned_pred = best_rf.predict(X_test)

    tuned_mae = mean_absolute_error(y_test, tuned_pred)
    tuned_rmse = math.sqrt(mean_squared_error(y_test, tuned_pred))
    tuned_r2 = r2_score(y_test, tuned_pred)

    results = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest (Default)",
                "Random Forest (Tuned)",
            ],
            "MAE": [lr_mae, rf_mae, tuned_mae],
            "RMSE": [lr_rmse, rf_rmse, tuned_rmse],
            "R2 Score": [lr_r2, rf_r2, tuned_r2],
        }
    )

    models = {
        "Linear Regression": lr_model,
        "Random Forest (Default)": rf_model,
        "Random Forest (Tuned)": best_rf,
    }
    cross_val_score(models["Linear Regression"], X, y, cv=5, scoring="r2", n_jobs=-1)
    cross_val_score(models["Random Forest (Default)"], X, y, cv=5, scoring="r2", n_jobs=-1)
    cross_val_score(models["Random Forest (Tuned)"], X, y, cv=5, scoring="r2", n_jobs=-1)

    artifacts = {
        "best_model": best_rf,
        "preds": [lr_pred, rf_pred, tuned_pred],
        "y_test": y_test,
        "features": X.columns.tolist(),
    }

    metrics = {
        "lr_mae": lr_mae,
        "lr_rmse": lr_rmse,
        "lr_r2": lr_r2,
        "rf_mae": rf_mae,
        "rf_rmse": rf_rmse,
        "rf_r2": rf_r2,
        "tuned_mae": tuned_mae,
        "tuned_rmse": tuned_rmse,
        "tuned_r2": tuned_r2,
    }

    return results, metrics, artifacts


def _make_model_plots(
    results: pd.DataFrame,
    artifacts: dict[str, object],
    model_dir: Path,
) -> None:
    _setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_list = results["Model"]
    colors = ["#E53935", "#43A047", "#1E88E5"]

    for i, metric in enumerate(["MAE", "RMSE", "R2 Score"]):
        axes[i].bar(models_list, results[metric], color=colors, edgecolor="white", alpha=0.85)
        axes[i].set_title(metric, fontsize=14, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=15)
        for j, value in enumerate(results[metric]):
            axes[i].text(j, value + value * 0.02, f"{value:.4f}", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)
    _save_and_close(fig, model_dir / "model_comparison.png")

    preds = artifacts["preds"]
    y_test = artifacts["y_test"]
    titles = ["Linear Regression", "Random Forest (Default)", "Random Forest (Tuned)"]
    colors_scatter = ["#E53935", "#43A047", "#1E88E5"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        axes[i].scatter(y_test, preds[i], alpha=0.3, s=10, c=colors_scatter[i])
        max_val = max(y_test.max(), max(pred.max() for pred in preds))
        axes[i].plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Perfect Prediction")
        axes[i].set_title(titles[i], fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Actual Sales ($)")
        axes[i].set_ylabel("Predicted Sales ($)")
        axes[i].legend()

    plt.suptitle("Actual vs Predicted Sales", fontsize=16, fontweight="bold", y=1.02)
    _save_and_close(fig, model_dir / "actual_vs_predicted.png")

    importances = artifacts["best_model"].feature_importances_
    features = artifacts["features"]
    feat_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(
        "Importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        feat_imp["Feature"],
        feat_imp["Importance"],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_imp))),
    )
    ax.set_title("Feature Importance (Tuned Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save_and_close(fig, model_dir / "feature_importance.png")

    lr_pred, rf_pred, tuned_pred = preds
    lr_residuals = y_test - lr_pred
    rf_residuals = y_test - tuned_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(lr_pred, lr_residuals, alpha=0.3, s=10, c="#E53935")
    axes[0].axhline(y=0, color="black", linestyle="--")
    axes[0].set_title("Linear Regression Residuals", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")

    axes[1].scatter(tuned_pred, rf_residuals, alpha=0.3, s=10, c="#1E88E5")
    axes[1].axhline(y=0, color="black", linestyle="--")
    axes[1].set_title("Tuned Random Forest Residuals", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    _save_and_close(fig, model_dir / "residual_analysis.png")


def run_analysis(csv_path: Path, output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    eda_dir = output_root / "eda_plots"
    model_dir = output_root / "model_results"
    eda_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv(csv_path)
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")
    if df.shape[0] < 50:
        raise ValueError("Dataset has too few rows for analysis (minimum 50).")

    for column, default_value in OPTIONAL_COLUMNS_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default_value
    _validate_columns(df)
    summary = _dataset_summary(df)
    df_ml = _prepare_dataframe(df)
    if df_ml.empty:
        raise ValueError("Dataset has no usable rows after preprocessing.")

    _make_eda_plots(df, df_ml, eda_dir)
    results, _metrics, artifacts = _train_models(df_ml)

    results.to_csv(model_dir / "model_comparison.csv", index=False)
    _make_model_plots(results, artifacts, model_dir)

    best_row = results.loc[results["R2 Score"].idxmax()].to_dict()

    return {
        "rows": results.to_dict(orient="records"),
        "best_model": best_row,
        "summary": summary,
    }
