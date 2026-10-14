
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train.csv"

FORECAST_FILE = BASE_DIR / "sales_forecast.csv"
DETAIL_FILE = BASE_DIR / "sales_forecast_by_store_family.csv"
EVALUATION_FILE = BASE_DIR / "model_evaluation.csv"
TEST_PREDICTIONS_FILE = BASE_DIR / "sales_test_predictions.csv"

TREND_CHART = BASE_DIR / "sales_trend.png"
FORECAST_CHART = BASE_DIR / "sales_forecast.png"

FORECAST_MONTHS = 3
RANDOM_STATE = 42


# =========================================================
# 2. LOAD AND CLEAN DATA
# =========================================================

def load_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    print("Loading Kaggle training data...")

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=[
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
        ],
        parse_dates=["date"],
    )

    df["sales"] = pd.to_numeric(
        df["sales"], errors="coerce"
    )

    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"], errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    )

    df["sales"] = df["sales"].clip(lower=0)
    df["onpromotion"] = df["onpromotion"].clip(lower=0)

    print(f"Daily rows loaded: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")

    return df


# =========================================================
# 3. AGGREGATE TO MONTHLY SALES
# =========================================================

def create_monthly_data(df):
    print("\nAggregating daily sales into monthly totals...")

    df["Date"] = (
        df["date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        df.groupby(
            ["Date", "store_nbr", "family"],
            as_index=False,
            observed=True,
        )
        .agg(
            Sales=("sales", "sum"),
            OnPromotion=("onpromotion", "mean"),
        )
    )

    monthly = monthly.sort_values(
        ["store_nbr", "family", "Date"]
    ).reset_index(drop=True)

    print(f"Monthly store-family rows: {len(monthly):,}")

    return monthly


# =========================================================
# 4. CREATE TIME-SERIES FEATURES
# =========================================================

def create_features(monthly):
    print("\nCreating lag and rolling features...")

    data = monthly.copy()

    grouped_sales = data.groupby(
        ["store_nbr", "family"],
        observed=True,
        sort=False,
    )["Sales"]

    for lag in [1, 2, 3, 12]:
        data[f"lag_{lag}"] = grouped_sales.shift(lag)

    data["rolling_mean_3"] = (
        data.groupby(
            ["store_nbr", "family"],
            observed=True,
            sort=False,
        )["Sales"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=3, min_periods=3
            ).mean()
        )
    )

    data["month_number"] = data["Date"].dt.month
    data["year_number"] = data["Date"].dt.year

    # Keep family as a column while creating model features.
    data = pd.get_dummies(
        data,
        columns=["family"],
        prefix="family",
        dtype=np.uint8,
    )

    family_columns = [
        col for col in data.columns
        if col.startswith("family_")
    ]

    feature_columns = [
        "store_nbr",
        "month_number",
        "year_number",
        "OnPromotion",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3",
    ] + family_columns

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_12",
            "rolling_mean_3",
        ]
    ).copy()

    return data, feature_columns, family_columns


# =========================================================
# 5. CHRONOLOGICAL MODEL EVALUATION
# =========================================================

def evaluate_models(data, feature_columns):
    print("\nPreparing time-based train/test split...")

    latest_date = data["Date"].max()
    test_start = latest_date - pd.DateOffset(months=2)

    train_data = data[data["Date"] < test_start].copy()
    test_data = data[data["Date"] >= test_start].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "Could not create the time-based train/test split."
        )

    X_train = train_data[feature_columns]
    y_train = train_data["Sales"]

    X_test = test_data[feature_columns]
    y_test = test_data["Sales"]

    print(f"Training rows: {len(train_data):,}")
    print(f"Testing rows: {len(test_data):,}")
    print(f"Test period begins: {test_start.date()}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=40,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        pred = np.maximum(model.predict(X_test), 0)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "Model": name,
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
        })

        predictions[name] = pred

        print(f"MAE:  {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")

    evaluation = pd.DataFrame(results).sort_values(
        "RMSE"
    ).reset_index(drop=True)

    evaluation.to_csv(EVALUATION_FILE, index=False)

    best_name = evaluation.iloc[0]["Model"]
    print(f"\nSelected model by holdout RMSE: {best_name}")

    # Save actual versus predicted test records.
    test_output = test_data[
        ["Date", "store_nbr", "Sales"]
    ].copy()

    test_output = test_output.rename(
        columns={"Sales": "Actual_Sales"}
    )

    test_output["Predicted_Sales"] = predictions[best_name]
    test_output["Absolute_Error"] = abs(
        test_output["Actual_Sales"]
        - test_output["Predicted_Sales"]
    )

    test_output.to_csv(
        TEST_PREDICTIONS_FILE, index=False
    )

    return best_name, evaluation


# =========================================================
# 6. CREATE FORECAST MODEL
# =========================================================

def train_final_model(data, feature_columns, best_name):
    print("\nTraining selected model on all available history...")

    if best_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=40,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        model = LinearRegression()

    model.fit(
        data[feature_columns],
        data["Sales"],
    )

    return model


# =========================================================
# 7. FORECAST NEXT THREE MONTHS
# =========================================================

def forecast_future(monthly, model, feature_columns, family_columns):
    print("\nForecasting future sales by store and family...")

    latest_date = monthly["Date"].max()

    families = sorted(monthly["family"].unique())
    stores = sorted(monthly["store_nbr"].unique())

    # Store each historical monthly sales series separately.
    histories = {}

    for (store, family), part in monthly.groupby(
        ["store_nbr", "family"],
        observed=True,
        sort=False,
    ):
        histories[(store, family)] = (
            part.sort_values("Date")["Sales"]
            .astype(float)
            .tolist()
        )

    forecasts = []

    for step in range(1, FORECAST_MONTHS + 1):
        future_date = (
            latest_date + pd.DateOffset(months=step)
        )

        rows = []
        keys = []

        for store in stores:
            for family in families:
                key = (store, family)
                history = histories.get(key, [])

                # Lag-12 needs at least twelve observations.
                if len(history) < 12:
                    continue

                row = {
                    "store_nbr": store,
                    "month_number": future_date.month,
                    "year_number": future_date.year,
                    # Future promotions are unknown.
                    # Zero is a baseline assumption.
                    "OnPromotion": 0,
                    "lag_1": history[-1],
                    "lag_2": history[-2],
                    "lag_3": history[-3],
                    "lag_12": history[-12],
                    "rolling_mean_3": float(
                        np.mean(history[-3:])
                    ),
                }

                for col in family_columns:
                    row[col] = int(
                        col == f"family_{family}"
                    )

                rows.append(row)
                keys.append(key)

        if not rows:
            raise ValueError(
                "No eligible store-family series to forecast."
            )

        future_features = pd.DataFrame(rows)

        # Ensure model features exist in the correct order.
        future_features = future_features.reindex(
            columns=feature_columns,
            fill_value=0,
        )

        predicted = np.maximum(
            model.predict(future_features), 0
        )

        for key, value in zip(keys, predicted):
            histories[key].append(float(value))

            forecasts.append({
                "Date": future_date,
                "store_nbr": key[0],
                "family": key[1],
                "Forecast_Sales": round(float(value), 2),
            })

        print(
            f"Completed forecast for {future_date.strftime('%Y-%m')}"
        )

    detailed = pd.DataFrame(forecasts)
    detailed.to_csv(DETAIL_FILE, index=False)

    summary = (
        detailed.groupby("Date", as_index=False)
        .agg(Sales=("Forecast_Sales", "sum"))
    )

    summary["Sales"] = summary["Sales"].round(2)
    summary.to_csv(FORECAST_FILE, index=False)

    return summary, detailed


# =========================================================
# 8. SAVE CHARTS
# =========================================================

def save_charts(monthly, summary):
    print("\nSaving charts...")

    historical = (
        monthly.groupby("Date", as_index=False)
        .agg(Sales=("Sales", "sum"))
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        historical["Date"],
        historical["Sales"],
        linewidth=1.5,
    )
    plt.title("Historical Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(TREND_CHART, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(
        historical["Date"],
        historical["Sales"],
        label="Historical Sales",
        linewidth=1.5,
    )
    plt.plot(
        summary["Date"],
        summary["Sales"],
        label="Forecast Sales",
        marker="o",
        linewidth=2,
    )
    plt.title("Sales Forecast — Next Three Months")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FORECAST_CHART, dpi=150)
    plt.close()


# =========================================================
# 9. MAIN PIPELINE
# =========================================================

def main():
    df = load_data()

    monthly = create_monthly_data(df)

    # Free memory before model training.
    del df

    data, feature_columns, family_columns = create_features(
        monthly
    )

    best_name, evaluation = evaluate_models(
        data, feature_columns
    )

    model = train_final_model(
        data, feature_columns, best_name
    )

    summary, detailed = forecast_future(
        monthly,
        model,
        feature_columns,
        family_columns,
    )

    save_charts(monthly, summary)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    print(evaluation.to_string(index=False))

    print("\nNEXT THREE MONTHS — TOTAL SALES FORECAST")
    print(summary.to_string(index=False))

    print("\nFiles saved successfully:")
    for path in [
        FORECAST_FILE,
        DETAIL_FILE,
        EVALUATION_FILE,
        TEST_PREDICTIONS_FILE,
        TREND_CHART,
        FORECAST_CHART,
    ]:
        print(f"- {path.name}")

    print("\nForecasting pipeline completed.")


if __name__ == "__main__":
    main()
