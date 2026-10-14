
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SalesIQ | Sales Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263449;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(110deg, #172554, #0f766e);
        border: 1px solid #334155;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        color: white;
        font-size: 2.2rem;
    }

    .hero p {
        color: #dbeafe;
        margin-top: 8px;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #172033;
        border: 1px solid #29364b;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #5eead4;
    }

    .section-note {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    div.stButton > button {
        border-radius: 9px;
        border: 1px solid #0d9488;
        background: #0f766e;
        color: white;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #115e59;
        border-color: #5eead4;
        color: white;
    }

    .stDownloadButton button {
        border-radius: 9px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>📊 SalesIQ</h1>
        <p>Sales Analysis, Forecasting and Model Evaluation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
if "sales_data" not in st.session_state:
    st.session_state.sales_data = None

if "source_name" not in st.session_state:
    st.session_state.source_name = "Sample data"

if "date_column" not in st.session_state:
    st.session_state.date_column = None

if "sales_column" not in st.session_state:
    st.session_state.sales_column = None


# ============================================================
# SAMPLE DATA
# ============================================================
def create_sample_data():
    """Create a sample sales dataset for demonstration."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(
        start="2023-01-01",
        periods=36,
        freq="MS",
    )

    trend = np.linspace(8000, 16000, len(dates))
    seasonality = 1800 * np.sin(
        2 * np.pi * np.arange(len(dates)) / 12
    )
    noise = rng.normal(0, 700, len(dates))

    sales = np.maximum(trend + seasonality + noise, 100)

    return pd.DataFrame(
        {
            "Date": dates,
            "Sales": np.round(sales, 2),
            "Orders": rng.integers(100, 400, len(dates)),
            "Marketing_Spend": np.round(
                rng.uniform(500, 2500, len(dates)), 2
            ),
        }
    )


# ============================================================
# DATA CLEANING
# ============================================================
def clean_sales_data(raw_df, date_col, sales_col):
    """
    Validate and clean the selected date and sales columns.
    Return one record per date, with sales aggregated by date.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The uploaded file has no data.")

    if date_col not in raw_df.columns:
        raise ValueError("Select a valid date column.")

    if sales_col not in raw_df.columns:
        raise ValueError("Select a valid sales column.")

    if date_col == sales_col:
        raise ValueError("Date and sales columns must be different.")

    df = raw_df[[date_col, sales_col]].copy()
    df.columns = ["Date", "Sales"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Date", "Sales"])

    if df.empty:
        raise ValueError(
            "No valid rows remain. Check the selected date and sales columns."
        )

    # Aggregate multiple records occurring on the same date.
    df = (
        df.groupby("Date", as_index=False)["Sales"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def make_time_features(dates):
    """Create numerical time features for regression models."""
    dates = pd.to_datetime(pd.Series(dates))

    features = pd.DataFrame()
    features["time_index"] = np.arange(len(dates))
    features["year"] = dates.dt.year.to_numpy()
    features["month"] = dates.dt.month.to_numpy()
    features["quarter"] = dates.dt.quarter.to_numpy()
    features["day_of_year"] = dates.dt.dayofyear.to_numpy()

    return features


def infer_frequency(dates):
    """Infer a likely forecasting frequency from observed dates."""
    dates = pd.to_datetime(pd.Series(dates)).sort_values()

    if len(dates) < 3:
        return "MS"

    try:
        inferred = pd.infer_freq(dates)
    except (ValueError, TypeError):
        inferred = None

    if inferred:
        return inferred

    median_gap = dates.diff().dropna().dt.days.median()

    if median_gap <= 2:
        return "D"
    if median_gap <= 8:
        return "W"
    if median_gap <= 20:
        return "MS"
    if median_gap <= 45:
        return "MS"
    if median_gap <= 100:
        return "QS"

    return "YS"


def create_future_dates(dates, periods):
    """Generate future dates using the inferred data frequency."""
    dates = pd.to_datetime(pd.Series(dates)).sort_values()
    frequency = infer_frequency(dates)
    last_date = dates.iloc[-1]

    try:
        future = pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=frequency,
        )[1:]
    except (ValueError, TypeError):
        future = pd.date_range(
            start=last_date + pd.Timedelta(days=30),
            periods=periods,
            freq="MS",
        )

    return future, frequency


# ============================================================
# FORECASTING
# ============================================================
def forecast_sales(df, model_name, periods):
    """
    Fit a regression model using date-based features and predict
    the requested number of future periods.
    """
    if len(df) < 4:
        raise ValueError(
            "At least four valid dated sales records are required."
        )

    df = df.sort_values("Date").reset_index(drop=True)

    X = make_time_features(df["Date"])
    y = df["Sales"].astype(float).to_numpy()

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=6,
        )
    else:
        model = LinearRegression()

    model.fit(X, y)

    future_dates, frequency = create_future_dates(
        df["Date"], periods
    )

    future_X = pd.DataFrame()
    future_X["time_index"] = np.arange(
        len(df), len(df) + periods
    )
    future_X["year"] = future_dates.year
    future_X["month"] = future_dates.month
    future_X["quarter"] = future_dates.quarter
    future_X["day_of_year"] = future_dates.dayofyear

    predictions = model.predict(future_X)
    predictions = np.maximum(predictions, 0)

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Forecast Sales": predictions,
        }
    )

    return forecast_df, model, frequency


# ============================================================
# MODEL EVALUATION
# ============================================================
def evaluate_model(df, model_name):
    """
    Use a chronological holdout split. The last 20% of records
    are held out for evaluation; the model only sees earlier records.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            "At least five records are required for evaluation."
        )

    split_index = int(len(df) * 0.8)
    split_index = max(2, min(split_index, len(df) - 1))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = make_time_features(train_df["Date"])
    X_test = make_time_features(test_df["Date"])

    # Preserve the time-index position of the held-out records.
    X_test["time_index"] = np.arange(
        len(train_df), len(train_df) + len(test_df)
    )

    y_train = train_df["Sales"].astype(float).to_numpy()
    y_test = test_df["Sales"].astype(float).to_numpy()

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=6,
        )
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    if len(y_test) >= 2 and np.unique(y_test).size > 1:
        r2 = r2_score(y_test, predictions)
    else:
        r2 = np.nan

    results = pd.DataFrame(
        {
            "Date": test_df["Date"].to_numpy(),
            "Actual Sales": y_test,
            "Predicted Sales": predictions,
            "Absolute Error": np.abs(y_test - predictions),
        }
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Training Records": len(train_df),
        "Testing Records": len(test_df),
    }

    return results, metrics, model


# ============================================================
# CHART HELPERS
# ============================================================
def style_axes(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, color="#f8fafc", fontsize=13, pad=14)
    ax.set_xlabel(xlabel or "", color="#cbd5e1")
    ax.set_ylabel(ylabel or "", color="#cbd5e1")
    ax.tick_params(colors="#cbd5e1")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_facecolor("#172033")


def finish_figure(fig):
    fig.patch.set_facecolor("#111827")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# SIDEBAR — DATA SOURCE AND NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## 📊 SalesIQ")
    st.caption("Sales intelligence workspace")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload sales CSV",
        type=["csv"],
        help="Your file should contain a date column and a numeric sales column.",
    )

    raw_data = None
    source_name = "Sample data"

    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
            source_name = uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

    if raw_data is None:
        raw_data = create_sample_data()

    if raw_data is not None and not raw_data.empty:
        columns = raw_data.columns.tolist()

        # Guess the date column.
        date_candidates = [
            col for col in columns
            if any(word in str(col).lower()
                   for word in ["date", "time", "day"])
        ]

        date_default = (
            columns.index(date_candidates[0])
            if date_candidates
            else 0
        )

        # Guess a sales column.
        sales_candidates = [
            col for col in columns
            if any(word in str(col).lower()
                   for word in ["sales", "revenue", "amount", "turnover"])
        ]

        if not sales_candidates:
            numeric_candidates = raw_data.select_dtypes(
                include=np.number
            ).columns.tolist()
            sales_candidates = numeric_candidates

        sales_default = (
            columns.index(sales_candidates[0])
            if sales_candidates and sales_candidates[0] in columns
            else min(1, len(columns) - 1)
        )

        st.markdown("### Dataset columns")

        date_column = st.selectbox(
            "Date column",
            options=columns,
            index=date_default,
        )

        sales_column = st.selectbox(
            "Sales column",
            options=columns,
            index=sales_default,
        )

        try:
            cleaned_data = clean_sales_data(
                raw_data, date_column, sales_column
            )
            st.session_state.sales_data = cleaned_data
            st.session_state.source_name = source_name
            st.session_state.date_column = date_column
            st.session_state.sales_column = sales_column
        except ValueError as exc:
            st.session_state.sales_data = None
            st.error(str(exc))

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Sales Analysis",
            "Forecast",
            "Actual vs Predicted",
            "Model Evaluation",
        ],
        index=0,
    )

    st.divider()
    st.caption("SalesIQ • Analytics & Forecasting")


# ============================================================
# VALIDATE DATA
# ============================================================
data = st.session_state.sales_data

if data is None or data.empty:
    st.error(
        "No usable sales data is available. Upload a CSV with valid "
        "dates and numeric sales values, then choose the correct columns."
    )
    st.stop()

data = data.sort_values("Date").reset_index(drop=True)

total_sales = float(data["Sales"].sum())
average_sales = float(data["Sales"].mean())
highest_sales = float(data["Sales"].max())
lowest_sales = float(data["Sales"].min())
record_count = len(data)

# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================
if page == "Dashboard":
    st.title("Dashboard")
    st.markdown(
        '<p class="section-note">A quick overview of your sales performance.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"{total_sales:,.2f}")
    col2.metric("Average Sales", f"{average_sales:,.2f}")
    col3.metric("Highest Sales", f"{highest_sales:,.2f}")
    col4.metric("Data Records", f"{record_count:,}")

    st.divider()

    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Sales Trend")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            data["Date"],
            data["Sales"],
            marker="o",
            linewidth=2.2,
            color="#2dd4bf",
            label="Sales",
        )
        style_axes(
            ax,
            "Sales Over Time",
            "Date",
            "Sales",
        )
        ax.legend(facecolor="#172033", labelcolor="#f8fafc")
        finish_figure(fig)

    with right:
        st.subheader("Sales Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            data["Sales"],
            bins=min(12, max(4, len(data) // 2)),
            color="#38bdf8",
            edgecolor="#0f172a",
            alpha=0.9,
        )
        style_axes(ax, "Sales Frequency", "Sales", "Count")
        finish_figure(fig)

    st.subheader("Recent Sales Records")
    st.dataframe(
        data.tail(10).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download cleaned sales data",
        data.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_sales_data.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE 2 — SALES ANALYSIS
# ============================================================
elif page == "Sales Analysis":
    st.title("Sales Analysis")
    st.markdown(
        '<p class="section-note">Explore sales trends, distributions, and summary statistics.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum Sales", f"{lowest_sales:,.2f}")
    col2.metric("Median Sales", f"{data['Sales'].median():,.2f}")
    col3.metric("Sales Standard Deviation", f"{data['Sales'].std(ddof=0):,.2f}")

    st.divider()

    st.subheader("Sales Over Time")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        data["Date"],
        data["Sales"],
        color="#2dd4bf",
        marker="o",
        linewidth=2,
    )
    ax.fill_between(
        data["Date"],
        data["Sales"].to_numpy(),
        alpha=0.12,
        color="#2dd4bf",
    )
    style_axes(ax, "Sales Trend", "Date", "Sales")
    finish_figure(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Sales Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            data["Sales"],
            bins=min(12, max(4, len(data) // 2)),
            color="#818cf8",
            edgecolor="#111827",
        )
        style_axes(ax, "Distribution of Sales", "Sales", "Frequency")
        finish_figure(fig)

    with right:
        st.subheader("Monthly Sales Pattern")
        monthly = data.copy()
        monthly["Month"] = monthly["Date"].dt.month
        monthly_summary = monthly.groupby("Month")["Sales"].mean()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            monthly_summary.index,
            monthly_summary.values,
            color="#f59e0b",
            alpha=0.9,
        )
        ax.set_xticks(range(1, 13))
        style_axes(ax, "Average Sales by Calendar Month", "Month", "Average Sales")
        finish_figure(fig)

    st.subheader("Numeric Summary")
    st.dataframe(
        data[["Sales"]].describe().T,
        use_container_width=True,
    )

    # Correlation heatmap using Matplotlib only.
    st.subheader("Correlation Heatmap")

    numeric_data = data.select_dtypes(include=np.number)

    if numeric_data.shape[1] >= 1:
        correlation = numeric_data.corr()

        fig, ax = plt.subplots(
            figsize=(max(6, len(correlation.columns) * 1.1),
                     max(4, len(correlation.index) * 0.8))
        )

        values = correlation.to_numpy()
        image = ax.imshow(
            values,
            cmap="Spectral",
            vmin=-1,
            vmax=1,
            aspect="auto",
        )

        ax.set_xticks(np.arange(len(correlation.columns)))
        ax.set_yticks(np.arange(len(correlation.index)))
        ax.set_xticklabels(
            correlation.columns, rotation=45, ha="right",
            color="#cbd5e1",
        )
        ax.set_yticklabels(
            correlation.index, color="#cbd5e1"
        )

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if np.isfinite(values[row, col]):
                    ax.text(
                        col,
                        row,
                        f"{values[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color="#111827",
                        fontsize=9,
                    )

        ax.set_title(
            "Numeric Feature Correlation",
            color="#f8fafc",
            pad=14,
        )
        fig.colorbar(image, ax=ax, label="Correlation")
        finish_figure(fig)
    else:
        st.info("No numeric columns are available for a correlation heatmap.")


# ============================================================
# PAGE 3 — FORECAST
# ============================================================
elif page == "Forecast":
    st.title("Sales Forecast")
    st.markdown(
        '<p class="section-note">Estimate future sales from historical observations.</p>',
        unsafe_allow_html=True,
    )

    if len(data) < 4:
        st.warning(
            "Upload at least four valid dated sales records to generate a forecast."
        )
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox(
            "Forecasting model",
            ["Random Forest", "Linear Regression"],
            key="forecast_model",
        )

    with col2:
        periods = st.slider(
            "Future periods to forecast",
            min_value=1,
            max_value=24,
            value=6,
            help="The forecast frequency is inferred from the dates in your dataset.",
        )

    if st.button("Generate Forecast", type="primary"):
        try:
            forecast_df, fitted_model, frequency = forecast_sales(
                data, model_name, periods
            )

            st.session_state.forecast_df = forecast_df
            st.session_state.forecast_model_name = model_name
            st.session_state.forecast_frequency = frequency

        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")

    if "forecast_df" in st.session_state:
        forecast_df = st.session_state.forecast_df

        st.subheader("Forecast Results")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            data["Date"],
            data["Sales"],
            color="#2dd4bf",
            marker="o",
            linewidth=2,
            label="Historical Sales",
        )
        ax.plot(
            forecast_df["Date"],
            forecast_df["Forecast Sales"],
            color="#f59e0b",
            marker="o",
            linestyle="--",
            linewidth=2.2,
            label="Forecast Sales",
        )

        if len(data) > 0:
            ax.axvline(
                data["Date"].max(),
                color="#94a3b8",
                linestyle=":",
                alpha=0.8,
            )

        style_axes(ax, "Historical Sales and Forecast", "Date", "Sales")
        ax.legend(facecolor="#172033", labelcolor="#f8fafc")
        finish_figure(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Model", st.session_state.forecast_model_name)
        c2.metric(
            "Average Forecast",
            f"{forecast_df['Forecast Sales'].mean():,.2f}",
        )
        c3.metric(
            "Total Forecast",
            f"{forecast_df['Forecast Sales'].sum():,.2f}",
        )

        st.dataframe(
            forecast_df.round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download forecast CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv",
        )

        st.caption(
            "Forecasts are estimates based on date-related patterns in historical data. "
            "Actual sales may differ."
        )
    else:
        st.info("Choose a model and click **Generate Forecast** to begin.")


# ============================================================
# PAGE 4 — ACTUAL VS PREDICTED
# ============================================================
elif page == "Actual vs Predicted":
    st.title("Actual vs Predicted")
    st.markdown(
        '<p class="section-note">Compare model predictions with unseen historical records.</p>',
        unsafe_allow_html=True,
    )

    if len(data) < 5:
        st.warning(
            "At least five valid dated sales records are required for this comparison."
        )
        st.stop()

    model_name = st.selectbox(
        "Select model",
        ["Random Forest", "Linear Regression"],
        key="comparison_model",
    )

    if st.button("Compare Actual and Predicted", type="primary"):
        try:
            results, metrics, fitted_model = evaluate_model(
                data, model_name
            )
            st.session_state.comparison_results = results
            st.session_state.comparison_metrics = metrics
            st.session_state.comparison_model_name = model_name
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    if "comparison_results" in st.session_state:
        results = st.session_state.comparison_results

        st.subheader(
            f"Holdout Comparison — {st.session_state.comparison_model_name}"
        )

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            results["Date"],
            results["Actual Sales"],
            marker="o",
            color="#2dd4bf",
            linewidth=2,
            label="Actual Sales",
        )
        ax.plot(
            results["Date"],
            results["Predicted Sales"],
            marker="s",
            color="#f59e0b",
            linewidth=2,
            linestyle="--",
            label="Predicted Sales",
        )
        style_axes(ax, "Actual vs Predicted Sales", "Date", "Sales")
        ax.legend(facecolor="#172033", labelcolor="#f8fafc")
        finish_figure(fig)

        st.dataframe(
            results.round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download comparison CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="actual_vs_predicted.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "Select a model and click **Compare Actual and Predicted**."
        )


# ============================================================
# PAGE 5 — MODEL EVALUATION
# ============================================================
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.markdown(
        '<p class="section-note">Evaluate forecasting error using a chronological holdout test.</p>',
        unsafe_allow_html=True,
    )

    if len(data) < 5:
        st.warning(
            "At least five valid dated sales records are required for model evaluation."
        )
        st.stop()

    st.info(
        "The model trains on the earlier 80% of the data and is tested on "
        "the later 20%. This helps avoid using future observations during training."
    )

    selected_models = st.multiselect(
        "Models to evaluate",
        ["Linear Regression", "Random Forest"],
        default=["Linear Regression", "Random Forest"],
    )

    if st.button("Evaluate Models", type="primary"):
        metric_rows = []
        result_sets = {}

        for name in selected_models:
            try:
                results, metrics, fitted_model = evaluate_model(data, name)

                metric_rows.append(
                    {
                        "Model": name,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "R²": metrics["R²"],
                        "Training Records": metrics["Training Records"],
                        "Testing Records": metrics["Testing Records"],
                    }
                )

                result_sets[name] = results

            except Exception as exc:
                st.error(f"{name} evaluation failed: {exc}")

        if metric_rows:
            st.session_state.evaluation_metrics_df = pd.DataFrame(
                metric_rows
            )
            st.session_state.evaluation_results = result_sets

    if "evaluation_metrics_df" in st.session_state:
        metrics_df = st.session_state.evaluation_metrics_df

        st.subheader("Evaluation Metrics")
        st.dataframe(
            metrics_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "MAE and RMSE measure prediction error; lower values are generally better. "
            "R² measures how well the model explains variation in the test set and "
            "can be negative when predictions perform poorly."
        )

        if not metrics_df.empty:
            best_index = metrics_df["MAE"].idxmin()
            best_model = metrics_df.loc[best_index, "Model"]
            best_mae = metrics_df.loc[best_index, "MAE"]

            c1, c2 = st.columns(2)
            c1.metric("Lowest MAE Model", best_model)
            c2.metric("Lowest MAE", f"{best_mae:,.3f}")

            st.subheader("MAE Comparison")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                metrics_df["Model"],
                metrics_df["MAE"],
                color=["#38bdf8", "#2dd4bf"][:len(metrics_df)],
            )
            style_axes(ax, "Mean Absolute Error by Model", "Model", "MAE")
            finish_figure(fig)

        st.download_button(
            "⬇️ Download evaluation metrics",
            metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="model_evaluation_metrics.csv",
            mime="text/csv",
        )
    else:
        st.info("Select your models and click **Evaluate Models**.")
