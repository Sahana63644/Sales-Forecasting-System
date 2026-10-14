
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SalesIQ | Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette
NAVY = "#182747"
TEAL = "#00A6A6"
BLUE = "#4285F4"
PURPLE = "#845EC2"
ORANGE = "#FF9671"
PINK = "#D65DB1"
GREEN = "#2EAD73"
YELLOW = "#FFC75F"
RED = "#E85D75"
LIGHT_BLUE = "#6C9BCF"

CHART_COLORS = [
    TEAL, BLUE, PURPLE, ORANGE, PINK,
    GREEN, YELLOW, RED, LIGHT_BLUE, "#7DCEA0"
]


# ============================================================
# 2. DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F3F6FC;
        color: #182747;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14213D, #243B65);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 35px;
        font-weight: 850;
        color: #182747;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #68758B;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .section-title {
        color: #182747;
        font-size: 23px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #00A6A6;
        border-radius: 13px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 4px 14px rgba(24,39,71,0.06);
    }

    .metric-label {
        color: #69778C;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #182747;
        font-size: 24px;
        font-weight: 850;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #8792A5;
        font-size: 12px;
        margin-top: 5px;
    }

    div[data-testid="stDownloadButton"] button,
    div.stButton > button {
        background: linear-gradient(90deg, #00A6A6, #168AAD);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 650;
    }

    div[data-testid="stDownloadButton"] button:hover,
    div.stButton > button:hover {
        background: #087F8C;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    .info-card {
        background: #E6F7F7;
        border: 1px solid #B8E7E5;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. SESSION STATE
# ============================================================

defaults = {
    "original_data": None,
    "cleaned_data": None,
    "cleaning_report": None,
    "date_column": None,
    "sales_column": None,
    "upload_signature": None,
    "forecast_df": None,
    "forecast_model_name": None,
    "validated_columns": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def page_heading(title, subtitle):
    st.markdown(
        f'<div class="main-title">{title}</div>'
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


def section_heading(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, note="", color=TEAL):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{color}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def money(value):
    return "N/A" if pd.isna(value) else f"{value:,.2f}"


def number(value):
    return "N/A" if pd.isna(value) else f"{value:,.2f}"


def detect_date_column(df):
    preferred = [
        "Order Date", "Order_Date", "Date",
        "Sales Date", "Transaction Date", "Invoice Date"
    ]

    for wanted in preferred:
        for column in df.columns:
            if str(column).strip().lower() == wanted.lower():
                return column

    for column in df.columns:
        if "date" in str(column).lower():
            return column

    return None


def detect_sales_column(df):
    preferred = [
        "Sales", "Revenue", "Total Sales",
        "Amount", "Total Amount"
    ]

    for wanted in preferred:
        for column in df.columns:
            if str(column).strip().lower() == wanted.lower():
                return column

    for column in df.columns:
        name = str(column).lower()
        if "sales" in name or "revenue" in name:
            return column

    return None


def detect_optional_column(df, names):
    for wanted in names:
        for column in df.columns:
            if str(column).strip().lower() == wanted.lower():
                return column
    return None


def read_uploaded_csv(file):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None

    for encoding in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding=encoding,
                sep=None,
                engine="python"
            )

            if df.shape[1] >= 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df

        except Exception as error:
            last_error = error

    raise ValueError(f"Unable to read CSV: {last_error}")


def missing_values_report(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [
            int(df[c].isna().sum()) for c in df.columns
        ],
        "Missing Percentage (%)": [
            round(df[c].isna().mean() * 100, 2)
            for c in df.columns
        ]
    }).sort_values(
        "Missing Values",
        ascending=False
    ).reset_index(drop=True)


def clean_dataset(raw, date_col, sales_col):
    df = raw.copy()
    original_rows = len(df)
    missing_before = missing_values_report(df)

    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    df["_Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    sales_values = (
        df[sales_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    df["_Sales"] = pd.to_numeric(
        sales_values,
        errors="coerce"
    )

    invalid_dates = int(df["_Date"].isna().sum())
    invalid_sales = int(df["_Sales"].isna().sum())

    df = df.dropna(subset=["_Date", "_Sales"]).copy()
    df = df[np.isfinite(df["_Sales"])].copy()

    # Keep the original columns plus internal analysis columns.
    df = df.sort_values("_Date").reset_index(drop=True)

    report = {
        "Original Rows": original_rows,
        "Rows After Cleaning": len(df),
        "Duplicate Rows Removed": duplicates_removed,
        "Invalid Dates Removed": invalid_dates,
        "Invalid Sales Removed": invalid_sales,
        "Missing Before Table": missing_before,
        "Missing After Table": missing_values_report(df)
    }

    return df, report


def get_monthly_sales(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    monthly = df.set_index("_Date")["_Sales"].resample("MS").sum()
    monthly = monthly.asfreq("MS", fill_value=0)
    monthly.index.name = "Month"
    monthly.name = "Sales"

    return monthly.astype(float)


def observed_monthly_sales(df):
    data = df.copy()
    data["_Month"] = data["_Date"].dt.to_period("M")

    result = data.groupby("_Month")["_Sales"].sum().reset_index()
    result.columns = ["Month", "Sales"]
    result["Month"] = result["Month"].astype(str)

    return result.sort_values("Month")


def make_features(series):
    features = pd.DataFrame(index=series.index)
    features["Sales"] = series
    features["Month Number"] = series.index.month
    features["Year"] = series.index.year
    features["Lag 1"] = series.shift(1)
    features["Lag 2"] = series.shift(2)
    features["Lag 3"] = series.shift(3)
    features["Lag 6"] = series.shift(6)
    features["Lag 12"] = series.shift(12)
    features["Rolling Mean 3"] = series.shift(1).rolling(3).mean()
    features["Rolling Mean 6"] = series.shift(1).rolling(6).mean()

    return features.dropna()


def next_feature(date, history):
    def lag(n):
        return float(history[-n]) if len(history) >= n else 0.0

    def rolling(n):
        return float(np.mean(history[-n:])) if history else 0.0

    return pd.DataFrame([{
        "Month Number": date.month,
        "Year": date.year,
        "Lag 1": lag(1),
        "Lag 2": lag(2),
        "Lag 3": lag(3),
        "Lag 6": lag(6),
        "Lag 12": lag(12),
        "Rolling Mean 3": rolling(3),
        "Rolling Mean 6": rolling(6)
    }])


def fit_forecast_model(monthly):
    features = make_features(monthly)

    if len(monthly) >= 8 and len(features) >= 5:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )
        model.fit(
            features.drop(columns=["Sales"]),
            features["Sales"]
        )
        name = "Random Forest Regressor"
    else:
        model = LinearRegression()
        X = np.arange(len(monthly)).reshape(-1, 1)
        model.fit(X, monthly.values)
        name = "Linear Regression"

    return model, name


def forecast_future(monthly, horizon):
    if len(monthly) < 2:
        raise ValueError("At least two months of data are required.")

    model, model_name = fit_forecast_model(monthly)

    future_dates = pd.date_range(
        start=monthly.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    predictions = []
    history = list(monthly.values)

    if model_name == "Random Forest Regressor":
        for date in future_dates:
            prediction = float(
                model.predict(next_feature(date, history))[0]
            )
            prediction = max(0.0, prediction)
            predictions.append(prediction)
            history.append(prediction)
    else:
        for position in range(len(monthly), len(monthly) + horizon):
            prediction = float(
                model.predict(np.array([[position]]))[0]
            )
            predictions.append(max(0.0, prediction))

    return pd.DataFrame({
        "Month": future_dates,
        "Forecast Sales": predictions
    }), model_name


def evaluate_model(monthly):
    if len(monthly) < 4:
        raise ValueError(
            "At least four months of sales history are required."
        )

    test_size = min(3, max(1, len(monthly) // 5))
    train = monthly.iloc[:-test_size]
    test = monthly.iloc[-test_size:]

    predictions = []

    if len(train) >= 8 and len(make_features(train)) >= 5:
        features = make_features(train)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(
            features.drop(columns=["Sales"]),
            features["Sales"]
        )

        history = list(train.values)

        for date in test.index:
            prediction = float(
                model.predict(next_feature(date, history))[0]
            )
            predictions.append(max(0.0, prediction))
            history.append(float(monthly.loc[date]))

        model_name = "Random Forest Regressor"

    else:
        model = LinearRegression()
        model.fit(
            np.arange(len(train)).reshape(-1, 1),
            train.values
        )

        for position in range(len(train), len(monthly)):
            prediction = float(
                model.predict(np.array([[position]]))[0]
            )
            predictions.append(max(0.0, prediction))

        model_name = "Linear Regression"

    actual = test.values
    predicted = np.array(predictions)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    r2 = (
        r2_score(actual, predicted)
        if len(actual) > 1 and np.std(actual) > 0
        else np.nan
    )

    nonzero = np.abs(actual) > 1e-9
    mape = (
        float(np.mean(np.abs(
            (actual[nonzero] - predicted[nonzero]) /
            actual[nonzero]
        )) * 100)
        if np.any(nonzero)
        else np.nan
    )

    results = pd.DataFrame({
        "Month": test.index,
        "Actual Sales": actual,
        "Predicted Sales": predicted
    })

    results["Error"] = results["Actual Sales"] - results["Predicted Sales"]
    results["Absolute Error"] = results["Error"].abs()

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Model": model_name,
        "Test Months": len(test)
    }

    return metrics, results


# ============================================================
# 5. COLORFUL CHART FUNCTIONS
# ============================================================

def line_chart(data, title, ylabel="Sales", color=TEAL):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        data.index, data.values,
        color=color, marker="o", linewidth=2.5
    )
    ax.fill_between(
        np.arange(len(data)),
        data.values,
        color=color,
        alpha=0.12
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def comparison_chart(results):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = results["Month"].dt.strftime("%Y-%m")

    ax.plot(
        labels, results["Actual Sales"],
        marker="o", color=NAVY, linewidth=2.5,
        label="Actual Sales"
    )
    ax.plot(
        labels, results["Predicted Sales"],
        marker="o", color=ORANGE, linewidth=2.5,
        linestyle="--", label="Predicted Sales"
    )

    ax.set_title("Actual vs Predicted Sales", fontweight="bold")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def category_column(df):
    return detect_optional_column(df, ["Category"])


def subcategory_column(df):
    return detect_optional_column(df, ["Sub-Category", "Sub Category"])


def region_column(df):
    return detect_optional_column(df, ["Region"])


def profit_column(df):
    return detect_optional_column(df, ["Profit"])


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 24px 0;">
            <div style="font-size:30px;font-weight:900;">📊 SalesIQ</div>
            <div style="font-size:13px;opacity:0.85;">
                Smart Sales Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Sales Analysis",
            "Forecast",
            "Actual vs Predicted",
            "Model Evaluation"
        ]
    )

    st.markdown("---")
    st.caption("Upload your CSV on the Dashboard to begin.")


# ============================================================
# 7. DASHBOARD PAGE
# ============================================================

if page == "Dashboard":
    page_heading(
        "Sales Dashboard",
        "A colorful overview of sales, data quality and business performance."
    )

    uploaded_file = st.file_uploader(
        "Upload your Superstore CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        signature = (uploaded_file.name, uploaded_file.size)

        try:
            if (
                st.session_state.upload_signature != signature
                or st.session_state.original_data is None
            ):
                raw = read_uploaded_csv(uploaded_file)

                st.session_state.original_data = raw.copy()
                st.session_state.cleaned_data = None
                st.session_state.cleaning_report = None
                st.session_state.upload_signature = signature
                st.session_state.validated_columns = None

            raw = st.session_state.original_data

            if raw.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            section_heading("Select Dataset Columns")

            columns = list(raw.columns)
            detected_date = detect_date_column(raw)
            detected_sales = detect_sales_column(raw)

            date_index = columns.index(detected_date) if detected_date in columns else 0
            sales_index = columns.index(detected_sales) if detected_sales in columns else min(1, len(columns) - 1)

            col1, col2 = st.columns(2)

            with col1:
                date_col = st.selectbox(
                    "Date column",
                    columns,
                    index=date_index
                )

            with col2:
                sales_col = st.selectbox(
                    "Sales column",
                    columns,
                    index=sales_index
                )

            if date_col == sales_col:
                st.error("Choose different columns for Date and Sales.")
                st.stop()

            if st.button(
                "Validate and Clean Dataset",
                use_container_width=True
            ):
                cleaned, report = clean_dataset(raw, date_col, sales_col)

                if cleaned.empty:
                    st.error(
                        "No valid rows remain. Check the selected date and sales columns."
                    )
                    st.stop()

                st.session_state.cleaned_data = cleaned
                st.session_state.cleaning_report = report
                st.session_state.date_column = date_col
                st.session_state.sales_column = sales_col
                st.session_state.validated_columns = (date_col, sales_col)
                st.session_state.forecast_df = None

                st.success("Dataset cleaned successfully!")

            cleaned = st.session_state.cleaned_data
            report = st.session_state.cleaning_report

            if cleaned is not None and report is not None:
                if st.session_state.validated_columns != (date_col, sales_col):
                    st.warning(
                        "You changed the selected columns. Validate the dataset again."
                    )
                    st.stop()

                monthly = get_monthly_sales(cleaned)
                observed = observed_monthly_sales(cleaned)

                st.markdown("---")
                section_heading("Key Performance Indicators")

                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    metric_card(
                        "Total Sales",
                        money(cleaned["_Sales"].sum()),
                        "Sales across valid rows",
                        TEAL
                    )

                with c2:
                    metric_card(
                        "Total Orders / Rows",
                        f"{len(cleaned):,}",
                        "Valid data rows",
                        BLUE
                    )

                with c3:
                    metric_card(
                        "Monthly Average",
                        money(monthly.mean()),
                        "Includes zero-filled missing months",
                        PURPLE
                    )

                with c4:
                    metric_card(
                        "Months Covered",
                        f"{len(observed):,}",
                        "Months with transactions",
                        ORANGE
                    )

                st.markdown("---")
                section_heading("Monthly Sales Trend")

                st.pyplot(
                    line_chart(monthly, "Monthly Sales Trend", color=TEAL),
                    use_container_width=True
                )

                # Extra dashboard charts
                c1, c2 = st.columns(2)

                cat_col = category_column(cleaned)
                reg_col = region_column(cleaned)

                with c1:
                    if cat_col:
                        cat_sales = cleaned.groupby(cat_col)["_Sales"].sum().sort_values()

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.barh(
                            cat_sales.index.astype(str),
                            cat_sales.values,
                            color=CHART_COLORS[:len(cat_sales)]
                        )
                        ax.set_title("Sales by Category", fontweight="bold")
                        ax.set_xlabel("Sales")
                        ax.grid(axis="x", alpha=0.2)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Category chart unavailable: no Category column.")

                with c2:
                    if reg_col:
                        region_sales = cleaned.groupby(reg_col)["_Sales"].sum().sort_values(ascending=False)

                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.pie(
                            region_sales.values,
                            labels=region_sales.index.astype(str),
                            autopct="%1.1f%%",
                            startangle=90,
                            colors=CHART_COLORS[:len(region_sales)]
                        )
                        ax.set_title("Sales Share by Region", fontweight="bold")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Region chart unavailable: no Region column.")

                st.markdown("---")
                section_heading("Data Cleaning Report")

                a, b, c = st.columns(3)

                with a:
                    metric_card(
                        "Duplicate Rows Removed",
                        f"{report['Duplicate Rows Removed']:,}",
                        color=PURPLE
                    )

                with b:
                    metric_card(
                        "Invalid Dates Removed",
                        f"{report['Invalid Dates Removed']:,}",
                        color=ORANGE
                    )

                with c:
                    metric_card(
                        "Invalid Sales Removed",
                        f"{report['Invalid Sales Removed']:,}",
                        color=RED
                    )

                section_heading("Missing Values Before Cleaning")
                st.dataframe(
                    report["Missing Before Table"],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Missing Values Before Cleaning",
                    report["Missing Before Table"].to_csv(index=False).encode("utf-8"),
                    "missing_values_before.csv",
                    "text/csv"
                )

                section_heading("Missing Values After Cleaning")
                st.dataframe(
                    report["Missing After Table"],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Missing Values After Cleaning",
                    report["Missing After Table"].to_csv(index=False).encode("utf-8"),
                    "missing_values_after.csv",
                    "text/csv"
                )

                section_heading("Cleaned Dataset Preview")
                st.dataframe(cleaned.head(20), use_container_width=True)

                st.download_button(
                    "Download Cleaned Dataset",
                    cleaned.to_csv(index=False).encode("utf-8"),
                    "cleaned_superstore.csv",
                    "text/csv"
                )

        except Exception as error:
            st.error(f"Could not process the CSV: {error}")

    else:
        st.markdown(
            """
            <div class="info-card">
                <b>Welcome to SalesIQ!</b><br>
                Upload your Superstore CSV, select the date and sales columns,
                and click Validate and Clean Dataset.
                Standard Superstore files commonly use
                <b>Order Date</b> and <b>Sales</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Dashboard capabilities")
        st.markdown(
            """
            - Colorful sales charts and category comparisons
            - Missing-values and data-cleaning reports
            - Monthly sales forecasting
            - Actual-versus-predicted comparison
            - Forecast accuracy evaluation
            """
        )


# ============================================================
# 8. REQUIRE A VALIDATED DATASET FOR OTHER PAGES
# ============================================================

if page != "Dashboard":
    df = st.session_state.cleaned_data

    if df is None or df.empty:
        page_heading(
            "SalesIQ",
            "Upload and validate your Superstore CSV on the Dashboard first."
        )
        st.warning("No validated dataset is available.")
        st.stop()

    monthly = get_monthly_sales(df)


# ============================================================
# 9. SALES ANALYSIS PAGE - MORE CHARTS
# ============================================================

if page == "Sales Analysis":
    page_heading(
        "Sales Analysis",
        "Explore sales trends, seasonality, categories, regions and distributions."
    )

    observed = observed_monthly_sales(df)

    if observed.empty:
        st.warning("No observed monthly sales data is available.")
        st.stop()

    highest = observed.loc[observed["Sales"].idxmax()]
    lowest = observed.loc[observed["Sales"].idxmin()]

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Highest Sales Month",
            highest["Month"],
            money(highest["Sales"]),
            GREEN
        )

    with c2:
        metric_card(
            "Lowest Sales Month",
            lowest["Month"],
            money(lowest["Sales"]),
            RED
        )

    with c3:
        metric_card(
            "Observed Months",
            f"{len(observed)}",
            "Months with valid transactions",
            BLUE
        )

    st.markdown("---")
    section_heading("1. Monthly Sales and Moving Average")

    moving_avg = monthly.rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        monthly.index, monthly.values,
        label="Monthly Sales", color=BLUE, marker="o"
    )
    ax.plot(
        moving_avg.index, moving_avg.values,
        label="3-Month Moving Average", color=ORANGE, linewidth=3
    )
    ax.set_title("Monthly Sales Trend", fontweight="bold")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    section_heading("2. Highest and Lowest Sales Months")

    st.caption(
        "This comparison uses months containing valid sales transactions. "
        "Missing calendar months are not treated as observed zero-sales months."
    )

    st.dataframe(
        observed.sort_values("Sales", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Monthly Sales Summary",
        observed.to_csv(index=False).encode("utf-8"),
        "monthly_sales_summary.csv",
        "text/csv"
    )

    st.markdown("---")
    section_heading("3. Seasonality Analysis")

    seasonal = df.copy()
    seasonal["Year"] = seasonal["_Date"].dt.year
    seasonal["Month Number"] = seasonal["_Date"].dt.month

    year_count = seasonal["Year"].nunique()
    years = sorted(seasonal["Year"].unique())

    st.write(f"**Years covered:** {', '.join(map(str, years))}")

    if year_count < 2:
        st.warning(
            "Year-coverage warning: only one year of data is available. "
            "The chart cannot reliably establish a recurring seasonal pattern."
        )
    else:
        st.success(
            f"The dataset contains {year_count} years. "
            "Monthly averages can help explore possible seasonal patterns."
        )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    by_year_month = (
        seasonal.groupby(["Year", "Month Number"])["_Sales"]
        .sum()
        .reset_index()
    )

    monthly_average = (
        by_year_month.groupby("Month Number")["_Sales"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(
        month_labels,
        monthly_average.values,
        color=CHART_COLORS[:12],
        edgecolor="white"
    )
    ax.set_title("Average Sales by Calendar Month", fontweight="bold")
    ax.set_ylabel("Average Sales")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    seasonality_table = pd.DataFrame({
        "Month": month_labels,
        "Average Sales": monthly_average.values
    })

    st.dataframe(seasonality_table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Seasonality Report",
        seasonality_table.to_csv(index=False).encode("utf-8"),
        "seasonality_report.csv",
        "text/csv"
    )

    st.markdown("---")
    section_heading("4. Sales Distribution: Histogram and Boxplot")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            df["_Sales"],
            bins=30,
            color=PURPLE,
            edgecolor="white",
            alpha=0.9
        )
        ax.set_title("Sales Histogram", fontweight="bold")
        ax.set_xlabel("Sales")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        box = ax.boxplot(
            df["_Sales"].dropna(),
            patch_artist=True
        )
        box["boxes"][0].set_facecolor(ORANGE)
        box["medians"][0].set_color(NAVY)
        ax.set_title("Sales Boxplot", fontweight="bold")
        ax.set_ylabel("Sales")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    section_heading("5. Sales by Category and Sub-Category")

    cat_col = category_column(df)
    subcat_col = subcategory_column(df)

    col1, col2 = st.columns(2)

    with col1:
        if cat_col:
            cat_sales = (
                df.groupby(cat_col)["_Sales"]
                .sum()
                .sort_values()
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(
                cat_sales.index.astype(str),
                cat_sales.values,
                color=CHART_COLORS[:len(cat_sales)]
            )
            ax.set_title("Category Sales Ranking", fontweight="bold")
            ax.set_xlabel("Sales")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Category chart requires a Category column.")

    with col2:
        if subcat_col:
            subcat_sales = (
                df.groupby(subcat_col)["_Sales"]
                .sum()
                .sort_values(ascending=False)
                .head(12)
                .sort_values()
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(
                subcat_sales.index.astype(str),
                subcat_sales.values,
                color=CHART_COLORS[:len(subcat_sales)]
            )
            ax.set_title("Top Sub-Categories", fontweight="bold")
            ax.set_xlabel("Sales")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Sub-category chart requires a Sub-Category column.")

    st.markdown("---")
    section_heading("6. Regional Sales Comparison")

    reg_col = region_column(df)

    if reg_col:
        region_sales = (
            df.groupby(reg_col)["_Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(
                region_sales.values,
                labels=region_sales.index.astype(str),
                colors=CHART_COLORS[:len(region_sales)],
                autopct="%1.1f%%",
                startangle=90
            )
            ax.set_title("Regional Sales Share", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                region_sales.index.astype(str),
                region_sales.values,
                color=CHART_COLORS[:len(region_sales)]
            )
            ax.set_title("Regional Sales Comparison", fontweight="bold")
            ax.set_ylabel("Sales")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("Regional charts require a Region column.")

    st.markdown("---")
    section_heading("7. Sales by Day of Week")

    weekday_data = df.copy()
    weekday_data["Weekday"] = weekday_data["_Date"].dt.day_name()

    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    weekday_sales = (
        weekday_data.groupby("Weekday")["_Sales"]
        .sum()
        .reindex(weekday_order)
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(
        weekday_sales.index,
        weekday_sales.values,
        color=CHART_COLORS[:7]
    )
    ax.set_title("Total Sales by Day of Week", fontweight="bold")
    ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    section_heading("8. Profit vs Sales")

    prof_col = profit_column(df)

    if prof_col:
        profit_df = df.copy()
        profit_df["_ProfitNumeric"] = pd.to_numeric(
            profit_df[prof_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

        profit_df = profit_df.dropna(
            subset=["_ProfitNumeric", "_Sales"]
        )

        if not profit_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            points = ax.scatter(
                profit_df["_Sales"],
                profit_df["_ProfitNumeric"],
                c=profit_df["_ProfitNumeric"],
                cmap="coolwarm",
                alpha=0.65,
                s=35
            )
            ax.axhline(0, color=NAVY, linestyle="--", linewidth=1)
            ax.set_title("Profit vs Sales Relationship", fontweight="bold")
            ax.set_xlabel("Sales")
            ax.set_ylabel("Profit")
            fig.colorbar(points, ax=ax, label="Profit")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.caption(
                "Each point represents a transaction. Color indicates profit; "
                "negative profit values represent losses."
            )
    else:
        st.info("Profit chart requires a Profit column.")

    st.markdown("---")
    section_heading("9. Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number).copy()

    # Internal analysis columns are included only if useful.
    if len(numeric_df.columns) >= 2:
        correlation = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="Spectral",
            center=0,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Numeric Feature Correlation", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("The heatmap requires at least two numeric columns.")


# ============================================================
# 10. FORECAST PAGE
# ============================================================

elif page == "Forecast":
    page_heading(
        "Sales Forecast",
        "Forecast future monthly sales using historical trends."
    )

    if len(monthly) < 2:
        st.error("At least two months of data are required.")
        st.stop()

    section_heading("Forecast Settings")

    horizon = st.slider(
        "Number of months to forecast",
        min_value=1,
        max_value=12,
        value=6
    )

    st.write(f"Historical monthly observations: **{len(monthly)}**")

    if st.button("Generate Sales Forecast", use_container_width=True):
        try:
            forecast_df, model_name = forecast_future(monthly, horizon)
            st.session_state.forecast_df = forecast_df
            st.session_state.forecast_model_name = model_name
            st.session_state.forecast_horizon = horizon
        except Exception as error:
            st.error(f"Forecast failed: {error}")

    if st.session_state.forecast_df is not None:
        forecast_df = st.session_state.forecast_df

        if st.session_state.get("forecast_horizon") != horizon:
            st.info("Change the horizon and click Generate Sales Forecast again.")
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card(
                    "Forecast Months",
                    str(len(forecast_df)),
                    color=BLUE
                )

            with col2:
                metric_card(
                    "Total Forecast",
                    money(forecast_df["Forecast Sales"].sum()),
                    color=PURPLE
                )

            with col3:
                metric_card(
                    "Average Forecast",
                    money(forecast_df["Forecast Sales"].mean()),
                    color=ORANGE
                )

            st.markdown("---")
            section_heading("Forecast Values")

            display_forecast = forecast_df.copy()
            display_forecast["Month"] = display_forecast["Month"].dt.strftime("%Y-%m")

            st.dataframe(
                display_forecast.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Forecast CSV",
                display_forecast.to_csv(index=False).encode("utf-8"),
                "sales_forecast.csv",
                "text/csv"
            )

            st.markdown("---")
            section_heading("Historical vs Forecast Sales")

            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(
                monthly.index, monthly.values,
                label="Historical Sales",
                color=BLUE, marker="o", linewidth=2.5
            )
            ax.plot(
                forecast_df["Month"],
                forecast_df["Forecast Sales"],
                label="Forecast Sales",
                color=ORANGE, marker="o",
                linestyle="--", linewidth=2.5
            )
            ax.axvline(
                monthly.index[-1],
                color=PURPLE,
                linestyle=":",
                label="Forecast Starts"
            )
            ax.set_title("Historical Sales vs Future Forecast", fontweight="bold")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.caption(
                f"Model used: {st.session_state.forecast_model_name}. "
                "Forecasts are estimates, not guaranteed future sales."
            )


# ============================================================
# 11. ACTUAL VS PREDICTED PAGE
# ============================================================

elif page == "Actual vs Predicted":
    page_heading(
        "Actual vs Predicted",
        "Compare historical sales with predictions on a time-based holdout."
    )

    try:
        metrics, results = evaluate_model(monthly)

        section_heading("Actual vs Predicted Chart")
        st.pyplot(comparison_chart(results), use_container_width=True)

        st.markdown("---")
        section_heading("Prediction Results Table")

        display = results.copy()
        display["Month"] = display["Month"].dt.strftime("%Y-%m")

        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Actual vs Predicted Results",
            display.to_csv(index=False).encode("utf-8"),
            "actual_vs_predicted.csv",
            "text/csv"
        )

        st.markdown("---")
        section_heading("Prediction Error Chart")

        labels = results["Month"].dt.strftime("%Y-%m")
        colors = [
            GREEN if error >= 0 else RED
            for error in results["Error"]
        ]

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(labels, results["Error"], color=colors)
        ax.axhline(0, color=NAVY, linewidth=1)
        ax.set_title("Prediction Error: Actual minus Predicted", fontweight="bold")
        ax.set_ylabel("Error")
        ax.grid(axis="y", alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    except Exception as error:
        st.error(f"Comparison could not be generated: {error}")


# ============================================================
# 12. MODEL EVALUATION PAGE
# ============================================================

elif page == "Model Evaluation":
    page_heading(
        "Model Evaluation",
        "Evaluate forecasting accuracy using historical holdout data."
    )

    try:
        metrics, results = evaluate_model(monthly)

        section_heading("Accuracy Metrics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "MAE",
                number(metrics["MAE"]),
                "Mean Absolute Error",
                TEAL
            )

        with c2:
            metric_card(
                "RMSE",
                number(metrics["RMSE"]),
                "Root Mean Squared Error",
                BLUE
            )

        with c3:
            metric_card(
                "R² Score",
                number(metrics["R2"]),
                "Can be negative on test data",
                PURPLE
            )

        with c4:
            metric_card(
                "MAPE",
                f"{metrics['MAPE']:.2f}%" if np.isfinite(metrics["MAPE"]) else "N/A",
                "Percentage error for non-zero actuals",
                ORANGE
            )

        st.markdown("---")
        section_heading("Model Details")

        st.write(f"**Model:** {metrics['Model']}")
        st.write(f"**Holdout months:** {metrics['Test Months']}")

        st.markdown(
            """
            - **MAE:** Average absolute prediction error.
            - **RMSE:** Penalizes larger prediction errors more strongly.
            - **R²:** Measures how well predictions explain variation in test data.
            - **MAPE:** Average percentage error for months with non-zero actual sales.

            Lower MAE and RMSE usually mean smaller errors. R² can be negative
            when predictions perform poorly on the test period.
            """
        )

        st.markdown("---")
        section_heading("Actual vs Predicted")

        st.pyplot(comparison_chart(results), use_container_width=True)

        display = results.copy()
        display["Month"] = display["Month"].dt.strftime("%Y-%m")

        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Evaluation Results",
            display.to_csv(index=False).encode("utf-8"),
            "model_evaluation_results.csv",
            "text/csv"
        )

        st.caption(
            "The evaluation holds out the last available months. The Random Forest "
            "evaluation uses rolling one-step predictions, so actual values from "
            "earlier test months are used when predicting later test months."
        )

    except Exception as error:
        st.error(f"Model evaluation failed: {error}")
        st.info(
            "Upload valid data with at least four months of sales history."
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#8290A5;font-size:12px;padding:10px;">
        SalesIQ | Sales Forecasting System<br>
        Sales Insights • Data Analytics • Forecasting
    </div>
    """,
    unsafe_allow_html=True
)
