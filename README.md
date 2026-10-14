# 📊 Sales Forecasting System

A machine learning project that analyzes historical sales data, identifies sales trends, and forecasts future sales using Python and an interactive Streamlit dashboard.

## 📌 Project Overview

This project helps users understand sales performance through data cleaning, monthly sales analysis, seasonality visualization, and machine learning-based forecasting. It includes an interactive dashboard with multiple colorful charts for exploring sales data and comparing actual sales with predicted values.

## 🎯 Objectives

- Analyze historical sales data.
- Identify the highest- and lowest-sales months.
- Explore monthly sales trends and seasonality.
- Handle missing values and clean the dataset.
- Train a machine learning model for sales forecasting.
- Evaluate model performance using appropriate metrics.
- Build an interactive dashboard for sales analysis and prediction.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Machine Learning
- Data Visualization

## 📊 Dataset

The project uses a sales dataset uploaded in CSV format.

The dataset should contain relevant sales information, including:

- `Order Date` — date of the sales transaction
- `Sales` — sales amount

Additional columns, such as category, sub-category, region, and profit, can support further analysis when available in the uploaded dataset.

## 🤖 Machine Learning Model

The application uses machine learning to generate sales forecasts and compare predicted results with actual sales.

Model evaluation includes forecasting metrics supported by the application.

The forecasting results depend on the uploaded dataset, date coverage, selected features, and model settings.

## ✨ Features

- Sales dataset CSV upload
- Missing-value report before and after cleaning
- Monthly sales trend analysis
- Highest- and lowest-sales month identification
- Seasonality analysis with year coverage checks
- Multiple colorful charts and visualizations
- Sales forecasting
- Actual vs. predicted sales comparison
- Forecast values and model evaluation metrics
- Interactive Streamlit dashboard

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sahana63644/Sales-Forecasting-System.git
cd Sales-Forecasting-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## ⚠️ Disclaimer

Sales forecasts are estimates based on historical data and the selected machine learning model. They are not guaranteed future results and should be considered alongside other business information when making decisions.

## 👩‍💻 Author

**Sahana B**

GitHub: [@Sahana63644](https://github.com/Sahana63644)

Project: [Sales Forecasting System](https://github.com/Sahana63644/Sales-Forecasting-System)
