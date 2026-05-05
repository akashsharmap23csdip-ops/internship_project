# 📊 Sales Forecasting using Machine Learning

## A Comparative Study of Linear Regression & Random Forest Regressor

---

### 📌 Overview

End-to-end ML pipeline for sales forecasting using the **Superstore Sales Dataset** (~10,000 records, 2014–2017). Compares Linear Regression (baseline) vs Random Forest Regressor (advanced), evaluated using MAE, RMSE, and R² metrics.

---

### 📁 Project Structure

```
int project/
├── data/
│   └── superstore_sales.csv           # Raw dataset (9,994 × 21 columns)
├── outputs/
│   ├── eda_plots/                     # 9 EDA visualizations
│   │   ├── sales_distribution.png
│   │   ├── sales_by_category.png
│   │   ├── sales_by_region.png
│   │   ├── monthly_sales_trend.png
│   │   ├── correlation_heatmap.png
│   │   ├── scatter_plots.png
│   │   ├── sales_by_segment.png
│   │   ├── top_subcategories.png
│   │   └── quarterly_sales.png
│   └── model_results/                 # Model evaluation outputs
│       ├── model_comparison.csv
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── feature_importance.png
│       └── residual_analysis.png
├── app.py                             # Local web server launcher
├── frontend/                          # Professional web dashboard
│   ├── index.html
│   ├── style.css
│   └── script.js
├── sales_forecasting.ipynb            # Source notebook
├── sales_forecasting_executed.ipynb   # Executed notebook (with outputs)
├── requirements.txt                   # Python dependencies
├── PROJECT_REPORT.md                  # Detailed project report
└── README.md                          # This file
```

---

### ⚙️ Installation

```bash
pip install -r requirements.txt
```

### ▶️ How to Run

```bash
# 1. Launch the local dashboard
python app.py

# 2. Open the dashboard manually if needed
http://localhost:8000/frontend/index.html

# 3. Open the pre-executed notebook
jupyter notebook sales_forecasting_executed.ipynb

# 4. Re-execute from scratch
jupyter nbconvert --to notebook --execute --output sales_forecasting_executed.ipynb sales_forecasting.ipynb
```

The dashboard uses only the Python standard library for hosting, so no extra web framework is required.

---

### 🌐 Frontend Dashboard

The new local dashboard provides a browser-friendly overview of the project:

- executive summary cards for the dataset and best model
- the full model comparison table from `outputs/model_results/model_comparison.csv`
- a metric bar view for quick comparison
- a gallery of the major EDA and model diagnostic plots
- direct links back to the report and notebook files

Start it with `python app.py`, then open `http://localhost:8000/frontend/index.html`.

---

### 🔬 Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Data Loading | Load CSV (9,994 rows × 21 columns), inspect shape, types, nulls |
| 2. Preprocessing | Date conversion, drop 10 irrelevant identifier columns |
| 3. Feature Engineering | 5 temporal features + shipping duration + label encoding |
| 4. EDA | 9 rich visualizations (distributions, trends, correlations) |
| 5. Model Building | Linear Regression (baseline) + Random Forest Regressor |
| 6. Hyperparameter Tuning | GridSearchCV with 24 combinations × 5-fold CV |
| 7. Cross-Validation | 5-fold CV on all 3 model variants |
| 8. Model Comparison | Side-by-side MAE, RMSE, R² evaluation |
| 9. Feature Importance | Top feature ranking from tuned Random Forest |
| 10. Residual Analysis | Diagnostic residual scatter plots |

---

### 📈 Results

#### Model Comparison

| Model | MAE ($) | RMSE ($) | R² Score |
|-------|---------|----------|----------|
| Linear Regression | 243.05 | 822.00 | -0.1439 |
| Random Forest (Default) | 86.45 | 502.69 | 0.5722 |
| **Random Forest (Tuned)** | **85.62** | **496.07** | **0.5834** |

#### Cross-Validation (5-Fold R²)

| Model | Mean R² | Std Dev |
|-------|---------|---------|
| Linear Regression | 0.2537 | ±0.2128 |
| Random Forest (Default) | 0.7405 | ±0.1008 |
| Random Forest (Tuned) | 0.7322 | ±0.1054 |

#### Top 5 Feature Importances

| Feature | Importance |
|---------|------------|
| Profit | 84.35% |
| Discount | 3.55% |
| Sub-Category | 3.38% |
| Quantity | 2.02% |
| Category | 1.77% |

---

### 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical computation |
| Scikit-learn | ML models, GridSearchCV, cross-validation |
| Matplotlib | Data and result visualisation |
| Seaborn | Statistical charts and heatmaps |
| Jupyter Notebook | Interactive development environment |

---

### 📊 Dataset Source

[Kaggle — Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

---

### 📄 Full Report

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the comprehensive project report with detailed analysis, methodology, business insights, and recommendations.

---

### 🖼️ Key Project Artifacts

- [Sales distribution](outputs/eda_plots/sales_distribution.png)
- [Sales by category](outputs/eda_plots/sales_by_category.png)
- [Sales by region](outputs/eda_plots/sales_by_region.png)
- [Monthly sales trend](outputs/eda_plots/monthly_sales_trend.png)
- [Correlation heatmap](outputs/eda_plots/correlation_heatmap.png)
- [Model comparison](outputs/model_results/model_comparison.png)
- [Feature importance](outputs/model_results/feature_importance.png)
- [Actual vs predicted](outputs/model_results/actual_vs_predicted.png)
- [Residual analysis](outputs/model_results/residual_analysis.png)
