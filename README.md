# Book Sales Forecasting - Multi-Model Time-Series Analysis

Comprehensive time-series forecasting project comparing classical (SARIMA), machine learning (XGBoost), deep learning (LSTM), and hybrid approaches for 32-week sales prediction of two books, demonstrating that hybrid SARIMA-LSTM models can outperform individual methods.

## Project Overview

Accurate demand forecasting is essential for inventory management, supply chain planning, and revenue optimisation in publishing and retail. This project applies and systematically compares multiple forecasting methodologies to weekly sales data for "The Alchemist" and "The Very Hungry Caterpillar," evaluating their effectiveness across different data patterns and volatility levels.

## Approach

**Section 1-2: Data Exploration and Preparation**
- Data cleaning and temporal alignment from 2012 onwards
- Sales pattern investigation and trend analysis for both books
- Feature engineering: lag features, rolling means, week/month indicators
- Train/test split: all data up to cutoff for training, final 32 weeks for evaluation

**Section 3: Classical Time-Series Techniques**
- Seasonal decomposition to isolate trend, seasonality, and residuals
- Stationarity testing (ADF, KPSS) and differencing
- ACF/PACF analysis for ARIMA order selection
- SARIMA modelling with auto_arima parameter optimisation
- 32-week forecast with confidence intervals

**Section 4: Machine Learning and Deep Learning**
- XGBoost with lag features, rolling means, and calendar features
- Grid search hyperparameter tuning with time-series cross-validation
- LSTM neural network with sequence-based lookback windows
- Keras Tuner for LSTM architecture optimisation
- Comparative evaluation of ML versus classical approaches

**Section 5: Hybrid Models**
- Sequential Hybrid: SARIMA captures linear patterns, LSTM models residuals
- Parallel Hybrid: independent SARIMA and LSTM forecasts with weighted combination
- Hybrid versus individual model performance comparison

**Section 6: Monthly Aggregation**
- Weekly to monthly sales aggregation
- Monthly SARIMA and XGBoost forecasting
- Weekly versus monthly granularity comparison

## Key Results

| Model | The Alchemist MAE | The Very Hungry Caterpillar MAE |
|-------|-------------------|-------------------------------|
| SARIMA | 169.33 | 415.45 |
| XGBoost | 607.91 | 2,167.09 |
| LSTM | 166.96 | 706.73 |
| Sequential Hybrid | **144.99** | 495.62 |

- **Sequential Hybrid** achieved the best performance for The Alchemist (MAE: 144.99), approximately 15% improvement over standalone SARIMA
- **SARIMA** performed best for The Very Hungry Caterpillar (MAE: 415.45), where higher sales volatility made ML/DL approaches less stable
- **XGBoost** consistently underperformed across both books, suggesting tree-based methods are less suited to this type of seasonal time-series data
- **LSTM** performed competitively for The Alchemist but struggled with The Very Hungry Caterpillar's volatile patterns
- Classical techniques achieved the lowest average MAPE (25.74%) across both books

## Tech Stack

- **Classical Forecasting** - Statsmodels (SARIMA, STL decomposition, ADF/KPSS tests), pmdarima (auto_arima)
- **Machine Learning** - XGBoost, LightGBM, Scikit-learn (GridSearchCV, TimeSeriesSplit)
- **Deep Learning** - TensorFlow/Keras (LSTM), Keras Tuner
- **Time-Series Tools** - sktime (PolynomialTrendForecaster, Deseasonalizer)
- **Visualisation** - Matplotlib, Seaborn
- **Data Processing** - Pandas, NumPy, SciPy

## Repository Structure

```
book-sales-forecasting/
|-- book_sales_forecasting.ipynb   # Main analysis notebook
|-- requirements.txt               # Python dependencies
|-- README.md                      # This file
```

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook book_sales_forecasting.ipynb
```

**Note:** LSTM training benefits from GPU acceleration. The notebook was developed on Google Colab with GPU runtime.

## Dataset

Weekly book sales data for "The Alchemist" and "The Very Hungry Caterpillar," spanning multiple years with seasonal patterns, trends, and varying levels of volatility. Data was sourced from the University of Cambridge Data Science programme.

## Author

**Raquel J.** - Data Scientist and Analytics Engineer

- Portfolio: [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
- LinkedIn: [Raquel J.](https://linkedin.com/in/664113153)
