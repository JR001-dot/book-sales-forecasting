# Book Sales Demand Forecasting

Weekly demand forecasting for two long-selling book titles, comparing statistical, machine-learning, deep-learning and hybrid models.

**Origin:** University of Cambridge Data Science Career Accelerator coursework (2024-25). **Data:** source anonymised and not included in this repository. Notebook outputs are cleared to protect the source data; the figures below come from the original run.

## What was done
- Weekly sales records (227,224 rows across eight sheets) resampled so that weeks with no sales count as zero.
- Two titles (Title A, Title B) from 2012 onwards; the final 32 weeks held out as the test set.
- Models: SARIMA via `auto_arima` (yearly seasonality, m = 52), XGBoost with lag and rolling-mean features, an LSTM tuned with KerasTuner, sequential and parallel SARIMA-LSTM hybrids, and monthly XGBoost and SARIMA (8-month horizon).

## Results (32-week test period)
| Model | Title A MAE | Title A error | Title B MAE | Title B error |
|---|---|---|---|---|
| SARIMA | 169.33 | MAPE 31.9% | 382.28 | MAPE 19.6% |
| LSTM | 170.63 | MAPE 22.2% | 592.25 | MAPE 28.1% |
| Parallel hybrid | 181.63 | sMAPE 34.8% | 1,174.39 | sMAPE 61.8% |
| Sequential hybrid | 214.33 | sMAPE 39.8% | 1,096.52 | sMAPE 56.6% |

- SARIMA was best for Title B; for Title A, SARIMA and LSTM were tied on MAE and LSTM had the lower MAPE.
- The hybrids did not improve on the single models.

## Known issues
- **Weekly XGBoost is broken:** its recursive forecasts stay flat at about 50 units against actual sales of 1,000-3,500, so it is excluded from the results.
- Title B's series was interpolated and smoothed **before** the train/test split, which flatters its SARIMA error; the hybrid pipeline uses a centred rolling mean (looks ahead).
- The optimised hybrid weight came out at 0.0 (one model only); hybrid errors are sMAPE, not MAPE, so they are not directly comparable with the first two rows.
- Monthly SARIMA failed to produce forecasts; monthly XGBoost MAPE was 48.3% (Title A) and 27.4% (Title B).

## Tech stack
Python, pandas, NumPy, statsmodels, pmdarima, XGBoost, TensorFlow/Keras, KerasTuner, sktime, Matplotlib

## Repository
- `book_sales_forecasting.ipynb`
- `requirements.txt`

---

**Author:** Raquel J. | [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
