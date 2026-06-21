# 🚀 Stock Prediction App (ARIMA) 📈

A Streamlit web application that forecasts short-term stock prices for major NASDAQ and NYSE-listed companies using **ARIMA** time series modeling. The app automatically selects model parameters, validates predictions against historical data, and visualizes both forecasts and confidence intervals.

---

## Features

- **Two ticker input modes** — pick from ~95 popular pre-loaded stocks, or type any ticker manually (including non-US markets, e.g. `THYAO.IS`).
- **Automatic ARIMA order selection** — performs a grid search over AR and MA orders (`p`, `q` from 0–2) and chooses the best fit by **AIC**.
- **Configurable forecast horizon** — 1 day, 3 days, 1 week, 1 month, or 3 months ahead.
- **Backtesting on a hold-out set** — predicts the most recent 20% of history and compares against actuals.
- **Performance metrics** — reports MAE, RMSE, and MAPE so you can judge reliability.
- **Interactive charts** — forecast plot plus a prediction-vs-actual chart with shaded confidence intervals (Plotly).

---

## What is ARIMA?

**ARIMA** stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage. It is one of the most widely used classical models for forecasting time series data — values measured sequentially over time, such as daily closing stock prices. An ARIMA model is described by three parameters, written as **ARIMA(p, d, q)**:

### AR — AutoRegressive (`p`)
The model predicts the next value as a weighted combination of its own **previous values** (lags). The parameter `p` is how many past observations are used. The intuition: today's price carries information about tomorrow's.

### I — Integrated (`d`)
ARIMA requires the series to be **stationary** — its mean and variance should not drift over time. Raw stock prices usually trend, so they are made stationary by **differencing** (subtracting the previous value from the current one). The parameter `d` is the number of differencing steps applied.

### MA — Moving Average (`q`)
The model also uses past **forecast errors** (the difference between what it predicted and what actually happened) to correct future predictions. The parameter `q` is how many past error terms are included.

### How this app uses ARIMA

This app fixes `d = 0` and grid-searches `p` and `q` over the range 0–2. For each `(p, q)` combination it fits an ARIMA model and records the **AIC** (Akaike Information Criterion) and **BIC**, which measure goodness-of-fit while penalizing complexity. The combination with the **lowest AIC** is selected, refit on the full series, and then used to:

1. **Backtest** — forecast the most recent 20% of the data and compare it to the real prices.
2. **Forecast** — project prices forward over your chosen horizon.

> **A note on `d = 0`:** Because the app does not difference the data, it is effectively fitting an **ARMA** model to the raw price levels. This works for short horizons but can underperform on strongly trending series. Adding a differencing term (`d = 1`) or auto-selecting it (e.g. via `pmdarima.auto_arima`) is a natural improvement.

---

## Understanding the Metrics

| Metric | Meaning | Lower is better? |
|--------|---------|------------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual price, in dollars. | ✅ Yes |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalizes large errors more heavily. | ✅ Yes |
| **MAPE** (Mean Absolute Percentage Error) | Average error as a percentage of actual price — easy to compare across stocks. | ✅ Yes |

A MAPE of, say, 3% means the model's backtested predictions were off by about 3% on average.

---

## Project Structure

```
stock_prediction/
├── App.py                 # Main Streamlit application
├── popular_stocks.json    # Mapping of company names → ticker symbols
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## How to Run

### 1. Prerequisites

- **Python 3.9 or 3.10** is recommended. The pinned dependencies (`pandas==1.5.3`, `numpy==1.22.0`, `statsmodels==0.12.1`) are older and may not build on Python 3.12+.
- `git` (optional, if you clone the repository).

### 2. Get the project files

If the project is in a Git repository:

```bash
git clone <your-repository-url>
cd stock_prediction
```

Otherwise, just place `App.py`, `popular_stocks.json`, and `requirements.txt` together in one folder and open a terminal in that folder.

### 3. Create and activate a virtual environment

Keeping dependencies isolated avoids conflicts with other projects.

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the app

```bash
streamlit run App.py
```

Streamlit will start a local server and automatically open the app in your browser at **http://localhost:8501**. If it doesn't open automatically, paste that URL into your browser.

### 6. Use the app

1. Choose a ticker — select from the dropdown or enter one manually.
2. Pick a forecasting interval.
3. Click **Analyze** and wait for the model to fit (the spinner shows progress).
4. Review the forecast chart, the forecast table, the prediction-vs-actual chart, and the metrics table.

---

## Troubleshooting

- **`No data found` / empty results:** The ticker may be invalid or delisted. Double-check the symbol on Yahoo Finance. Non-US tickers need their exchange suffix (e.g. `.IS` for Borsa Istanbul, `.L` for London).
- **Dependency build errors on install:** You are likely on a newer Python version. Use Python 3.9 or 3.10, or relax the pinned versions in `requirements.txt`.
- **`yfinance` download fails or rate-limits:** Yahoo occasionally throttles requests. Wait a moment and retry, or upgrade with `pip install --upgrade yfinance`.
- **Slow analysis:** The grid search fits up to 9 models on several years of daily data. This is expected to take a few seconds to a minute depending on the stock.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web app framework / UI |
| `yfinance` | Downloads historical price data from Yahoo Finance |
| `statsmodels` | ARIMA model fitting |
| `scikit-learn` | Error metrics (MAE, RMSE, MAPE) |
| `pandas`, `numpy` | Data handling and numerics |
| `plotly` | Charting |

---

## Contributing

Contributions of all kinds are welcome! Whether it's a bug fix, a new feature, or a documentation improvement, feel free to open issues and pull requests.

### Future Development Ideas

This project is actively open to development and expansion. Contributions in the following directions are especially valued:

- **Seasonal models (SARIMA):** The current model runs with `d = 0` and effectively fits an ARMA model to raw prices. Adding models that capture seasonality — such as **SARIMA** (Seasonal ARIMA) — could improve forecast quality on series with recurring periodic patterns. Introducing the seasonal parameters `(P, D, Q, s)` is a natural extension.
- **Automatic parameter selection:** Integrating `pmdarima.auto_arima` to automatically select the `d` parameter and seasonal terms.
- **Differencing support (`d ≥ 1`):** Adding differencing to improve performance on strongly trending series.
- **Alternative models:** Optionally offering different forecasting approaches such as Prophet, LSTM, or GARCH.
- **Additional metrics and visualizations:** New performance metrics, residual analysis plots, or model comparison panels.
- **Performance improvements:** Speeding up the grid search through parallelization or caching.


## Disclaimer

This application is for **educational and informational purposes only**. ARIMA forecasts of financial markets are inherently uncertain, and short-horizon price predictions should **not** be used as the basis for investment decisions. Past performance and backtested accuracy do not guarantee future results.
