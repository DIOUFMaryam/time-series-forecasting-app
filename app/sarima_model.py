#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:38:24 2024

@author: diouf.maryam5100@gmail.com

Library-compatibility updates:
- plt.stem(..., use_line_collection=True): the argument was deprecated in
  matplotlib 3.6 and removed in 3.8 (TypeError). Stems always use a
  LineCollection now, so it is simply dropped.
- plt.show() removed: the app renders with the non-interactive Agg backend,
  where show() does nothing except emit a warning.

Stationarity / differencing fix:
- KPSS used automatic lag selection. On a differenced series its long-run
  variance estimate can come out negative, and arch then raises
  "Residuals are all zero and so automatic bandwidth selection cannot be
  used". KPSS now uses a fixed number of lags (same rule as the Data tab),
  whose estimate is never negative.
- differencing() kept differencing until the tests said "stationary", with
  no upper limit. It now stops at d = 2 (more is never useful for SARIMA).
- A test that cannot run on a series no longer crashes the app: it simply
  doesn't count as evidence of stationarity.

Order selection fix (SARIMA was very slow):
- p and q counted EVERY significant lag of the undifferenced series, while
  the comment said "take the first significant lag". On a trending series
  almost every lag is significant, so q reached 8-10. They are now read
  on the differenced series, as the number of consecutive significant
  lags from lag 1 (where the PACF / ACF cuts off), capped at 3.
- Typical result: SARIMA(1,1,1)(1,0,1,12) instead of SARIMA(1,1,9)(4,0,1,12),
  which fits in seconds instead of minutes and overfits far less.

Flat forecast fix (Nov 2026). Data traffic, ARPU, mobile money, churn and
customer care calls got flat future forecasts, for three reasons:
- No constant: SARIMAX has no intercept unless asked. With d = 1 that
  means no drift, so a trending KPI is forecast flat; with d = 0 the AR
  terms must mimic the mean and end up close to a random walk. A constant
  is now included when d + D <= 1 (a mean when d + D = 0, a drift when
  d + D = 1), the same rule as R's auto.arima. With two differences a
  constant would be a quadratic trend, so none.
- D was always 0: it was decided by testing the decomposition RESIDUALS
  for stationarity, but the residuals are what is left once seasonality
  is removed, so they never call for a seasonal difference. D is now 1
  when the series' seasonal strength is above 0.64 (Hyndman &
  Athanasopoulos' rule), measured on the log scale for positive KPIs
  (growing seasonal swings) and with an outlier-resistant spread (outage
  spikes), and the regular difference d is chosen after it.
- P and Q were read against Bartlett's confidence band, which widens with
  each lag and is wide at lag 12 on a differenced series, so the seasonal
  lag was rarely judged significant. Lag 12 is now compared with the
  white-noise band 1.96 / sqrt(n), the one the Data tab uses.
- Churn and customer care calls were still flat after these fixes: their
  seasonality is real but below the seasonal-difference threshold, and the
  seasonal AR/MA terms fade it out. Without a seasonal difference, the
  yearly pattern is now added as sine/cosine terms of the month (Fourier
  terms), their number (0-3) chosen by AICc (sarima_training).

Speed fix (Nov 2026): choosing the number of waves takes 4 fits, and the
app was fitting SARIMA 3 times per KPI (training data, whole series for
the plot, whole series again for the Forecast tab), so up to 12 fits. The
waves are now chosen once, on the training data, and passed on (K
argument of future_forecast); the whole-series fit is done once and
reused by the plot and the Forecast tab (forecast_from).

Forecast fixes:
- the test forecast was shifted by one month (each forecast was compared
  with the previous month's actual value); the training fit in the plot too
- future forecasts came from the model fitted on the first 80% only, so
  they ignored the most recent 20% of the data; the model is now refitted
  on the whole series first (future_forecast)
- baselines (seasonal_naive_mape, persistence_accuracy) to put the
  accuracy figures in context
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # web app: render figures to images, never open Tk windows
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from arch.unitroot import PhillipsPerron, KPSS, DFGLS

from sklearn.metrics import mean_absolute_error, mean_squared_error
#import pmdarima as pm
#import itertools


MAX_D = 2           # maximum total differencing d + D
MAX_PQ = 3          # maximum non-seasonal AR / MA order
SEASON = 12         # monthly data: yearly seasonality
SEASONAL_STRENGTH_THRESHOLD = 0.64  # above it: one seasonal difference (D = 1)
MAX_FOURIER = 3      # maximum sine/cosine pairs for a yearly pattern when D = 0


def pacf_plot(timeseries):
    sample_size = len(timeseries)
    max_nlags = (sample_size // 2) - 1
    # Set nlags to a valid value
    nlags = min(15, max_nlags)
    fig, ax = plt.subplots()
    plot_pacf(timeseries, lags=nlags, ax=ax)
    return fig


def acf_plot(timeseries):
    sample_size = len(timeseries)
    max_nlags = (sample_size // 2) - 1
    # Set nlags to a valid value
    nlags = min(15, max_nlags)
    fig, ax = plt.subplots()
    plot_acf(timeseries, lags=nlags, ax=ax)
    return fig


def seasonal_acf_plot(timeseries):
    (trend,seasonal,residual) = series_decompose(timeseries)
    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(residual)
    fig,ax = plt.subplots()
    plt.stem(range(len(acf_values)), acf_values)
    plt.fill_between(range(len(acf_values)), acf_confint[:, 0] - acf_values, acf_confint[:, 1] - acf_values, color='r', alpha=0.3)
    plt.title('Seasonal ACF')
    return fig


def seasonal_pacf_plot(timeseries):
    (trend,seasonal,residual) = series_decompose(timeseries)
    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(residual)
    fig,ax = plt.subplots()
    plt.stem(range(len(pacf_values)), pacf_values)
    plt.fill_between(range(len(pacf_values)), pacf_confint[:, 0] - pacf_values, pacf_confint[:, 1] - pacf_values, color='r', alpha=0.3)
    plt.title('Seasonal PACF')
    return fig


def series_decompose(timeseries):
    decomposition = seasonal_decompose(timeseries.dropna(), model='additive', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return (trend,seasonal,residual)


def acf_pacf_values(timeseries, max_lags=15):
    sample_size = len(timeseries.dropna())
    max_nlags = (sample_size // 2) - 1
    # Set nlags to a valid value
    nlags = min(max_lags, max_nlags)
    acf_values, acf_confint = acf(timeseries.dropna(), alpha=0.05, nlags=nlags, fft=False)
    pacf_values, pacf_confint = pacf(timeseries.dropna(), alpha=0.05, nlags=nlags)
    return (acf_values, acf_confint, pacf_values, pacf_confint)


def test_lags(nobs):
    # Schwert's rule, the lag count used by the tests shown in the Data tab
    return max(1, int(12 * (nobs / 100) ** (1 / 4)))


def _pvalue(test):
    # p-value of a unit-root test, or NaN if the test cannot run on this
    # series (too short, constant, ...). NaN fails every "< 0.05" and
    # ">= 0.05" comparison, so a test that can't run counts as no evidence.
    try:
        return test().pvalue
    except Exception:
        return np.nan


def stationarity_test(timeseries):
    ts = timeseries.dropna()
    dfgls = _pvalue(lambda: DFGLS(ts))
    pp = _pvalue(lambda: PhillipsPerron(ts, test_type="rho"))
    # Fixed lags: with automatic lag selection, KPSS can estimate a negative
    # long-run variance on differenced series and raise an error
    kpss = _pvalue(lambda: KPSS(ts, lags=test_lags(len(ts))))
    if ((dfgls < 0.05 and pp < 0.05) or (dfgls < 0.05 and kpss >= 0.05) or (pp < 0.05 and kpss >= 0.05)):
        s = "Yes"
    else:
        s = "No"
    return s


def differencing(timeseries, max_d=MAX_D):
    new_ts = timeseries.dropna()
    d = 0
    while d < max_d and stationarity_test(new_ts) == "No":
        d += 1
        new_ts = new_ts.diff().dropna()
    return d


def _robust_variance(values):
    # Variance estimated from the median absolute deviation: a few outage
    # spikes don't inflate it the way they inflate the usual variance
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    mad = np.median(np.abs(values - np.median(values)))
    return (1.4826 * mad) ** 2


def seasonal_strength(timeseries, season=SEASON):
    """Share of the detrended variation explained by the seasonal pattern,
    0 (none) to 1 (pure seasonality): 1 - Var(residual) / Var(seasonal +
    residual), after Hyndman & Athanasopoulos, Forecasting: Principles and
    Practice. Measured on the log scale when the KPI is positive (seasonal
    swings that grow with the level become constant), with an outlier-
    resistant residual spread."""
    ts = timeseries.dropna().astype(float)
    if len(ts) < 2 * season:
        return 0.0
    if (ts > 0).all():
        ts = np.log(ts)
    decomposition = seasonal_decompose(ts, model='additive', period=season)
    residual = decomposition.resid
    mask = residual.notna()
    total = np.var((decomposition.seasonal + residual)[mask])
    if total == 0:
        return 0.0
    return float(max(0.0, 1 - _robust_variance(residual[mask]) / total))


def seasonal_differencing(timeseries, season=SEASON):
    """D = 1 when the seasonality is strong and there are at least three
    years to learn it from, else 0."""
    ts = timeseries.dropna()
    if len(ts) < 3 * season:
        return 0
    return int(seasonal_strength(ts, season) > SEASONAL_STRENGTH_THRESHOLD)


def _significant(values, confint):
    # statsmodels returns confidence intervals centred on each value;
    # the band around zero has the same half-width
    half_width = confint[:, 1] - values
    return np.abs(values) > half_width


def _cutoff(significant, max_order):
    # Number of consecutive significant lags starting at lag 1, i.e. the
    # lag after which the (P)ACF cuts off
    order = 0
    for lag in range(1, len(significant)):
        if not significant[lag] or order == max_order:
            break
        order += 1
    return order


def _difference(timeseries, d, D=0, season=SEASON):
    # Seasonal difference(s) first, then regular ones
    ts = timeseries.dropna()
    for _ in range(D):
        ts = ts.diff(season).dropna()
    for _ in range(d):
        ts = ts.diff().dropna()
    return ts


def select_orders(train, residual=None):
    """Automatic SARIMA orders (p, d, q, P, D, Q) for a training series.

    1. D from the strength of the seasonality (seasonal_differencing)
    2. d from the stationarity tests, on the seasonally differenced series,
       so that d + D <= MAX_D
    3. p and q where the PACF / ACF of the differenced series cuts off
    4. P and Q: is lag 12 of that PACF / ACF outside the white-noise band?

    `residual` (decomposition residuals) is no longer used; the argument is
    kept so existing callers still work."""
    D = seasonal_differencing(train)
    d = differencing(_difference(train, 0, D), max_d=MAX_D - D)
    stationary = _difference(train, d, D)

    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(stationary, max_lags=SEASON)
    p = _cutoff(_significant(pacf_values, pacf_confint), MAX_PQ)
    q = _cutoff(_significant(acf_values, acf_confint), MAX_PQ)

    if len(acf_values) <= SEASON:
        return (p, d, q, 0, D, 0)  # too short to see a full season
    band = 1.96 / np.sqrt(len(stationary))
    P = int(abs(pacf_values[SEASON]) > band)
    Q = int(abs(acf_values[SEASON]) > band)
    return (p, d, q, P, D, Q)


def trend_term(d, D):
    """Constant of the model: a mean when nothing is differenced, a drift
    (a steady change per month, or per year after a seasonal difference)
    when the series is differenced once, none beyond (it would become a
    quadratic trend). Same rule as R's auto.arima."""
    return "c" if d + D <= 1 else "n"


def fourier_terms(dates, K, season=SEASON):
    """Yearly seasonal pattern as K pairs of sine/cosine waves of the month
    of the year (1 to K cycles per year). Tied to the calendar, so the
    same columns can be computed for any future month."""
    month = pd.DatetimeIndex(dates).month.values - 1
    columns = {}
    for k in range(1, K + 1):
        angle = 2 * np.pi * k * month / season
        columns[f"sin{k}"] = np.sin(angle)
        columns[f"cos{k}"] = np.cos(angle)
    return pd.DataFrame(columns, index=pd.DatetimeIndex(dates))


def fourier_k(fitted_model):
    """Number of sine/cosine pairs a fitted model uses (0 = none)."""
    exog = fitted_model.model.exog
    return 0 if exog is None else exog.shape[1] // 2


def model_label(p, d, q, P, D, Q, K=0):
    extra = {0: " with mean", 1: " with drift"}.get(d + D, "")
    if K:
        extra += f" + {K} yearly wave(s)"
    return f"SARIMA(({p}, {d}, {q}),({P}, {D}, {Q})){extra}"


def _fit(timeseries, p, d, q, P, D, Q, K):
    exog = fourier_terms(timeseries.index, K) if K else None
    model = SARIMAX(timeseries, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, SEASON),
                    trend=trend_term(d, D))
    return model.fit(disp=False)


def sarima_training(timeseries, p, d, q, P, D, Q, K=None):
    """Fit SARIMA. Without a seasonal difference (D = 0), a stable yearly
    pattern (a January peak every year...) is added as K sine/cosine pairs,
    K from 0 to MAX_FOURIER chosen by AICc (dynamic harmonic regression,
    Hyndman & Athanasopoulos). The seasonal AR/MA terms alone tend to fade
    the pattern out over the forecast, which gave flat forecasts for churn
    and customer care calls. With D = 1 the seasonal difference already
    repeats last year's pattern, so K = 0. Pass K to force a value."""
    if K is None:
        if D > 0 or len(timeseries) < 3 * SEASON:
            K = 0
        else:
            fits = [_fit(timeseries, p, d, q, P, D, Q, k) for k in range(MAX_FOURIER + 1)]
            return min(fits, key=lambda f: f.aicc)
    return _fit(timeseries, p, d, q, P, D, Q, K)


def _future_exog(fitted_model, dates):
    K = fourier_k(fitted_model)
    return fourier_terms(dates, K) if K else None


def sarima_forecasting(test, fitted_model):
    # Forecast exactly the test period: step 1 is the first test month.
    # (It used to forecast len(test)+1 steps and drop the first one, so each
    # forecast was compared with the previous month's actual value.)
    forecast = fitted_model.get_forecast(steps=len(test), exog=_future_exog(fitted_model, test.index))
    return pd.Series(np.asarray(forecast.predicted_mean), index=test.index)


def next_months(last_date, horizon):
    # Monthly dates following `last_date`
    return pd.DatetimeIndex([last_date + pd.DateOffset(months=k) for k in range(1, horizon + 1)])


def forecast_from(fitted_model, last_date, horizon):
    """Forecast the `horizon` months after `last_date` with a model already
    fitted up to that date (no refit: changing the horizon is instant)."""
    if horizon is None or horizon < 1:
        return pd.Series(dtype=float)
    dates = next_months(last_date, horizon)
    values = np.asarray(fitted_model.forecast(horizon, exog=_future_exog(fitted_model, dates)))
    return pd.Series(values, index=dates)


def future_forecast(timeseries, p, d, q, P, D, Q, horizon, K=None):
    """Refit the model on the WHOLE series and forecast the `horizon` months
    after its last date. The model fitted on the training split only is for
    testing: forecasting the future with it would ignore the test period
    (the most recent 20% of the data) and start from the wrong point.
    Pass K (the number of yearly waves the training fit chose, fourier_k)
    to avoid choosing it again: that takes 4 fits instead of 1."""
    ts = timeseries.dropna()
    if horizon is None or horizon < 1:
        return pd.Series(dtype=float)
    full_model = sarima_training(ts, p, d, q, P, D, Q, K)
    return forecast_from(full_model, ts.index[-1], horizon)


def sarima_predictions(fitted_model, timeseries, train, forecast, future, p, d, q, P, D, Q):
    """Plot of the data, the training fit, the test forecast and `future`
    (a Series from future_forecast / forecast_from, possibly empty)."""
    timeseries = timeseries.dropna()
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, mode='lines', name=f"{timeseries.name}"))
    # One-step-ahead fit on the training months. The first d + 12*D values
    # are start-up values of the differencing, not real predictions.
    burn_in = d + 12 * D
    fitted = pd.Series(np.asarray(fitted_model.fittedvalues), index=train.index)[burn_in:]
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name="Train predictions"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name="Test predictions"))
    if future is not None and not future.empty:
        fig.add_trace(go.Scatter(x=future.index, y=future.values, mode='lines', name="Future predictions"))
    fig.update_layout(yaxis_title=f"{train.name}", xaxis_title="Date", template='plotly_white')
    fig.update_layout(title_text=f"{model_label(p, d, q, P, D, Q, fourier_k(fitted_model))} plot of {train.name}")
    return fig


def seasonal_naive_mape(timeseries, train_share=0.80, season=SEASON):
    """Average % error of the "same month last year" forecast on the same
    test split as SARIMA and the LSTM: a reference that needs no model."""
    ts = np.asarray(timeseries.dropna(), dtype=float)
    n = int(len(ts) * train_share)
    if n < season or n >= len(ts):
        return np.nan
    test = ts[n:]
    naive = np.array([ts[n - season + (i % season)] for i in range(len(test))])
    return accuracy_metrics(naive, test)[2]


def seasonal_naive_accuracy(timeseries, train_share=0.80, season=SEASON):
    """100 - seasonal_naive_mape (kept for older callers)."""
    return 100 - seasonal_naive_mape(timeseries, train_share, season)


def persistence_accuracy(timeseries, train_share=0.80):
    """Accuracy of "next month = this month's actual value", one month
    ahead over the test split."""
    ts = np.asarray(timeseries.dropna(), dtype=float)
    n = int(len(ts) * train_share)
    if n < 1 or n >= len(ts):
        return np.nan
    return 100 - accuracy_metrics(ts[n - 1:-1], ts[n:])[2]


def accuracy_metrics(forecast, test):
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    # Guard against division by zero: exclude points where the actual
    # value is 0 (undefined percentage error) instead of letting them
    # blow up or silently become inf/nan.
    test = np.asarray(test, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    nonzero = test != 0
    if nonzero.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((test[nonzero] - forecast[nonzero]) / test[nonzero])) * 100
    return (mae, mse, mape)
