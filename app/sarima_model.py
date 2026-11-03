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
  on the series differenced d times, as the number of consecutive
  significant lags from lag 1 (where the PACF / ACF cuts off), capped at 3.
- P and Q were read from lags 1-15 of the decomposition residuals, not at
  the seasonal lag. They are now 1 if the PACF / ACF of the differenced
  series is significant at lag 12, else 0. D (from the residuals) is
  capped at 1.
- Typical result: SARIMA(1,1,1)(1,0,1,12) instead of SARIMA(1,1,9)(4,0,1,12),
  which fits in seconds instead of minutes and overfits far less.

Forecast fixes:
- the test forecast was shifted by one month (each forecast was compared
  with the previous month's actual value); the training fit in the plot too
- future forecasts came from the model fitted on the first 80% only, so
  they ignored the most recent 20% of the data; the model is now refitted
  on the whole series first (future_forecast)
- baselines (seasonal_naive_accuracy, persistence_accuracy) to put the
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


MAX_D = 2           # maximum order of differencing tried by differencing()
MAX_PQ = 3          # maximum non-seasonal AR / MA order
SEASON = 12         # monthly data: yearly seasonality


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


#def pdq(modelchoice, timeseries):
#
#    # Define the p, d, q ranges
#    p = range(0, 5)
#    d = range(0, 2)
#    q = range(0, 5)
#    # Generate all combinations of p, d, q
#    pdq = list(itertools.product(p, d, q))
#    # Find the best parameters
#    best_aic = np.inf
#    best_pdq = None
#
#    for param in pdq:
#        try:
#            model = ARIMA(timeseries, order=param)
#            results = model.fit()
#            #print(results.aic)
#            if results.aic < best_aic:
#                best_aic = results.aic
#                best_pdq = param
#        except:
#            continue
#    newp,newd,newq = best_pdq
#    if modelchoice == "arma":
#        newd = 0
#    if modelchoice == "arima":
#        if newd == 0:
#            newd = 1
#    return (newp,newd,newq)


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


def _difference(timeseries, d):
    ts = timeseries.dropna()
    for _ in range(d):
        ts = ts.diff().dropna()
    return ts


def select_pdq(timeseries):
    # d first, then p and q on the series made stationary by differencing:
    # on the raw series a trend makes nearly every lag look significant
    d = differencing(timeseries)
    stationary = _difference(timeseries, d)
    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(stationary)
    p = _cutoff(_significant(pacf_values, pacf_confint), MAX_PQ)
    q = _cutoff(_significant(acf_values, acf_confint), MAX_PQ)
    return (p, d, q)


def select_seasonal_pdq(timeseries, residual, d):
    # D: same stationarity check as before, on the decomposition residuals,
    # but at most one seasonal difference
    D = min(differencing(residual), 1)
    # P and Q: is the seasonal lag (12) significant on the differenced series?
    stationary = _difference(timeseries, d)
    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(stationary, max_lags=SEASON)
    if len(acf_values) <= SEASON:
        return (0, D, 0)  # too short to see a full season
    P = int(_significant(pacf_values, pacf_confint)[SEASON])
    Q = int(_significant(acf_values, acf_confint)[SEASON])
    return (P, D, Q)


def select_orders(train, residual):
    """Automatic SARIMA orders (p, d, q, P, D, Q) for a training series and
    the residuals of its seasonal decomposition."""
    (p, d, q) = select_pdq(train)
    (P, D, Q) = select_seasonal_pdq(train, residual, d)
    return (p, d, q, P, D, Q)


def sarima_training(timeseries, p, d, q, P, D, Q):
    order = (p, d, q)
    s = 12
    seasonal_order = (P, D, Q, s)
    model = SARIMAX(timeseries, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    return fitted_model


def sarima_forecasting(test, fitted_model):
    # Forecast exactly the test period: step 1 is the first test month.
    # (It used to forecast len(test)+1 steps and drop the first one, so each
    # forecast was compared with the previous month's actual value.)
    forecast = fitted_model.get_forecast(steps=len(test))
    return pd.Series(np.asarray(forecast.predicted_mean), index=test.index)


def next_months(last_date, horizon):
    # Monthly dates following `last_date`
    return pd.DatetimeIndex([last_date + pd.DateOffset(months=k) for k in range(1, horizon + 1)])


def future_forecast(timeseries, p, d, q, P, D, Q, horizon):
    """Refit the model on the WHOLE series and forecast the `horizon` months
    after its last date. The model fitted on the training split only is for
    testing: forecasting the future with it would ignore the test period
    (the most recent 20% of the data) and start from the wrong point."""
    ts = timeseries.dropna()
    if horizon is None or horizon < 1:
        return pd.Series(dtype=float)
    full_model = sarima_training(ts, p, d, q, P, D, Q)
    values = np.asarray(full_model.forecast(horizon))
    return pd.Series(values, index=next_months(ts.index[-1], horizon))


def sarima_predictions(fitted_model, timeseries, train, forecast, horizon, p, d, q, P, D, Q):
    timeseries = timeseries.dropna()
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, mode='lines', name=f"{timeseries.name}"))
    # One-step-ahead fit on the training months. The first d + 12*D values
    # are start-up values of the differencing, not real predictions.
    burn_in = d + 12 * D
    fitted = pd.Series(np.asarray(fitted_model.fittedvalues), index=train.index)[burn_in:]
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted.values, mode='lines', name="Train predictions"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name="Test predictions"))
    future = future_forecast(timeseries, p, d, q, P, D, Q, horizon)
    if not future.empty:
        fig.add_trace(go.Scatter(x=future.index, y=future.values, mode='lines', name="Future predictions"))
    fig.update_layout(yaxis_title=f"{train.name}", xaxis_title="Date", template='plotly_white')
    fig.update_layout(title_text=f"SARIMA (({p}, {d}, {q}),({P}, {D}, {Q})) plot of {train.name}")
    return fig


def seasonal_naive_accuracy(timeseries, train_share=0.80, season=SEASON):
    """Accuracy (100 - MAPE) of the "same month last year" forecast on the
    same test split as SARIMA: a reference that needs no model at all."""
    ts = np.asarray(timeseries.dropna(), dtype=float)
    n = int(len(ts) * train_share)
    if n < season or n >= len(ts):
        return np.nan
    test = ts[n:]
    naive = np.array([ts[n - season + (i % season)] for i in range(len(test))])
    return 100 - accuracy_metrics(naive, test)[2]


def persistence_accuracy(timeseries, train_share=0.80):
    """Accuracy of "next month = this month's actual value", one month
    ahead over the test split: the reference for a one-step-ahead test such
    as the LSTM's."""
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
