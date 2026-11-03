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


MAX_D = 2  # maximum order of differencing tried by differencing()


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


def acf_pacf_values(timeseries):
    sample_size = len(timeseries.dropna())
    max_nlags = (sample_size // 2) - 1
    # Set nlags to a valid value
    nlags = min(15, max_nlags)
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


def select_pdq(timeseries):
    #model = pm.auto_arima(timeseries, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    # Calculate ACF and PACF
    (acf_values, acf_confint, pacf_values, pacf_confint) = acf_pacf_values(timeseries)
    nb_lags = np.arange(len(acf_values))

    # Identify cut-off lags for p and q
    confint_inf_acf = acf_confint[:, 0] - acf_values
    confint_sup_acf = acf_confint[:, 1] - acf_values
    confint_inf_pacf = pacf_confint[:, 0] - pacf_values
    confint_sup_pacf = pacf_confint[:, 1] - pacf_values

    significant_acf_lags = []
    significant_pacf_lags = []
    for i in nb_lags:
        if acf_values[i] > confint_sup_acf[i] or acf_values[i] < confint_inf_acf[i]:
            significant_acf_lags.append(i)
        if pacf_values[i] > confint_sup_pacf[i] or pacf_values[i] < confint_inf_pacf[i]:
            significant_pacf_lags.append(i)

    # Typically, we take the first significant lag as p or q
    p = len(significant_pacf_lags)-1 if len(significant_pacf_lags) <= 10 else 10
    q = len(significant_acf_lags)-1 if len(significant_acf_lags) <= 10 else 10
    d = differencing(timeseries)
    return (p,d,q)


def sarima_training(timeseries, p, d, q, P, D, Q):
    order = (p, d, q)
    s = 12
    seasonal_order = (P, D, Q, s)
    model = SARIMAX(timeseries, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)
    return fitted_model


def sarima_forecasting(test, fitted_model):
    forecast_steps = len(test)+1
    forecast = fitted_model.get_forecast(steps=forecast_steps)
    #forecast_ci = forecast.conf_int()
    return forecast.predicted_mean[1:]


def sarima_predictions(fitted_model, timeseries, train, forecast, horizon, p, d, q, P, D, Q):
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, mode='lines', name=f"{timeseries.name}"))
    train_predictions = fitted_model.predict(start=0, end=len(train)-1)
    fig.add_trace(go.Scatter(x=train_predictions.index, y=fitted_model.predict(start=1, end=len(train)), mode='lines', name="Train predictions"))
    test_predictions = fitted_model.predict(start=len(train), end=len(train)+len(forecast)-1)
    fig.add_trace(go.Scatter(x=test_predictions.index, y=forecast, mode='lines', name="Test predictions"))
    future_predictions = fitted_model.predict(start=len(train)+len(forecast), end=len(train)+len(forecast)+horizon-1)
    fig.add_trace(go.Scatter(x=future_predictions.index, y=fitted_model.predict(start=len(train)+len(forecast)+1, end=len(train)+len(forecast)+horizon+1), mode='lines', name="Future predictions"))
    fig.update_layout(yaxis_title=f"{train.name}", xaxis_title="Date", template='plotly_white')
    fig.update_layout(title_text=f"SARIMA (({p}, {d}, {q}),({P}, {D}, {Q})) plot of {train.name}")
    return fig


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
