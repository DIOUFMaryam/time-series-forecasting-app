#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:30:49 2024

@author: diouf.maryam5100@gmail.com

Library-compatibility update:
- make_future_dataframe(freq='m'): month-end string aliases changed in
  pandas 2.2 ('M' -> 'ME', lowercase forms deprecated). A MonthEnd offset
  object means the same thing on every pandas version, so it replaces the
  string. Behaviour is unchanged (month-end dates).
"""
from prophet import Prophet
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def prophet_training(timeseries):
    model_timeseries = timeseries.reset_index().rename(columns={'Date':'ds', timeseries.name:'y'})
    # Normalize the 'y' values
    scaler = MinMaxScaler()
    y_reshaped = np.array(model_timeseries['y']).reshape(-1,1)
    model_timeseries['y'] = scaler.fit_transform(y_reshaped)
    # Train the model
    model = Prophet()
    model.fit(model_timeseries)
    return model, scaler

def prophet_forecasting(model, period, scaler):
    future = model.make_future_dataframe(periods=period+1, freq=pd.offsets.MonthEnd())
    forecast = model.predict(future)
    # Inverse transforming the predictions to the original scale
    forecast['yhat_lower'] = scaler.inverse_transform(forecast[['yhat_lower']])
    forecast['yhat'] = scaler.inverse_transform(forecast[['yhat']])
    forecast['yhat_upper'] = scaler.inverse_transform(forecast[['yhat_upper']])
    return forecast

def prophet_performance(model, scaler):
    # The CV windows used to be hardcoded (initial=366d, period=180d,
    # horizon=180d), which assumes ~2 years of history. Our KPI series
    # are monthly and often shorter than that, so hardcoded windows
    # could leave too little data for the "initial" fit or silently
    # produce zero folds. Scale the windows to the series' own span
    # instead.
    history = model.history
    total_days = (history['ds'].max() - history['ds'].min()).days

    if total_days < 90:
        # Not enough history for a meaningful rolling cross-validation;
        # return empty frames rather than letting Prophet raise.
        return (pd.DataFrame(), pd.DataFrame())

    initial_days = max(60, int(total_days * 0.5))
    remaining_days = max(total_days - initial_days, 60)
    horizon_days = max(30, remaining_days // 2)
    period_days = max(30, remaining_days // 2)

    df_cv = cross_validation(
        model,
        initial=f'{initial_days} days',
        period=f'{period_days} days',
        horizon=f'{horizon_days} days',
    )
    for col in df_cv.columns[1:5]:
        df_cv[col] = scaler.inverse_transform(df_cv[[col]])
    df_performance = performance_metrics(df_cv)
    return (df_cv, df_performance)
