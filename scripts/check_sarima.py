#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check SARIMA on every KPI of the demo data: chosen model, test error,
error of the no-model reference, and whether the 12-month forecast is flat.

Usage, from the project folder (next to app/ and files/), env `tsf` active:
    python scripts/check_sarima.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import sarima_model as sm  # noqa: E402

warnings.filterwarnings("ignore")  # statsmodels convergence chatter

DATA = os.path.join(ROOT, "files", "telecom_kpis_synthetic.xlsx")
HORIZON = 12

df = pd.read_excel(DATA, index_col=0, parse_dates=[0])
rows = []
for kpi in df.columns:
    ts = df[kpi].dropna()
    n = int(len(ts) * 0.80)
    train, test = ts[:n], ts[n:]
    p, d, q, P, D, Q = sm.select_orders(train)
    fitted = sm.sarima_training(train, p, d, q, P, D, Q)
    test_mape = sm.accuracy_metrics(sm.sarima_forecasting(test, fitted), test)[2]
    future = sm.future_forecast(ts, p, d, q, P, D, Q, HORIZON)
    # "Flat": the forecast moves less than a quarter as much as the
    # series did over its last 12 months
    last_year = ts[-12:].max() - ts[-12:].min()
    movement = (future.max() - future.min()) / last_year if last_year else np.nan
    rows.append({
        "KPI": kpi,
        "Seasonal strength": round(sm.seasonal_strength(train), 2),
        "Model": sm.model_label(p, d, q, P, D, Q),
        "Test error %": round(test_mape, 2),
        "Reference error %": round(sm.seasonal_naive_mape(ts), 2),
        "Forecast movement": round(movement, 2),
        "Flat?": "yes" if movement < 0.25 else "no",
    })

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 10)
print(pd.DataFrame(rows).to_string(index=False))
