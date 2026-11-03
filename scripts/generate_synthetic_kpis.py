#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the synthetic dataset used by the public demo of the app.

The original app was built on confidential KPI data from a mobile
operator, which cannot be published. This script builds a fictional
dataset with the same structure (monthly dates as the first column, one
column per KPI) so the app runs unchanged and anyone can reproduce it.

Each KPI is designed to exercise a different part of the app:

| KPI                           | Behaviour                                         |
|-------------------------------|---------------------------------------------------|
| Active subscribers (k)        | S-shaped growth that saturates, mild seasonality  |
| Data traffic (TB)             | Exponential growth, multiplicative seasonality    |
| Voice traffic (M minutes)     | Peaks in 2018 then declines as usage moves to data|
| ARPU (EUR)                    | Slow decline with noise                           |
| Mobile money transactions (k) | Launched mid-2017 (missing before), level shift   |
|                               | in 2020                                           |
| Churn rate                    | Stationary, mean-reverting, January bump          |
| Customer care calls (k)       | Tracked from 2016, outage spikes, seasonal        |
| Revenue (M EUR)               | Built from the other KPIs: trend + seasonality    |

Outputs (paths relative to the project root, next to app/):
- files/telecom_kpis_synthetic.xlsx : the data the app lists and loads
- kpi.ods                           : KPI descriptions (tooltips) and empty
                                      SARIMA / LSTM saved-parameter sheets

Usage (from anywhere):
    python scripts/generate_synthetic_kpis.py

The random seed is fixed, so the output is identical on every run.
Note: running it again overwrites kpi.ods, including any model
parameters saved from the app.
"""

import os

import numpy as np
import pandas as pd


SEED = 42
START = "2015-01-01"
N_MONTHS = 120  # Jan 2015 - Dec 2024

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "files", "telecom_kpis_synthetic.xlsx")
KPI_PATH = os.path.join(ROOT, "kpi.ods")


def seasonal_profile(month, peaks, base=0.0):
    """Monthly multipliers: `peaks` maps a month (1-12) to an extra
    amplitude, smoothed over the neighbouring months."""
    profile = np.full(len(month), base, dtype=float)
    for peak_month, amplitude in peaks.items():
        distance = np.minimum(np.abs(month - peak_month), 12 - np.abs(month - peak_month))
        profile += amplitude * np.exp(-0.5 * distance ** 2)
    return profile


def ar1_noise(rng, n, phi, sigma):
    """Autocorrelated noise: real KPIs rarely have independent errors."""
    e = rng.normal(0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=N_MONTHS, freq="MS")
    t = np.arange(N_MONTHS)
    month = dates.month.values
    years = t / 12.0

    # Active subscribers (thousands): logistic growth towards ~11.5M
    subscribers = 5200 + 6300 / (1 + np.exp(-(years - 4.0) / 1.3))
    subscribers *= 1 + seasonal_profile(month, {12: 0.012, 8: 0.006})
    subscribers *= 1 + ar1_noise(rng, N_MONTHS, 0.6, 0.004)

    # Data traffic (TB): ~35% yearly growth, peaks in summer and December
    data_traffic = 900 * np.exp(0.30 * years)
    data_traffic *= 1 + seasonal_profile(month, {8: 0.10, 12: 0.07, 2: -0.05})
    data_traffic *= 1 + ar1_noise(rng, N_MONTHS, 0.5, 0.025)

    # Voice traffic (million minutes): grows until 2018, then declines
    voice = 2600 + 140 * years - 22 * np.maximum(years - 3.5, 0) ** 2
    voice *= 1 + seasonal_profile(month, {12: 0.06, 4: 0.03})
    voice *= 1 + ar1_noise(rng, N_MONTHS, 0.5, 0.012)

    # ARPU (EUR): slow erosion from price competition, noisy
    arpu = 5.4 - 0.11 * years + ar1_noise(rng, N_MONTHS, 0.4, 0.07)
    arpu += seasonal_profile(month, {12: 0.12})

    # Mobile money transactions (thousands): launched July 2017,
    # adoption S-curve, jump from April 2020 (contactless payments)
    launch = 30
    mm_t = np.maximum(t - launch, 0) / 12.0
    mobile_money = 4000 / (1 + np.exp(-(mm_t - 2.2) / 0.7))
    mobile_money += np.where(t >= 63, 1100, 0)
    mobile_money *= 1 + seasonal_profile(month, {12: 0.08, 6: 0.04})
    mobile_money *= 1 + ar1_noise(rng, N_MONTHS, 0.5, 0.03)
    mobile_money = mobile_money.astype(float)
    mobile_money[:launch] = np.nan

    # Churn rate (% of subscribers leaving per month): stationary
    churn = 2.4 + ar1_noise(rng, N_MONTHS, 0.55, 0.12)
    churn += seasonal_profile(month, {1: 0.35})

    # Customer care calls (thousands): tracked from January 2016,
    # seasonal, with a few network-outage spikes
    calls = 410 - 6 * years + 40 * seasonal_profile(month, {1: 1.0, 9: 0.6})
    calls += ar1_noise(rng, N_MONTHS, 0.5, 9)
    for outage in (27, 58, 97):
        calls[outage] += rng.uniform(80, 130)
    calls = calls.astype(float)
    calls[:12] = np.nan

    # Revenue (million EUR): subscribers x ARPU plus mobile money fees
    mm_fees = np.nan_to_num(mobile_money) * 0.0009
    revenue = subscribers * arpu / 1000 + mm_fees
    revenue *= 1 + ar1_noise(rng, N_MONTHS, 0.3, 0.008)

    df = pd.DataFrame(
        {
            "Active subscribers (k)": subscribers,
            "Data traffic (TB)": data_traffic,
            "Voice traffic (M minutes)": voice,
            "ARPU (EUR)": arpu,
            "Mobile money transactions (k)": mobile_money,
            "Churn rate": churn,
            "Customer care calls (k)": calls,
            "Revenue (M EUR)": revenue,
        },
        index=pd.Index(dates, name="Date"),
    )
    return df.round(2)


DESCRIPTIONS = {
    "Active subscribers (k)": "Subscribers with at least one billable event in the last 90 days, in thousands.",
    "Data traffic (TB)": "Total mobile data volume carried on the network over the month, in terabytes.",
    "Voice traffic (M minutes)": "Total outgoing voice minutes over the month, in millions.",
    "ARPU (EUR)": "Average revenue per user: monthly revenue divided by active subscribers, in euros.",
    "Mobile money transactions (k)": "Mobile money transactions over the month, in thousands. Service launched in July 2017.",
    "Churn rate": "Percentage of active subscribers who left the network during the month.",
    "Customer care calls (k)": "Calls received by customer care over the month, in thousands. Tracked since January 2016.",
    "Revenue (M EUR)": "Total monthly service revenue, in millions of euros.",
}

SARIMA_COLUMNS = ["KPI", "p", "d", "q", "P", "D", "Q"]
LSTM_COLUMNS = ["KPI", "Layers", "Units", "Optimizer", "Sequence length",
                "Epochs", "Batch size", "Verbose", "Validation split"]


def build_kpi_sheets(kpis):
    """Sheet order matters to the app: Description, SARIMA, LSTM.
    Parameter cells start empty, so the app uses its defaults until the
    user saves parameters from the Model tab."""
    description = pd.DataFrame({"KPI": kpis,
                                "Description": [DESCRIPTIONS[k] for k in kpis]})
    sarima = pd.DataFrame({c: kpis if c == "KPI" else [None] * len(kpis)
                           for c in SARIMA_COLUMNS})
    lstm = pd.DataFrame({c: kpis if c == "KPI" else [None] * len(kpis)
                         for c in LSTM_COLUMNS})
    return {"Description": description, "SARIMA": sarima, "LSTM": lstm}


def main():
    df = generate()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_excel(DATA_PATH, engine="openpyxl")

    with pd.ExcelWriter(KPI_PATH, engine="odf") as writer:
        for name, sheet in build_kpi_sheets(list(df.columns)).items():
            sheet.to_excel(writer, sheet_name=name, index=False)

    print(f"Wrote {DATA_PATH} ({df.shape[0]} months x {df.shape[1]} KPIs)")
    print(f"Wrote {KPI_PATH}")


if __name__ == "__main__":
    main()
