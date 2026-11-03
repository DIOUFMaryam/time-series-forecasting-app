#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:19:32 2024

@author: diouf.maryam5100@gmail.com

Migrated to the Shiny for Python 1.x API:
- @output decorators removed (render functions register by name)
- @reactive.Calc / @reactive.Effect -> @reactive.calc / @reactive.effect
- @session.download -> @render.download (yields bytes)
- ui.nav -> ui.nav_panel
- session.send_input_message -> ui.update_checkbox
- render.ui / render.text no longer return DataFrames

Public demo on synthetic data:
- KPIs missing from kpi.ods (or a missing kpi.ods) fall back to default
  parameters instead of raising a KeyError
- the Forecast tab lists the columns of the loaded file, not a fixed list
- data summary: median/min/max dates no longer need an exact value match
- company logo removed from the header

Deployment:
- paths built from the project folder, not the folder the app starts in
- TSF_READ_ONLY=1 hides the "Save" buttons (parameters, documentation)

Stationarity verdict (Data tab) now uses sarima_model.stationarity_test,
the same fixed-lag rule as the SARIMA differencing step.
"""

from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # web app: render figures to images, never open Tk windows
import matplotlib.pyplot as plt
import numpy as np
import math as m
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import PhillipsPerron, KPSS, DFGLS
from statsmodels.stats.diagnostic import acorr_ljungbox

import sarima_model
import prophet_model
import lstm_model
from ui import DATA_DIR, ROOT_DIR

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import markdown2  # For converting markdown to HTML

from pymannkendall import original_test as mk_test

from io import BytesIO


KPI_FILE = os.path.join(ROOT_DIR, "kpi.ods")
DOC_FILE = os.path.join(ROOT_DIR, "Documentation.md")

# Read-only mode for the public deployment: set the environment variable
# TSF_READ_ONLY=1 so visitors can't overwrite the saved parameters or the
# documentation shared by everyone. Unset (local use): everything editable.
READ_ONLY = os.environ.get("TSF_READ_ONLY", "0") == "1"

# Columns of the saved-parameter sheets in kpi.ods
# (scripts/generate_synthetic_kpis.py creates the file with these headers)
SARIMA_COLUMNS = ["KPI", "p", "d", "q", "P", "D", "Q"]
LSTM_COLUMNS = ["KPI", "Layers", "Units", "Optimizer", "Sequence length",
                "Epochs", "Batch size", "Verbose", "Validation split"]


# Load the kpi.ods file
# The file contains description of the KPIs
# and saved parameters for SARIMA and LSTM models

def load_kpis(file_path=KPI_FILE):
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None, engine='odf')  # Read all sheets
    except FileNotFoundError:
        excel_data = {}
    # Make sure the three sheets exist, so a missing or partial file
    # means "no descriptions / no saved parameters" rather than a crash
    excel_data.setdefault("Description", pd.DataFrame(columns=["KPI", "Description"]))
    excel_data.setdefault("SARIMA", pd.DataFrame(columns=SARIMA_COLUMNS))
    excel_data.setdefault("LSTM", pd.DataFrame(columns=LSTM_COLUMNS))
    return excel_data

params_data = load_kpis()


def saved_params(model_name, kpi):
    """Saved parameters of `kpi` for "SARIMA" or "LSTM", as a list in the
    sheet's column order (without the KPI name), or None when the KPI has
    no row or an incomplete one. None means: use default parameters."""
    sheet = params_data.get(model_name)
    if sheet is None or sheet.empty or "KPI" not in sheet.columns:
        return None
    rows = sheet[sheet["KPI"] == kpi]
    if rows.empty:
        return None
    values = rows.iloc[0].drop("KPI").tolist()
    if any(pd.isna(v) or (isinstance(v, str) and v.strip() == "") for v in values):
        return None
    return values


def store_params(model_name, columns, row):
    """Replace (or add) the row of one KPI in a saved-parameter sheet."""
    sheet = params_data.get(model_name, pd.DataFrame(columns=columns))
    sheet = sheet[sheet["KPI"] != row[0]]
    new_row = pd.DataFrame([row], columns=sheet.columns[:len(row)])
    params_data[model_name] = pd.concat([sheet, new_row], ignore_index=True)


# Function to style table outputs
# bg color, hover, etc

def style_table(df):
    # Round the values
    df = df.round(2)

    # Convert to HTML with custom styling
    html = df.to_html(index=False, classes='table table-striped table-hover', border=0)

    # Additional styling or formatting if needed
    html = html.replace('<table', '<table style="background-color: #f9f9f9; color: #333; border-color: #ddd;"')

    return html


# Read the content of the markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def create_server():

    # Define server logic
    def server(input, output, session):


        # Reactive value to store the uploaded data
        @reactive.calc
        def data():
            file_name = input.file()
            if file_name is None or file_name == "":
                return None
            df = pd.read_excel(os.path.join(DATA_DIR, file_name), index_col=0, parse_dates=[0])
            return df


        # Update column choices based on the uploaded file
        @reactive.effect
        def update_columns():
            df = data()
            if df is not None:
                ui.update_select("column", choices=df.columns.tolist())
            else:
                ui.update_select("column", choices=[])


        # Reactive function to get the selected data
        @reactive.calc
        def selected_data():
            df = data()
            selected_column = input.column()
            if df is None or selected_column is None or selected_column == "":
                return None
            return df[selected_column]


        # Time series plot
        @render_widget
        def timeSeriesPlot():
            df = selected_data()
            if df is None:
                return go.Figure()

            fig = px.line(data_frame=df, x=df.index, y=df, title=f"Time Series plot of {df.name}").update_layout(
                          yaxis_title=f"{df.name}", xaxis_title="Date", template='plotly_white')
            return fig


        # Stationarity plot
        @render_widget
        def StationarityPlot():
            df = selected_data()
            if df is None:
                return go.Figure()

            # Moving average and moving std
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))
            scaled_data = pd.DataFrame(scaled_data)
            rolling_mean = scaled_data.rolling(window=12).mean()
            rolling_std = scaled_data.rolling(window=12).std()

            fig = make_subplots()
            fig.add_trace(go.Scatter(x=df.index, y=scaled_data[:][0], mode='lines', name=f"{df.name}"))
            fig.add_trace(go.Scatter(x=df.index, y=rolling_mean[:][0], mode='lines', name="Moving average"))
            fig.add_trace(go.Scatter(x=df.index, y=rolling_std[:][0], mode='lines', name="Moving std", line=dict(color='green')))
            fig.add_trace(go.Scatter(x=df.index, y=np.zeros(len(df)), mode='lines', name="x axis : y = 0", line=dict(color='black')))
            fig.update_layout(title_text=f"Moving average and moving std plot of {df.name}",
                              yaxis_title=f"{df.name}", xaxis_title="Date", template='plotly_white')
            return fig


        # Time series table
        @render.ui
        def dataTable():
            df = selected_data()
            if df is None:
                return None
            new_df = df.to_frame()
            new_df["Date"] = df.index
            return ui.HTML(style_table(new_df.tail(7).round(2)))


        # Time series data summary
        @render.ui
        def dataSummary():
            df = selected_data()
            if df is None:
                return None
            date = df.index
            values = df.dropna()

            def when(timestamp):
                return timestamp.strftime('%B %Y')

            # The median of an even number of points is the average of the
            # two middle values, which usually isn't in the series: show the
            # date of the closest observation instead of an exact match
            # (the exact match raised an IndexError).
            median = values.median()
            median_date = (values - median).abs().idxmin()

            new_df = pd.DataFrame()
            new_df["Modalities"] = ["Time range", "Number of observations",
                                    "Number of missing values",
                                    "Mean", "Min", "Median", "Max"]
            new_df["Values"] = [(date[0], date[-1]), df.shape[0], df.isnull().sum(),
                                round(values.mean(), 2),
                                f"{round(values.min(), 2)} ({when(values.idxmin())})",
                                f"{round(median, 2)} (closest: {when(median_date)})",
                                f"{round(values.max(), 2)} ({when(values.idxmax())})"]
            return ui.HTML(style_table(new_df))


        # DFGLS stationarity test
        @render.text
        def DFGLSTest():
            df = selected_data()
            if df is None:
                return ""
            nb_lags = int(12*(len(df)/100)**(1/4))
            dfgls_test = DFGLS(df.dropna(), lags=nb_lags)
            return dfgls_test.summary().as_text()


        # PP stationarity test
        @render.text
        def PPTest():
            df = selected_data()
            if df is None:
                return ""
            nb_lags = int(12*(len(df)/100)**(1/4))
            pp_test = PhillipsPerron(df.dropna(), test_type="rho", lags=nb_lags, trend='ct')
            return pp_test.summary().as_text()


        # KPSS stationarity test
        @render.text
        def KPSSTest():
            df = selected_data()
            if df is None:
                return ""
            nb_lags = int(12*(len(df)/100)**(1/4))
            kpss_test = KPSS(df.dropna(), lags=nb_lags, trend='ct')
            return kpss_test.summary().as_text()


        # Parameters for model recommendation
        # data length, % of missing values, stationarity
        @reactive.calc
        def params():
            df = selected_data()
            if df is None:
                return (None, None, None, None, None, None)
            missing_thresholds = {'low': 5, 'medium': 20}
            data_length_thresholds = {'small': 1000, 'medium': 10000}

            real_data_length = df.dropna().shape[0]  # df length without missing values
            data_length = df.shape[0]  # df length
            percentage_missing = round((df.isnull().sum()/df.shape[0])*100, 2)

            # Same decision rule as the SARIMA differencing step
            # (DF-GLS, PP and fixed-lag KPSS, see sarima_model.py)
            stationarity = sarima_model.stationarity_test(df)
            return (real_data_length, data_length, data_length_thresholds, percentage_missing, missing_thresholds, stationarity)


        # Volume of the data
        @render.text
        def dataVolume():
            real_data_length, data_length, _, _, _, stationarity = params()
            if real_data_length is None or stationarity is None:
                return ""
            return f"{real_data_length} out of {data_length} data points"


        # % of missing values
        @render.text
        def missingValues():
            real_data_length, _, _, percentage_missing, _, stationarity = params()
            if real_data_length is None or stationarity is None:
                return ""
            return f"{percentage_missing} %"


        # Is the series stationary
        @render.text
        def stationarity():
            real_data_length, _, _, _, _, stationarity = params()
            if real_data_length is None or stationarity is None:
                return ""
            return stationarity


        # Parameters for model recommendation: trend and seasonality
        @reactive.calc
        def trend_seasonality():
            df = selected_data()

            if df is None:
                return (None, None, None, None, None)

            timeseries = df.dropna()

            # Decompose the time series
            decomposition = seasonal_decompose(timeseries, model='additive')
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Trend detection using Mann-Kendall test
            trend_result = mk_test(timeseries)
            if trend_result.p < 0.01:
                trend_strength = 'strong'
            elif trend_result.p >= 0.01 and trend_result.p < 0.05:
                trend_strength = 'weak'
            else:
                trend_strength = 'none'
            trend_direction = trend_result.trend

            # Significant lags in PACF suggest seasonality
            pacf_values = pacf(timeseries)
            threshold = 1.96 / np.sqrt(len(timeseries))
            seasonal_lags = np.where(np.abs(pacf_values) > threshold)[0] + 1
            if len(seasonal_lags) > 2:
                seasonality_detected = True
                seasonal_period = seasonal_lags[2]
            else:
                seasonality_detected = False
                seasonal_period = 0

            if seasonality_detected:
                if seasonal.std() > 0.1 * residual.std():
                    seasonal_strength = 'strong'
                elif seasonal.std() > 0.05 * residual.std():
                    seasonal_strength = 'weak'
                else:
                    seasonal_strength = 'none'
            else:
                seasonal_strength = 'none'

            return (trend_direction, trend_strength, seasonality_detected, seasonal_period, seasonal_strength)


        def _trend_seasonality_ready():
            values = trend_seasonality()
            return all(v is not None and not (isinstance(v, str) and v == "") for v in values)


        # Trend direction
        @render.text
        def trend():
            if not _trend_seasonality_ready():
                return ""
            return trend_seasonality()[0]


        # Seasonality detected
        @render.text
        def seasonality():
            if not _trend_seasonality_ready():
                return ""
            return "Yes" if trend_seasonality()[2] else "No"


        # Seasonal period
        @render.text
        def period():
            if not _trend_seasonality_ready():
                return ""
            return f"{trend_seasonality()[3]} months"


        # Recommended model
        @reactive.calc
        def select_model():
            if input.column() is None or selected_data() is None:
                return (None, None, None, None, None, None)
            real_data_length, data_length, data_length_thresholds, percentage_missing, missing_thresholds, stationarity = params()
            trend_direction, trend_strength, seasonality_detected, seasonal_period, seasonal_strength = trend_seasonality()

            # Check volume of data
            if data_length < data_length_thresholds['small']:
                data_volume = 'small'
            elif data_length < data_length_thresholds['medium']:
                data_volume = 'medium'
            else:
                data_volume = 'large'

            # Check percentage of missing values
            if percentage_missing < missing_thresholds['low']:
                missing_status = 'low'
            elif percentage_missing < missing_thresholds['medium']:
                missing_status = 'medium'
            else:
                missing_status = 'high'

            # Model selection
            if data_volume == 'small':
                if missing_status == 'low' and (trend_strength != 'strong' or seasonal_strength != 'strong'):
                    model_choice = "SARIMA"
                else:
                    model_choice = "Prophet"
            else:
                if (trend_strength == 'strong' or seasonal_strength == 'strong') and stationarity == 'No':
                    model_choice = "Prophet"
                else:
                    model_choice = "LSTM"
            return (model_choice, data_volume, missing_status, stationarity, trend_strength, seasonal_strength)


        def _model_ready():
            return all(v is not None for v in select_model())


        # Data volume for recommended model
        @render.ui
        def data_volume():
            if not _model_ready():
                return None
            data_volume = select_model()[1]
            if data_volume == "small":
                return ui.HTML(f'<div style="background-color: #FFDDDD; color: #B22222; text-align: center;">Data volume is {data_volume} (< 1000 data points)</div>')
            if data_volume == "medium":
                return ui.HTML(f'<div style="background-color: #FFF9E6; color: #FF8C00; text-align: center;">We have a {data_volume} amount of data (between 1000 and 10000 data points)</div>')
            return ui.HTML(f'<div style="background-color: #DDFFDD; color: #006400; text-align: center;">We have a {data_volume} volume of data (> 10000 data points)</div>')


        # Missing values status
        @render.ui
        def missing_status():
            if not _model_ready():
                return None
            missing_status = select_model()[2]
            if missing_status == "low":
                return ui.HTML(f'<div style="background-color: #DDFFDD; color: #006400; text-align: center;">We have a {missing_status} percentage of missing data (< 5%)</div>')
            if missing_status == "medium":
                return ui.HTML(f'<div style="background-color: #FFF9E6; color: #FF8C00; text-align: center;">A {missing_status} percentage of missing data (between 5% and 20%)</div>')
            return ui.HTML(f'<div style="background-color: #FFDDDD; color: #B22222; text-align: center;">Very {missing_status} percentage of missing data (> 20%)</div>')


        # Trend strength
        @render.ui
        def trend_strength():
            if not _model_ready():
                return None
            trend_strength = select_model()[4]
            if trend_strength == "none":
                return ui.HTML('<div style="background-color: #F5F5F5; color: black; text-align: center;">There is no trend</div>')
            if trend_strength == "weak":
                return ui.HTML(f'<div style="background-color: #D3D3D3; color: black; text-align: center;">The trend is {trend_strength}</div>')
            return ui.HTML(f'<div style="background-color: #A9A9A9; color: black; text-align: center;">The trend is {trend_strength}</div>')


        # Seasonal strength
        @render.ui
        def seasonal_strength():
            if not _model_ready():
                return None
            seasonal_strength = select_model()[5]
            if seasonal_strength == "none":
                return ui.HTML('<div style="background-color: #F5F5F5; color: black; text-align: center;">There is no seasonality</div>')
            if seasonal_strength == "weak":
                return ui.HTML(f'<div style="background-color: #D3D3D3; color: black; text-align: center;">The seasonality is {seasonal_strength}</div>')
            return ui.HTML(f'<div style="background-color: #A9A9A9; color: black; text-align: center;">The seasonality is {seasonal_strength}</div>')


        # Stationarity for recommended model
        @render.ui
        def stationary():
            if not _model_ready():
                return None
            if select_model()[3] == "Yes":
                return ui.HTML('<div style="background-color: #DDFFDD; color: #006400; text-align: center;">Series is stationary</div>')
            return ui.HTML('<div style="background-color: #FFDDDD; color: #B22222; text-align: center;">Series is not stationary</div>')


        # Recommended model
        @render.ui
        def ModelSelection():
            if not _model_ready():
                return None
            return ui.HTML(f'<div style="text-align: center;"><h4>Recommended model : {select_model()[0]}</h4></div>')


        # The two parameter checkboxes are mutually exclusive.
        # One effect per checkbox, each triggered only by its own box,
        # so ticking one unticks the other (the old single effect read
        # both and could immediately untick the box you just ticked).
        @reactive.effect
        @reactive.event(input.model_params)
        def _uncheck_saved_params():
            if input.model_params():
                ui.update_checkbox("saved_params", value=False)

        @reactive.effect
        @reactive.event(input.saved_params)
        def _uncheck_model_params():
            if input.saved_params():
                ui.update_checkbox("model_params", value=False)


        # Let the user choose parameters for SARIMA and LSTM models
        @render.ui
        @reactive.event(input.model_params, input.model_choice)
        def modelParams():
            if input.model_params() and input.model_choice() == "sarima":
                return ui.div(
                          ui.row(
                              ui.column(2, ui.input_numeric("p_sarima", "p", value=1)),
                              ui.column(2, ui.input_numeric("d_sarima", "d", value=1)),
                              ui.column(2, ui.input_numeric("q_sarima", "q", value=1)),
                              ui.column(2, ui.input_numeric("P_sarima", "P", value=1)),
                              ui.column(2, ui.input_numeric("D_sarima", "D", value=1)),
                              ui.column(2, ui.input_numeric("Q_sarima", "Q", value=1)),
                          ),
                          ui.row(
                              ui.column(5,),
                              ui.column(2, ui.div(ui.input_action_button("train_model_btn", ui.HTML('<i class="fa fa-cogs"></i> Train'), class_="btn"))),
                              ui.column(5,)
                          )
                       )
            if input.model_params() and input.model_choice() == "lstm":
                return ui.div(
                          ui.row(
                              ui.column(3, ui.input_numeric("layers", "Number of LSTM layers", value=2)),
                              ui.column(3, ui.input_numeric("units", "Number of units per layer", value=10)),
                              ui.column(3, ui.input_select("optimizer", "Optimizer", choices={"adam": "adam", "nadam": "nadam", "rmsprop": "RMSprop", "sgd": "sgd", "adagrad": "adagrad"})),
                              ui.column(3, ui.input_numeric("seq_length", "Sequence length", value=12)),
                          ),
                          ui.row(
                              ui.column(3, ui.input_numeric("epoch", "Epochs", value=50)),
                              ui.column(3, ui.input_numeric("batchsize", "Batch size", value=16)),
                              ui.column(3, ui.input_slider("verbose", "Verbose", min=0, max=2, value=1)),
                              ui.column(3, ui.input_slider("validation_split", "Validation split", min=0, max=1, value=0.2)),
                          ),
                          ui.div(ui.input_action_button("train_model_btn", "Train"), style="text-align: center;")
                       )
            return None


        # Save model parameters once the user is satisfied with them
        @reactive.effect
        @reactive.event(input.save_params)
        def save_parameters():
            if READ_ONLY:
                return
            selected_column = input.column()
            if input.model_choice() == "sarima":
                store_params("SARIMA", SARIMA_COLUMNS,
                             [selected_column, input.p_sarima(), input.d_sarima(), input.q_sarima(),
                              input.P_sarima(), input.D_sarima(), input.Q_sarima()])
            elif input.model_choice() == "lstm":
                store_params("LSTM", LSTM_COLUMNS,
                             [selected_column, input.layers(), input.units(), input.optimizer(),
                              input.seq_length(), input.epoch(), input.batchsize(),
                              input.verbose(), input.validation_split()])
            else:
                return
            with pd.ExcelWriter(KPI_FILE, engine='odf') as writer:
                for sheet_name, df in params_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)


        # Message when the user picks saved parameters but none exist
        @render.ui
        def saved_params_message():
            missing = ui.HTML('<div style="color: #696969;">No saved parameters found. Going with default parameters.</div>')
            if input.saved_params() and input.model_choice() == "sarima":
                return missing if saved_params("SARIMA", input.column()) is None else None
            if input.saved_params() and input.model_choice() == "lstm":
                return missing if saved_params("LSTM", input.column()) is None else None
            return None


        # ACF / PACF plots (help the user choose p, q, P, Q)
        def _sarima_train_split():
            df = selected_data()
            if df is None or input.model_choice() != "sarima":
                return None
            train_size = int(len(df.dropna()) * 0.80)
            return df.dropna()[:train_size]

        @render.plot
        def sarimaPACF():
            ts = _sarima_train_split()
            return None if ts is None else sarima_model.pacf_plot(ts)

        @render.plot
        def sarimaACF():
            ts = _sarima_train_split()
            return None if ts is None else sarima_model.acf_plot(ts)

        @render.plot
        def sarima_seasonalACF():
            ts = _sarima_train_split()
            return None if ts is None else sarima_model.seasonal_acf_plot(ts)

        @render.plot
        def sarima_seasonalPACF():
            ts = _sarima_train_split()
            return None if ts is None else sarima_model.seasonal_pacf_plot(ts)


        # SARIMA orders for a KPI: saved ones if complete, else estimated
        def _sarima_orders(kpi, train, residual):
            saved = saved_params("SARIMA", kpi)
            if saved is not None:
                return tuple(int(v) for v in saved)
            (p, d, q) = sarima_model.select_pdq(train)
            (P, D, Q) = sarima_model.select_pdq(residual)
            return (p, d, q, P, D, Q)


        # LSTM settings for a KPI: defaults, overridden by saved ones if complete
        def _lstm_settings(kpi, use_saved):
            settings = dict(nb_layers=2, nb_units=10, optimizer='RMSprop', look_back=12,
                            epoch=100, batchsize=1, Verbose=2, Validationsplit=0.2)
            saved = saved_params("LSTM", kpi) if use_saved else None
            if saved is not None:
                s_layers, s_units, s_opt, s_lb, s_epoch, s_bs, s_verbose, s_vs = saved
                settings.update(nb_layers=int(s_layers), nb_units=int(s_units), optimizer=s_opt,
                                look_back=int(s_lb), epoch=int(s_epoch), batchsize=int(s_bs),
                                Verbose=int(s_verbose), Validationsplit=float(s_vs))
            return settings


        # Train the SARIMA model
        @reactive.calc
        def sarima():
            df = selected_data()

            if df is None:
                return (None, None, None, None)

            timeseries = df.dropna()
            train_size = int(len(timeseries) * 0.80)
            train, test = timeseries[:train_size], timeseries[train_size:]
            (trend, seasonal, residual) = sarima_model.series_decompose(train)

            if input.model_params() and input.train_model_btn():
                (p, d, q) = input.p_sarima(), input.d_sarima(), input.q_sarima()
                (P, D, Q) = input.P_sarima(), input.D_sarima(), input.Q_sarima()
            elif input.saved_params():
                p, d, q, P, D, Q = _sarima_orders(timeseries.name, train, residual)
            else:
                (p, d, q) = sarima_model.select_pdq(train)
                (P, D, Q) = sarima_model.select_pdq(residual)

            horizon = input.predictionHorizon()
            if horizon is None or horizon == "":
                horizon = 0

            fitted_model = sarima_model.sarima_training(train, p, d, q, P, D, Q)
            forecast = sarima_model.sarima_forecasting(test, fitted_model)
            fig = sarima_model.sarima_predictions(fitted_model, timeseries, train, forecast, horizon, p, d, q, P, D, Q)

            return (fitted_model, fig, test, forecast)


        # SARIMA predictions plot
        @render_widget
        def sarimaPlot():
            if input.model_choice() != "sarima":
                return None
            (fitted_model, fig, test, forecast) = sarima()
            if fitted_model is None:
                return go.Figure()
            return fig


        # Train the Prophet model
        @reactive.calc
        def prophet():
            df = selected_data()

            if df is None:
                return (None, None)

            timeseries = df
            fitted_model, scaler = prophet_model.prophet_training(timeseries)
            period = input.predictionHorizon()
            if period is None or period == "":
                period = 0
            forecast = prophet_model.prophet_forecasting(fitted_model, period, scaler)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval', line=dict(color='lightblue'), fill=None))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval', line=dict(color='lightblue'), fill='tonexty'))
            fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries, mode='markers', name='Original Data', marker=dict(color='black', size=5)))
            fig.update_layout(title=f"Prophet model plot of {timeseries.name}", xaxis_title='Date', yaxis_title=f"{timeseries.name}", template='plotly_white')

            return (forecast, fig)


        # Prophet predictions plot
        @render_widget
        def prophetPlot():
            if input.model_choice() != "prophet":
                return None
            forecast, fig = prophet()
            if forecast is None or fig is None:
                return go.Figure()
            return fig


        # Train the LSTM model
        @reactive.calc
        def lstm():
            df = selected_data()

            if df is None:
                return (None,) * 11

            timeseries = df.dropna()
            time_indices = timeseries.index
            name = timeseries.name

            # Reshaping the ts
            timeseries = timeseries.values.reshape(-1, 1).astype('float32')

            # User's entry
            if input.model_params() and input.train_model_btn():
                epoch = input.epoch()
                batchsize = input.batchsize()
                Validationsplit = input.validation_split()
                look_back = input.seq_length()
                nb_layers = input.layers()
                nb_units = input.units()
                optimizer = input.optimizer()
                Verbose = input.verbose()
            # Saved parameters if ticked and complete, defaults otherwise
            else:
                s = _lstm_settings(name, use_saved=input.saved_params())
                nb_layers, nb_units, optimizer = s["nb_layers"], s["nb_units"], s["optimizer"]
                look_back, epoch, batchsize = s["look_back"], s["epoch"], s["batchsize"]
                Verbose, Validationsplit = s["Verbose"], s["Validationsplit"]

            # train/test split on RAW (unscaled) values first, so the
            # scaler never sees the test window
            train_size = int(len(timeseries) * 0.80)
            raw_train = timeseries[0:train_size, :]
            raw_test = timeseries[train_size-look_back-1:len(timeseries), :]

            # Fit the scaler on the training split only
            scaler, train = lstm_model.scaling(raw_train)
            test = lstm_model.scale_transform(scaler, raw_test)
            scaled_data = lstm_model.scale_transform(scaler, timeseries)

            trainX, trainY, testX, testY = lstm_model.reshaping(train, test, look_back)

            model = lstm_model.lstm_model(look_back, nb_layers, nb_units, optimizer)
            lstm_model.lstm_training(model, trainX, trainY, epoch, batchsize, Verbose, Validationsplit)

            # Predictions
            trainPredict, testPredict, trainY, testY = lstm_model.lstm_prediction(model, trainX, testX, trainY, testY, scaler)
            last_sequence = testX[-1]
            horizon = input.predictionHorizon()
            if horizon is None or horizon == "":
                horizon = 1
            horizon += 1
            future_predictions = lstm_model.lstm_future_predictions(model, last_sequence, look_back, horizon, scaler)
            future_time_indices = lstm_model.future_timeindices(time_indices, horizon)

            # Visualize the result
            trainPredictPlot, testPredictPlot, fig = lstm_model.visualize_predictions(scaled_data, look_back, time_indices, trainPredict, testPredict, scaler)

            fig.add_trace(go.Scatter(x=future_time_indices, y=future_predictions.flatten(), mode='lines', name='Future values'))
            fig.update_layout(title=f"LSTM model plot of {name}", xaxis_title='Date', yaxis_title=f"{name}", template='plotly_white')

            return (model, trainY, testY, trainPredict, testPredict, time_indices, testX, testPredictPlot, fig, future_predictions, future_time_indices)


        # LSTM predictions plot
        @render_widget
        def lstmPlot():
            if input.model_choice() != "lstm":
                return None
            result = lstm()
            if any(v is None for v in result):
                return go.Figure()
            return result[8]


        # Test results: test data vs test predictions
        @render.ui
        def testResults():
            df = selected_data()

            if df is None:
                return None

            modelchoice = input.model_choice()
            timeseries = df.dropna()

            if modelchoice == "sarima":
                (fitted_model, fig, test, forecast) = sarima()
                newdf = pd.DataFrame()
                newdf["Date"] = test.index
                newdf["Test data (y)"] = test.values
                newdf["Test predictions (yhat)"] = forecast.values
                newdf["Error (y - yhat)"] = newdf["Test data (y)"] - newdf["Test predictions (yhat)"]
                return ui.HTML(style_table(newdf.tail(10)))

            if modelchoice == "prophet":
                fitted_model, scaler = prophet_model.prophet_training(timeseries)
                (df_cv, df_performance) = prophet_model.prophet_performance(fitted_model, scaler)
                if df_cv.empty:
                    return ui.HTML('<div style="color: #696969;">Not enough history for Prophet cross-validation.</div>')
                newdf = pd.DataFrame()
                newdf["Date"] = df_cv.ds
                newdf["Test data (y)"] = df_cv.y
                newdf["Test predictions (yhat)"] = df_cv.yhat
                newdf["Error (y - yhat)"] = newdf["Test data (y)"] - newdf["Test predictions (yhat)"]
                return ui.HTML(style_table(newdf.tail(10)))

            if modelchoice == "lstm":
                (fitted_model, trainY, testY, trainPredict, testPredict, time_indices, testX, testPredictPlot, fig, future_predictions, future_time_indices) = lstm()
                test_size = testY.shape[1]
                newdf = pd.DataFrame()
                newdf["Date"] = time_indices[-test_size:]
                newdf["Test data (y)"] = testY[0]
                newdf["Test predictions (yhat)"] = testPredictPlot[-test_size:, 0]
                newdf["Error (y - yhat)"] = newdf["Test data (y)"] - newdf["Test predictions (yhat)"]
                return ui.HTML(style_table(newdf.iloc[:-1, ].tail(10)))

            return None


        # Breusch-Pagan test for homoscedasticity
        @render.text
        def BPTest():
            if selected_data() is None or input.model_choice() != "sarima":
                return ""
            sarima_fitted_model, _, _, _ = sarima()
            residuals = sarima_fitted_model.resid
            bp_test = sms.het_breuschpagan(residuals, sm.add_constant(np.arange(len(residuals))))
            labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value', 'Null Hypothesis', 'Alternative Hypothesis']
            bp_results = dict(zip(labels, bp_test + ("The residuals have constant variance (homoscedasticity)", "The residuals do not have constant variance (heteroscedasticity)")))
            return '\n'.join([f"{key}: {value}" for key, value in bp_results.items()])


        # Ljung-Box test for autocorrelation
        @render.text
        def LBTest():
            if selected_data() is None or input.model_choice() != "sarima":
                return ""
            sarima_fitted_model, _, _, _ = sarima()
            residuals = sarima_fitted_model.resid
            ljung_box_test = acorr_ljungbox(residuals, lags=[15], return_df=True)
            ljung_box_test['Null Hypothesis'] = 'There is no autocorrelation in the residuals'
            ljung_box_test['Alternative Hypothesis'] = 'There is autocorrelation in the residuals'
            ljung_box_test.index = ['']
            # render.text needs a string, not a DataFrame
            return ljung_box_test.T.to_string()


        # SARIMA accuracy on test data
        @reactive.calc
        def sarima_accuracy():
            if selected_data() is None:
                return None
            (fitted_model, fig, test, forecast) = sarima()
            (mae, mse, mape) = sarima_model.accuracy_metrics(forecast, test)
            metrics = {"Mean Squared Error": mse,
                       "Root Mean Squared Error": m.sqrt(mse),
                       "Mean Absolute Error": mae,
                       "Mean Absolute Percentage Error (%)": mape}
            return pd.DataFrame(metrics, index=[0]).round(2)


        # Prophet accuracy after cross validation
        @reactive.calc
        def prophet_accuracy():
            df = selected_data()
            if df is None:
                return None
            timeseries = df.dropna()
            fitted_model, scaler = prophet_model.prophet_training(timeseries)
            (df_cv, df_performance) = prophet_model.prophet_performance(fitted_model, scaler)
            if df_performance.empty:
                return None

            mae = np.mean(df_performance.mae.values)
            mse = np.mean(df_performance.mse.values)
            rmse = np.mean(df_performance.rmse.values)
            mape = np.mean(df_performance.mae.values)/np.mean(df_cv["y"])*100
            metrics = {"Mean Squared Error": mse,
                       "Root Mean Squared Error": rmse,
                       "Mean Absolute Error": mae,
                       "Mean Absolute Percentage Error (%)": mape}
            return pd.DataFrame(metrics, index=[0]).round(2)


        # LSTM accuracy on test data
        @reactive.calc
        def lstm_accuracy():
            if selected_data() is None:
                return None
            (fitted_model, trainY, testY, trainPredict, testPredict, time_indices, testX, testPredictPlot, fig, future_predictions, future_time_indices) = lstm()
            trainScore, testScore = lstm_model.lstm_eval(fitted_model, trainY, trainPredict, testY, testPredict)
            metrics = {"Mean Squared Error": testScore[0],
                       "Root Mean Squared Error": testScore[1],
                       "Mean Absolute Error": testScore[2],
                       "Mean Absolute Percentage Error (%)": testScore[3]}
            return pd.DataFrame(metrics, index=[0]).round(2)


        # Test metrics for the selected model: MSE, RMSE, MAE, MAPE
        @reactive.calc
        def accuracy():
            if selected_data() is None:
                return None
            modelchoice = input.model_choice()
            if modelchoice == "sarima":
                return sarima_accuracy()
            if modelchoice == "prophet":
                return prophet_accuracy()
            if modelchoice == "lstm":
                return lstm_accuracy()
            return None


        @render.ui
        def accuracyMetrics():
            df = accuracy()
            if df is None or df.empty:
                return None
            return ui.HTML(style_table(df))


        # Model precision, homoscedasticity and autocorrelation
        @reactive.calc
        def isModelValid():
            df = accuracy()
            if df is None or df.empty:
                return (None,) * 6

            error_percentage = df["Mean Absolute Percentage Error (%)"].values[-1]
            acc = round(100 - error_percentage, 2)
            if error_percentage < 5.0:
                error_message = ui.HTML(f'<div style="background-color: #DDFFDD; color: #006400;">Model is {acc}% accurate</div>')
            elif error_percentage < 20.0:
                error_message = ui.HTML(f'<div style="background-color: #FFF9E6; color: #FF8C00;">Model is {acc}% accurate</div>')
            else:
                error_message = ui.HTML(f'<div style="background-color: #FFDDDD; color: #B22222;">Model is {acc}% accurate</div>')
            homoscedasticity = ui.HTML("")
            autocorrelation = ui.HTML("")
            hm, ac = "", ""

            if input.model_choice() == "sarima":
                sarima_fitted_model, _, _, _ = sarima()
                residuals = sarima_fitted_model.resid
                bp_test = sms.het_breuschpagan(residuals, sm.add_constant(np.arange(len(residuals))))
                ljung_box_test = acorr_ljungbox(residuals, lags=[15], return_df=True)
                lm_pvalue = bp_test[1]
                f_pvalue = bp_test[3]
                lb_pvalue = ljung_box_test.lb_pvalue[15]
                if lm_pvalue > 0.05 and f_pvalue > 0.05:
                    homoscedasticity = ui.HTML('<div style="background-color: #DDFFDD; color: #006400;">Homoscedasticity is checked</div>')
                    hm = "Yes"
                else:
                    homoscedasticity = ui.HTML('<div style="background-color: #FFDDDD; color: #B22222;">Homoscedasticity is not checked</div>')
                    hm = "No"
                if lb_pvalue > 0.05:
                    autocorrelation = ui.HTML('<div style="background-color: #DDFFDD; color: #006400;">There is no autocorrelation between residuals</div>')
                    ac = "No"
                else:
                    autocorrelation = ui.HTML('<div style="background-color: #FFDDDD; color: #B22222;">There is autocorrelation between residuals</div>')
                    ac = "Yes"

            return (acc, hm, ac, error_message, homoscedasticity, autocorrelation)


        @render.ui
        def modelValidity():
            (acc, hm, ac, error_message, homoscedasticity, autocorrelation) = isModelValid()
            if acc is None:
                return None
            return ui.TagList(error_message, homoscedasticity, autocorrelation)


        def _accuracy_note(acc, prefix=""):
            if acc < 80.0:
                text = f"{prefix}ccuracy is really low, consider changing parameters to improve it or choose a model with better accuracy."
            elif acc > 95.0:
                text = "Accuracy is very high, predictions are reliable."
            else:
                text = "Accuracy is good but can maybe be improved by changing parameters."
            return ui.HTML(f'<div style="text-align: center; color: #696969;">{text}</div>')


        @render.ui
        def validityMessage():
            (acc, hm, ac, error_message, homoscedasticity, autocorrelation) = isModelValid()
            if acc is None:
                return None

            valid = ui.HTML('<div style="text-align: center; color: #696969;">Model is valid</div>')
            if input.model_choice() == "sarima":
                if hm == "Yes" and ac == "No":
                    return ui.TagList(valid, _accuracy_note(acc, prefix="A"))
                invalid = ui.HTML('<div style="text-align: center; color: #696969;">Model is not valid</div>')
                advice = ui.HTML('<div style="text-align: center; color: #696969;">Change parameters until homoscedasticity is achieved or residuals are no longer autocorrelated. Or consider changing model altogether.</div>')
                return ui.TagList(invalid, advice)
            return ui.TagList(valid, _accuracy_note(acc, prefix="But a"))


        # Information icon next to the selected KPI
        @render.ui
        def description():
            selected_column = input.column()
            kpi_desc = params_data["Description"]
            icon_map = kpi_desc.set_index('KPI')['Description'].to_dict() if not kpi_desc.empty else {}

            if selected_column in icon_map:
                return ui.div(ui.span("ⓘ",
                                      class_="tooltip",
                                      style="cursor: pointer;",
                                      title=f"{icon_map[selected_column]}"))
            return ui.div(ui.span("ⓘ",
                                  style="cursor: pointer;",
                                  title="Information about KPIs here. See documentation for more."))


        # Documentation tab: markdown editor + preview
        @reactive.calc
        def markdown_content():
            return input.markdown_editor()

        @render.ui
        def markdown_preview():
            return ui.HTML(markdown2.markdown(markdown_content()))

        @reactive.effect
        @reactive.event(input.save_btn)
        def save_markdown_file():
            if READ_ONLY:
                return
            with open(DOC_FILE, "w") as file:
                file.write(markdown_content())


        # Predictions made by each model for the selected data
        @reactive.calc
        def predictions_table():
            df = selected_data()

            if df is None:
                return pd.DataFrame()

            timeseries = df.dropna()
            horizon = input.predictionHorizon()

            sarima_fitted_model, _, _, _ = sarima()
            sarima_previsions = sarima_fitted_model.predict(start=0, end=len(timeseries)+horizon)
            index = sarima_previsions.index[:-1]

            prophet_fitted_model, scaler = prophet_model.prophet_training(timeseries)
            prophet_forecast = prophet_model.prophet_forecasting(prophet_fitted_model, horizon, scaler)

            try:
                future_predictions = lstm()[9]
                lstm_forecast = future_predictions[:, 0][-horizon:]
            except lstm_model.TensorFlowUnavailableError:
                # TensorFlow cannot load on this machine: keep SARIMA/Prophet
                lstm_forecast = np.full(horizon, np.nan)

            new_df = pd.DataFrame()
            new_df["Date"] = index[-horizon:]
            new_df["SARIMA"] = sarima_previsions[-horizon:].values
            new_df["Prophet"] = prophet_forecast['yhat'][-horizon:].values
            new_df["LSTM"] = lstm_forecast
            return new_df.round(2)


        def _mape(acc_df):
            # NaN when the model has no usable metrics (e.g. too little
            # history for Prophet cross-validation)
            return np.nan if acc_df is None else acc_df["Mean Absolute Percentage Error (%)"][0]


        def _lstm_mape():
            try:
                return _mape(lstm_accuracy())
            except lstm_model.TensorFlowUnavailableError:
                return np.nan


        # Predictions table, highlighting the most accurate model
        @render.ui
        def predictions():
            df = predictions_table()
            if df is None or df.empty:
                return None
            # Lowest error among the models that ran
            # (LSTM is skipped when TensorFlow is unavailable)
            mapes = {"SARIMA": _mape(sarima_accuracy()),
                     "Prophet": _mape(prophet_accuracy()),
                     "LSTM": _lstm_mape()}
            mapes = {k: v for k, v in mapes.items() if not pd.isna(v)}
            if not mapes:
                return ui.HTML(df.to_html(index=False, classes='table table-striped table-hover', border=0))
            most_accurate_model = min(mapes, key=mapes.get)

            df = df.round(2)
            html = df.to_html(index=False, classes='table table-striped table-hover', border=0)
            html = html.replace(
                f'<th>{most_accurate_model}</th>',
                f'<th style="border: 3px solid green;">{most_accurate_model}</th>'
            )
            for value in df[most_accurate_model]:
                html = html.replace(
                    f'<td>{value}</td>',
                    f'<td style="border: 3px solid green;">{value}</td>'
                )
            return ui.HTML(html)


        # Accuracy of all models for the selected data
        @render.ui
        def models_accuracy():
            if selected_data() is None:
                return None
            def _display(mape):
                return "unavailable" if pd.isna(mape) else 100 - mape
            acc = {"Accuracy": "%",
                   "SARIMA": _display(_mape(sarima_accuracy())),
                   "Prophet": _display(_mape(prophet_accuracy())),
                   "LSTM": _display(_lstm_mape())}
            return ui.HTML(style_table(pd.DataFrame(acc, index=[0])))


        # Plot of all models' predictions
        @render_widget
        def predictions_plot():
            df = predictions_table()
            if df is None or df.empty:
                return go.Figure()
            fig = make_subplots()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["SARIMA"], mode='lines', name="SARIMA predictions"))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Prophet"], mode='lines', name="Prophet predictions"))
            if not df["LSTM"].isna().all():
                fig.add_trace(go.Scatter(x=df["Date"], y=df["LSTM"], mode='lines', name="LSTM predictions"))
            fig.update_layout(title_text=f"Models forecast plot of the next {len(df)} months",
                              yaxis_title="Forecast", xaxis_title="Date", template='plotly_white')
            return fig


        def _kpi_id(kpi):
            return kpi.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("/", "_")


        # One model dropdown per selected KPI
        @render.ui
        def kpi_model():
            selected_kpis = input.selected_kpis()

            if not selected_kpis:
                return ui.div("No KPIs selected")

            ui_elements = []
            for kpi in selected_kpis:
                ui_elements.append(ui.h6(f"Select Model for {kpi}"))
                ui_elements.append(
                    ui.input_select(
                        id=f"model_{_kpi_id(kpi)}",
                        label=None,
                        choices=["SARIMA", "Prophet", "LSTM"],
                        selected=None
                    )
                )
            return ui.div(*ui_elements)


        # Train and predict for the selected KPIs
        @reactive.calc
        def generate_predictions():
            df = data()
            if df is None:
                return pd.DataFrame()

            selected_kpis = input.selected_kpis()
            predictions = pd.DataFrame()
            horizon = input.predictionHorizon()
            if horizon is None or horizon == "":
                horizon = 0

            for kpi in selected_kpis:
                selected_model = input[f"model_{_kpi_id(kpi)}"]()
                timeseries = df[kpi].dropna()
                train_size = int(len(timeseries) * 0.80)
                time_indices = timeseries.index
                future_time_indices = lstm_model.future_timeindices(time_indices, horizon+1)
                predictions["Date"] = future_time_indices[-horizon:]

                if selected_model == "SARIMA":
                    train, test = timeseries[:train_size], timeseries[train_size:]
                    (trend, seasonal, residual) = sarima_model.series_decompose(train)
                    p, d, q, P, D, Q = _sarima_orders(kpi, train, residual)

                    sarima_fitted_model = sarima_model.sarima_training(train, p, d, q, P, D, Q)
                    sarima_previsions = sarima_fitted_model.predict(start=0, end=len(timeseries)+horizon)
                    predictions[kpi] = sarima_previsions[-horizon:].values

                if selected_model == "Prophet":
                    prophet_fitted_model, scaler = prophet_model.prophet_training(timeseries)
                    prophet_previsions = prophet_model.prophet_forecasting(prophet_fitted_model, horizon, scaler)
                    predictions[kpi] = prophet_previsions['yhat'][-horizon:].values

                if selected_model == "LSTM":
                    values = timeseries.values.reshape(-1, 1).astype('float32')

                    # Defaults, overridden by saved ones for this KPI
                    s = _lstm_settings(kpi, use_saved=True)
                    look_back = s["look_back"]

                    # Split raw values first, fit the scaler on train only
                    raw_train = values[0:train_size, :]
                    raw_test = values[train_size-look_back-1:len(values), :]
                    scaler, train = lstm_model.scaling(raw_train)
                    test = lstm_model.scale_transform(scaler, raw_test)

                    trainX, trainY, testX, testY = lstm_model.reshaping(train, test, look_back)

                    model = lstm_model.lstm_model(look_back, s["nb_layers"], s["nb_units"], s["optimizer"])
                    lstm_model.lstm_training(model, trainX, trainY, s["epoch"], s["batchsize"], s["Verbose"], s["Validationsplit"])

                    last_sequence = testX[-1]
                    future_predictions = lstm_model.lstm_future_predictions(model, last_sequence, look_back, horizon + 1, scaler)
                    predictions[kpi] = future_predictions[:, 0][-horizon:]

            return predictions


        # Download predictions as an Excel file
        # (render.download: the function name is the output id used by
        # ui.download_button("download_predictions", ...))
        @render.download(filename="forecast.xlsx")
        def download_predictions():
            predictions_df = generate_predictions()
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                predictions_df.to_excel(writer, index=False)
            yield buffer.getvalue()


        # Predictions table for all selected KPIs
        @render.ui
        @reactive.event(input.forecast)
        def previsions_table():
            return ui.HTML(style_table(generate_predictions()))


        # ---- Tab contents -------------------------------------------------

        def ts_plot_content():
            return ui.div(
                ui.HTML("<h4 style='text-align:center;'>Time Series Plot</h4>"),
                ui.br(),
                output_widget("timeSeriesPlot"),
                ui.br(),
                ui.br(),
                output_widget("StationarityPlot"),
            )


        def data_content():
            return ui.div(
                ui.HTML("<h4 style='text-align:center;'>Data</h4>"),
                ui.br(),

                ui.row(
                    ui.column(4, ui.output_ui("dataTable")),
                    ui.column(8, ui.output_ui("dataSummary")),
                ),

                ui.row(ui.column(12, ui.hr(), ui.br())),

                ui.row(ui.HTML("<h6 style='text-align:center;'>Stationarity test</h6>")),

                ui.row(
                    ui.column(4, ui.output_text_verbatim("DFGLSTest")),
                    ui.column(4, ui.output_text_verbatim("PPTest")),
                    ui.column(4, ui.output_text_verbatim("KPSSTest")),
                ),

                ui.row(ui.column(12, ui.br(), ui.br())),

                ui.row(
                   ui.column(3, ui.h6("Data volume :")),
                   ui.column(3, ui.output_text("dataVolume")),
                   ui.column(3, ui.h6("Trend direction :")),
                   ui.column(3, ui.output_text("trend")),
                ),

                ui.row(
                   ui.column(3, ui.h6("Percentage of missing values :")),
                   ui.column(3, ui.output_text("missingValues")),
                   ui.column(3, ui.h6("Is the series seasonal ?")),
                   ui.column(3, ui.output_text("seasonality")),
                ),

                ui.row(
                   ui.column(3, ui.h6("Is the time series stationary ?")),
                   ui.column(3, ui.output_text("stationarity")),
                   ui.column(3, ui.h6("Seasonal period :")),
                   ui.column(3, ui.output_text("period")),
                ),

                ui.row(ui.column(12, ui.br(), ui.hr(), ui.br())),

                ui.row(
                    ui.column(4, ui.output_ui("data_volume"), ui.br(), ui.output_ui("missing_status")),
                    ui.column(4, ui.output_ui("trend_strength"), ui.br(), ui.output_ui("seasonal_strength")),
                    ui.column(4, ui.output_ui("stationary")),
                ),

                ui.br(),
                ui.br(),

                ui.row(ui.output_ui("ModelSelection"))
            )


        def model_content():
            return ui.div(
                ui.navset_tab(
                   # Plots of the model
                   ui.nav_panel("Model Plot",
                          ui.HTML("<h4 style='text-align:center;'>Model Plot</h4>"),
                          ui.br(),
                          ui.row(
                              ui.column(4, ui.input_select("model_choice", "Choose a model:", choices={"sarima": "SARIMA", "prophet": "Prophet", "lstm": "LSTM"})),
                              ui.column(4, ui.input_numeric("predictionHorizon", "Prediction time (in months)", value=12)),
                          ),

                          ui.row(
                              ui.column(4, ui.input_checkbox("model_params", "Choose model parameters", False)),
                              ui.column(4, ui.input_checkbox("saved_params", "Use saved parameters", False)),
                              ui.column(2,),
                              ui.column(2, None if READ_ONLY else ui.panel_conditional(
                                  "input.model_params",
                                  ui.input_action_button("save_params", ui.HTML('<i class="fa fa-save"></i> Save'), class_="btn"),
                              ))
                          ),
                          ui.output_ui("saved_params_message"),
                          ui.output_ui("modelParams"),

                          ui.br(),

                          ui.panel_conditional(
                              "input.model_choice == 'sarima'",
                              output_widget("sarimaPlot"),
                              ui.br(),
                              ui.row(
                                  ui.column(6, ui.output_plot("sarimaPACF")),
                                  ui.column(6, ui.output_plot("sarimaACF"))
                              ),
                              ui.br(),
                              ui.row(
                                  ui.column(6, ui.output_plot("sarima_seasonalPACF")),
                                  ui.column(6, ui.output_plot("sarima_seasonalACF"))
                              ),
                          ),

                          ui.panel_conditional(
                              "input.model_choice == 'prophet'",
                              output_widget("prophetPlot"),
                          ),

                          ui.panel_conditional(
                              "input.model_choice == 'lstm'",
                              output_widget("lstmPlot"),
                          ),
                   ),

                   # Performance of the model
                   ui.nav_panel("Performance",
                          ui.HTML("<h4 style='text-align:center;'>Model Performance</h4>"),
                          ui.br(),
                          ui.row(
                              ui.column(6, ui.output_ui("testResults")),
                              ui.column(6, ui.panel_conditional(
                                           "input.model_choice == 'sarima'",
                                           ui.h6("Breusch-Pagan test results (homoscedasticity):"),
                                           ui.output_text_verbatim("BPTest"),
                                           ui.br(),
                                           ui.h6("Ljung-Box test results (autocorrelation):"),
                                           ui.output_text_verbatim("LBTest")
                                           )
                              ),
                          ),
                          ui.br(),
                          ui.row(ui.column(6, ui.output_ui("accuracyMetrics"))),
                          ui.br(),
                          ui.row(
                              ui.h6("Model validity"),
                              ui.row(
                                  ui.column(6, ui.output_ui("modelValidity")),
                                  ui.column(6, ui.output_ui("validityMessage"))
                              )
                          ),
                   ),
                ),
            )


        def previsions_content():
            # The KPIs of the loaded file (falls back to the kpi.ods list)
            df = data()
            if df is not None:
                kpi_choices = df.columns.tolist()
            else:
                kpi_choices = params_data["Description"]["KPI"].dropna().tolist()
            return ui.div(
                ui.navset_tab(
                    ui.nav_panel("Modeling",
                           ui.HTML("<h4 style='text-align:center;'>Modeling</h4>"),
                           ui.br(),
                           ui.row(
                               ui.column(4, ui.output_ui("models_accuracy"),
                                            ui.br(),
                                            ui.output_ui("predictions")),
                               ui.column(8, output_widget("predictions_plot"),
                                            ui.br(),
                                            ui.br(),
                                            ui.row(
                                                ui.column(6,
                                                          ui.input_select(
                                                              id="selected_kpis",
                                                              label="Choose KPIs",
                                                              choices=kpi_choices,
                                                              selected=None,
                                                              multiple=True
                                                          )
                                                ),
                                                ui.column(6, ui.output_ui("kpi_model")),
                                            ),
                                            ui.br(),
                                            ui.row(
                                                ui.column(4,),
                                                ui.column(4, ui.download_button("download_predictions", ui.HTML('<i class="fa fa-download"></i> Download Predictions'), class_="btn")),
                                                ui.column(4,)
                                            ),
                               )
                           ),
                    ),

                    ui.nav_panel("Forecast",
                           ui.HTML("<h4 style='text-align:center;'>Forecast Data</h4>"),
                           ui.br(),
                           ui.output_ui("previsions_table"),
                           ui.row(
                               ui.column(5,),
                               ui.column(2, ui.input_action_button("forecast", ui.HTML('<i class="fa fa-calendar"></i> Show Forecast'), class_="btn")),
                               ui.column(5,)
                           )
                    )
                )
            )


        def documentation_content():
            if READ_ONLY:
                # Public demo: show the documentation only, no editor
                return ui.div(ui.HTML(markdown2.markdown(read_markdown_file(DOC_FILE))))
            return ui.div(
                ui.row(
                    ui.column(4, ui.input_text_area("markdown_editor", ui.h6("Edit Documentation"), read_markdown_file(DOC_FILE), rows=50),
                                 ui.input_action_button("save_btn", ui.HTML('<i class="fa fa-save"></i> Save Changes'), class_="btn")),
                    ui.column(8, ui.output_ui("markdown_preview"))
                )
            )


        # Show the content of the selected sidebar tab
        @render.ui
        def main_content():
            selected_tab = input.sidebar_menu()
            if selected_tab == "ts_plot_tab":
                return ts_plot_content()
            elif selected_tab == "data_tab":
                return data_content()
            elif selected_tab == "model_tab":
                return model_content()
            elif selected_tab == "previsions_tab":
                return previsions_content()
            elif selected_tab == "documentation_tab":
                return documentation_content()
            return ui.div("Please select an option from the sidebar menu.")


    return server
