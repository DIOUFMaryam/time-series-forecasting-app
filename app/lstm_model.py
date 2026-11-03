#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:38 2024

@author: diouf.maryam5100@gmail.com

LSTM forecaster for monthly KPIs.

Rework (Nov 2026). The previous version scored 90%+ "accuracy" but could
not forecast: it was tested one month ahead from actual values (an easy
test that copying last month's value also passes), and its future
forecast, made recursively from its own outputs, came out flat. Changes:

1. The network learns month-to-month DIFFERENCES, not levels. A level
   model on a trending series must extrapolate outside the range it was
   scaled on, which is where an LSTM fails. Forecast differences are added
   back to the last known value. The scaler (StandardScaler, robust to the
   outage spikes MinMax would squash everything around) is fit on the
   training differences only.
2. It is tested like SARIMA: the whole test period (last 20%) is forecast
   recursively from the end of the training data, aligned on the test
   dates, and scored against the same "same month last year" baseline.
   Its accuracy is therefore comparable with SARIMA and Prophet.
3. A smaller network for ~100 data points: one plain LSTM layer with 8
   units by default (bidirectional layers read the sequence backwards too,
   which is meaningless when forecasting the next value), MAE loss, and
   early stopping on the validation split, keeping the best weights.
4. Future values come from a model refitted on the WHOLE series for the
   number of epochs early stopping chose, as SARIMA does, so the most
   recent 20% of the data is not ignored.
5. Seeded before each fit: the same data and settings give the same result.

The network code is confined to build_network / train_network /
predict_next; everything else is plain numpy / pandas.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


SEED = 7

# Default settings, used when the user chooses nothing and no parameters
# are saved for the KPI
DEFAULTS = dict(
    look_back=12,          # months of differences the network sees
    nb_layers=1,
    nb_units=8,
    optimizer="adam",
    epochs=200,            # maximum: early stopping usually ends sooner
    batch_size=16,
    verbose=0,
    validation_split=0.2,  # last 20% of the training windows
)
PATIENCE = 20              # epochs without improvement before stopping
MIN_WINDOWS = 12           # fewer training windows than this: too short


# ---------------------------------------------------------------------------
# TensorFlow is loaded lazily (only when an LSTM model is actually built).
# It is heavy (slow startup) and, on some machines, Windows policies such as
# Smart App Control block its DLLs. Importing it at module level used to make
# the WHOLE app crash at startup; now only the LSTM features are affected.
# ---------------------------------------------------------------------------
class TensorFlowUnavailableError(RuntimeError):
    """Raised when TensorFlow cannot be imported on this machine."""


_tf = None
_tf_error = None


def _load_tensorflow():
    global _tf, _tf_error
    if _tf is not None:
        return _tf
    if _tf_error is not None:
        raise _tf_error
    try:
        import tensorflow as tf
    except (ImportError, OSError) as e:
        _tf_error = TensorFlowUnavailableError(
            "LSTM is unavailable on this machine: TensorFlow could not be "
            f"loaded ({e}). SARIMA and Prophet are not affected."
        )
        raise _tf_error from e
    _tf = tf
    return _tf


def tensorflow_available():
    try:
        _load_tensorflow()
        return True
    except TensorFlowUnavailableError:
        return False


# ---------------------------------------------------------------------------
# Network (the only TensorFlow code)
# ---------------------------------------------------------------------------
def build_network(look_back, nb_layers=1, nb_units=8, optimizer="adam"):
    """Stack of `nb_layers` LSTM layers and one output: the next difference."""
    tf = _load_tensorflow()
    tf.keras.utils.set_random_seed(SEED)  # numpy, python and TF seeds
    layers = tf.keras.layers
    nb_layers = max(1, int(nb_layers))
    model = tf.keras.models.Sequential([layers.Input(shape=(look_back, 1))])
    for i in range(nb_layers):
        # every layer but the last passes its whole sequence to the next one
        model.add(layers.LSTM(int(nb_units), return_sequences=(i < nb_layers - 1)))
    model.add(layers.Dense(1))
    model.compile(loss="mean_absolute_error", optimizer=optimizer)
    return model


def train_network(model, X, y, epochs, batch_size=16, verbose=0, validation_split=0.2):
    """Fit the network. With a validation split, stop when the validation
    loss hasn't improved for PATIENCE epochs and keep the best weights.
    Returns the number of epochs that gave the kept weights."""
    tf = _load_tensorflow()
    epochs = int(epochs)
    validation_split = float(validation_split or 0)
    callbacks = []
    # Keras takes the validation windows from the END of the training data
    # (the most recent months), which is what we want for a time series.
    # Too few validation windows make early stopping meaningless.
    if int(len(X) * validation_split) < 3:
        validation_split = 0.0
    if validation_split > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True))
    history = model.fit(X, y, epochs=epochs, batch_size=int(batch_size),
                        verbose=int(verbose), validation_split=validation_split,
                        callbacks=callbacks)
    if validation_split > 0:
        return int(np.argmin(history.history["val_loss"])) + 1
    return epochs


def predict_next(model, windows):
    """Network output for a batch of windows, shape (n, look_back, 1) -> (n,)."""
    # Calling the model directly is much faster than model.predict()
    # for the small batches of a recursive forecast
    return np.asarray(model(windows, training=False)).reshape(-1)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def make_windows(values, look_back):
    """X = `look_back` consecutive values, y = the value that follows.
    Returns X with shape (samples, look_back, 1) as Keras expects."""
    values = np.asarray(values, dtype="float32").reshape(-1)
    n = len(values) - look_back
    if n < 1:
        return np.empty((0, look_back, 1), dtype="float32"), np.empty(0, dtype="float32")
    X = np.stack([values[i:i + look_back] for i in range(n)])
    return X.reshape(n, look_back, 1), values[look_back:]


def _check_length(n_values, look_back):
    n_windows = n_values - 1 - look_back  # one value lost to differencing
    if n_windows < MIN_WINDOWS:
        raise ValueError(
            f"Not enough data for an LSTM with a sequence length of {look_back}: "
            f"{n_values} training months give {max(n_windows, 0)} training "
            f"sequences (at least {MIN_WINDOWS} needed). Use a shorter sequence length.")


@dataclass
class _Fitted:
    model: object
    scaler: StandardScaler
    scaled_diffs: np.ndarray   # scaled differences the model was trained on
    epochs_used: int


def _fit(levels, s, validation_split):
    """Train a network on the month-to-month differences of `levels`."""
    diffs = np.diff(np.asarray(levels, dtype="float64")).reshape(-1, 1)
    scaler = StandardScaler().fit(diffs)  # fit on these (training) values only
    scaled = scaler.transform(diffs).reshape(-1).astype("float32")
    X, y = make_windows(scaled, s["look_back"])
    model = build_network(s["look_back"], s["nb_layers"], s["nb_units"], s["optimizer"])
    epochs_used = train_network(model, X, y, s["epochs"], s["batch_size"],
                                s["verbose"], validation_split)
    return _Fitted(model, scaler, scaled, epochs_used)


def _forecast_levels(fitted, last_level, steps, look_back):
    """Forecast `steps` months recursively: each predicted difference is fed
    back as input for the next one, then the differences are added up from
    the last known value."""
    if steps < 1:
        return np.empty(0)
    window = list(fitted.scaled_diffs[-look_back:])
    scaled_preds = []
    for _ in range(steps):
        x = np.asarray(window[-look_back:], dtype="float32").reshape(1, look_back, 1)
        nxt = float(predict_next(fitted.model, x)[0])
        scaled_preds.append(nxt)
        window.append(nxt)
    diffs = fitted.scaler.inverse_transform(np.asarray(scaled_preds).reshape(-1, 1)).reshape(-1)
    return last_level + np.cumsum(diffs)


def _one_step_fit(fitted, levels, look_back):
    """In-sample, one month ahead from actual values: how well the network
    fits the training months (shown on the plot, not used for scoring)."""
    X, _ = make_windows(fitted.scaled_diffs, look_back)
    if len(X) == 0:
        return np.empty(0)
    diffs = fitted.scaler.inverse_transform(predict_next(fitted.model, X).reshape(-1, 1)).reshape(-1)
    levels = np.asarray(levels, dtype="float64")
    # window i ends with the difference levels[i+L] - levels[i+L-1] and
    # predicts levels[i+L+1] - levels[i+L]
    return levels[look_back:look_back + len(diffs)] + diffs


def next_months(last_date, horizon):
    # Monthly dates following `last_date`
    return pd.DatetimeIndex([last_date + pd.DateOffset(months=k) for k in range(1, horizon + 1)])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass
class LSTMResult:
    name: str
    series: pd.Series          # the data used (missing values removed)
    train: pd.Series
    test: pd.Series
    test_forecast: pd.Series   # multi-step, from the end of the training data
    train_fit: pd.Series       # one step ahead, training months only
    future: pd.Series          # from the model refitted on the whole series
    epochs_used: int
    settings: dict = field(default_factory=dict)


def settings_with_defaults(**overrides):
    """DEFAULTS updated with the non-empty values of `overrides`."""
    s = dict(DEFAULTS)
    s.update({k: v for k, v in overrides.items() if v is not None and k in DEFAULTS})
    s["look_back"] = int(s["look_back"])
    return s


def evaluate_and_forecast(timeseries, horizon, train_share=0.80, **settings):
    """Test the LSTM on the last (1 - train_share) of the series, then
    forecast the `horizon` months after it.

    - Test: trained on the first `train_share` of the data (scaler included),
      the whole test period forecast recursively from the end of training.
      Same split as SARIMA and seasonal_naive_accuracy, so the scores compare.
    - Future: retrained on the whole series for the number of epochs early
      stopping chose, then forecast recursively from the last month.
    """
    s = settings_with_defaults(**settings)
    L = s["look_back"]
    ts = timeseries.dropna().astype(float)
    n = int(len(ts) * train_share)
    train, test = ts[:n], ts[n:]
    _check_length(len(train), L)

    # 1. Evaluation model: training data only
    fitted = _fit(train.values, s, s["validation_split"])
    test_forecast = pd.Series(
        _forecast_levels(fitted, train.values[-1], len(test), L), index=test.index)
    fit_values = _one_step_fit(fitted, train.values, L)
    train_fit = pd.Series(fit_values, index=train.index[L + 1:L + 1 + len(fit_values)])

    # 2. Forecasting model: whole series, fixed number of epochs
    horizon = int(horizon or 0)
    if horizon > 0:
        full = _fit(ts.values, dict(s, epochs=fitted.epochs_used), validation_split=0.0)
        future = pd.Series(_forecast_levels(full, ts.values[-1], horizon, L),
                           index=next_months(ts.index[-1], horizon))
    else:
        future = pd.Series(dtype=float)

    return LSTMResult(name=str(ts.name), series=ts, train=train, test=test,
                      test_forecast=test_forecast, train_fit=train_fit, future=future,
                      epochs_used=fitted.epochs_used, settings=s)


def result_figure(result):
    """Plot: data, training fit, test forecast and future forecast."""
    r = result
    s = r.settings
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.series.index, y=r.series.values, mode="lines", name="Original data"))
    fig.add_trace(go.Scatter(x=r.train_fit.index, y=r.train_fit.values, mode="lines",
                             name="Train predictions (1 month ahead)"))
    fig.add_trace(go.Scatter(x=r.test_forecast.index, y=r.test_forecast.values, mode="lines",
                             name="Test predictions"))
    if not r.future.empty:
        fig.add_trace(go.Scatter(x=r.future.index, y=r.future.values, mode="lines",
                                 name="Future predictions"))
    fig.update_layout(
        title_text=(f"LSTM ({s['nb_layers']} layer(s) x {s['nb_units']} units, "
                    f"{s['look_back']} months, {r.epochs_used} epochs) plot of {r.name}"),
        xaxis_title="Date", yaxis_title=r.name, template="plotly_white")
    return fig
