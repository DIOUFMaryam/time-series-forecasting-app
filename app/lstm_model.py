#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:23:38 2024

@author: diouf.maryam5100@gmail.com
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # web app: render figures to images, never open Tk windows
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dateutil.relativedelta import relativedelta
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import plotly.express as px


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
    tf.random.set_seed(7)
    _tf = tf
    return _tf


def tensorflow_available():
    try:
        _load_tensorflow()
        return True
    except TensorFlowUnavailableError:
        return False




# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Scaling the data
# IMPORTANT: fit the scaler on the training split ONLY, never on the full
# series - fitting on everything leaks test-set min/max into training.
def scaling(train):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train)
    return scaler, scaled_train


# Transform any other split (test set, full series for plotting, etc.)
# with a scaler that was already fit on the training data.
def scale_transform(scaler, data):
    return scaler.transform(data)


# reshape into X=t and Y=t+1
def reshaping(train, test, look_back):
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    return trainX, trainY, testX, testY


# create and fit the LSTM network
def lstm_model(look_back, nb_layers, nb_units, optimizer):
    tf = _load_tensorflow()
    Sequential = tf.keras.models.Sequential
    Dense = tf.keras.layers.Dense
    LSTM = tf.keras.layers.LSTM
    Bidirectional = tf.keras.layers.Bidirectional

    model = Sequential()

    # Adding the first layer with input shape specified
    model.add(Bidirectional(LSTM(nb_units, return_sequences=(nb_layers > 1)), input_shape=(look_back, 1)))

    # Adding hidden layers
    for _ in range(1, nb_layers - 1):
        model.add(Bidirectional(LSTM(nb_units, return_sequences=True)))

    # Adding the last LSTM layer with return_sequences=False
    if nb_layers > 1:
        model.add(Bidirectional(LSTM(nb_units, return_sequences=False)))

    # Adding the output layer
    model.add(Dense(1))

    # Compiling the model
    model.compile(loss='mean_absolute_error', optimizer=optimizer)

    return model


def lstm_training(model, trainX, trainY, epoch, batchsize, Verbose, Validationsplit):
    model.fit(trainX, trainY, epochs=epoch, batch_size=batchsize, verbose=Verbose, validation_split = Validationsplit)


def _safe_mape(actual, predicted):
    # Guard against division by zero: points where actual == 0 give an
    # undefined percentage error, so exclude them rather than letting
    # them produce inf/nan silently.
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    if nonzero.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100


def lstm_eval(model, trainY, trainPredict, testY, testPredict):
    # calculate root mean squared error
    trainScore = [mean_squared_error(trainY[0], trainPredict[:,0]),
                  np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0])),
                  mean_absolute_error(trainY[0], trainPredict[:,0]),
                  _safe_mape(trainY[0], trainPredict[:,0])]
    testScore = [mean_squared_error(testY[0], testPredict[:,0]),
                 np.sqrt(mean_squared_error(testY[0], testPredict[:,0])),
                 mean_absolute_error(testY[0], testPredict[:,0]),
                 _safe_mape(testY[0], testPredict[:,0])]
    return (trainScore, testScore)


def lstm_prediction(model, trainX, testX, trainY, testY, scaler):
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    return trainPredict, testPredict, trainY, testY


def visualize_predictions(dataset, look_back, time_indices, trainPredict, testPredict, scaler):
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back-1:len(trainPredict)+look_back-1, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back)-2:len(dataset)-1, :] = testPredict

    # plot baseline and predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_indices, y=scaler.inverse_transform(dataset).flatten(), mode='lines', name='Original data'))
    fig.add_trace(go.Scatter(x=time_indices, y=trainPredictPlot.flatten(), mode='lines', name='Train predictions'))
    fig.add_trace(go.Scatter(x=time_indices, y=testPredictPlot.flatten(), mode='lines', name='Test predictions'))

    #fig,ax = plt.subplots()
    #plt.plot(time_indices, scaler.inverse_transform(dataset), label='True')
    #plt.plot(time_indices, trainPredictPlot, label='Train predictions')
    #plt.plot(time_indices, testPredictPlot, label='Test predictions')

    return (trainPredictPlot, testPredictPlot, fig)


def lstm_future_predictions(model, last_sequence, look_back, horizon, scaler):
    # Initialize the list to store the predicted values
    future_predictions = []

    # Predict future values
    for _ in range(horizon):
        # Reshape the sequence to match the model input
        last_sequence_reshaped = last_sequence.reshape(1, look_back, 1)
        next_value = model.predict(last_sequence_reshaped)
        future_predictions.append(np.round(next_value, 2))

        # Update the sequence: remove the first value and append the predicted value
        last_sequence = np.append(last_sequence[1:], next_value, axis=0)

    future_predictions = np.array(future_predictions).reshape(-1,1)
    # Inverse transform the predictions to original scale
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions


def future_timeindices(time_indices, horizon):
    future_time_indices = []
    current = time_indices[-1]

    for _ in range(horizon):
        future_time_indices.append(current)
        current += relativedelta(months=1)
    return future_time_indices
