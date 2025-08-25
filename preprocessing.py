import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def handle_missing_values(df):
    """
    Handles missing values by forward filling, backward filling, and mean imputation.
    """
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

def scale_features(df, feature_cols):
    """
    Scales selected features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler  # Return scaler for inverse transformation

def split_train_test(series, train_size=0.8):
    """
    Splits a time series into training and testing sets.
    """
    train_size = int(len(series) * train_size)
    train, test = series[:train_size], series[train_size:]
    return train, test

def create_lstm_sequences(series, seq_length=10):
    """
    Prepares sequences for LSTM input.
    """
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df):
    """
    Full preprocessing pipeline: missing values, scaling, and return cleaned data.
    """
    df = handle_missing_values(df)
    feature_cols = ['Temperature (ºC)', 'D.O. (mg/l)', 'pH', 'Turbidity (NTU)', 'Conductivity (µmhos/cm)', 'B.O.D. (mg/l)']
    df, scaler = scale_features(df, feature_cols)
    return df, scaler
