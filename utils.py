# utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()
    df = df.fillna(method='ffill')  # handle missing values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse
