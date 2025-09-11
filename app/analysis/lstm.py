# app/analysis/lstm.py
import pandas as pd
import numpy as np
from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from typing import List, Tuple

OPTIMIZER_MAP = {"Adam" : "adam",
                 "RMSprop" : "rmsprop",
                 "SGD" : 'sgd'}

LOSS_FUNCTION_MAP = {"Mean Squared Error" : "mean_squared_error",
                     "Mean Absolute Error" : "mean_absolute_error"}

def create_sliding_window(train_data: pd.Series, validation_data: pd.Series, test_data: pd.Series, window_size: int):
    # Mengubah Series menjadi numpy array dan menggabungkannya
    combined_data = np.concatenate([
        train_data.values, 
        validation_data.values, 
        test_data.values
    ])

    X, y = [], []
    for i in range(window_size, len(combined_data)):
        X.append(combined_data[i-window_size:i])
        y.append(combined_data[i])
        
    X, y = np.array(X), np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_split_point = len(train_data) - window_size
    validation_split_point = train_split_point + len(validation_data)
    
    X_train, y_train = X[:train_split_point], y[:train_split_point]
    X_val, y_val = X[train_split_point:validation_split_point], y[train_split_point:validation_split_point]
    X_test, y_test = X[validation_split_point:], y[validation_split_point:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: List[int],
    dropout_rate: float = 0.2,
    optimizer: str = 'adam',
    loss_function: str = 'mean_squared_error'
):  
    model = Sequential()
    
    # --- Tambahkan lapisan-lapisan LSTM secara dinamis ---
    num_lstm_layers = len(lstm_units)
    for i, units in enumerate(lstm_units):
        is_last_lstm_layer = (i == num_lstm_layers - 1)
        
        # return_sequences=True untuk semua lapisan kecuali lapisan LSTM terakhir
        return_sequences = not is_last_lstm_layer
        
        if i == 0:
            # Lapisan pertama memerlukan input_shape
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_sequences))
            
        # Tambahkan Dropout setelah setiap lapisan LSTM
        model.add(Dropout(dropout_rate))
        
    # --- Tambahkan lapisan output ---
    model.add(Dense(units=1))
    
    return model

def train_model(epochs, batch_size, optimizer, loss_function):
    pass

def forecast_lstm(forecast_steps):
    pass