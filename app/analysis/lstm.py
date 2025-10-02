# app/analysis/lstm.py
import pandas as pd
import numpy as np
from typing import Tuple

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout

import warnings

warnings.filterwarnings('ignore')

from typing import List, Tuple

def BiLSTM(units, return_sequences, input_shape=None):
    if input_shape:
        return Bidirectional(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
    return Bidirectional(LSTM(units=units, return_sequences=return_sequences))

MODEL_MAP = {
    "LSTM" : LSTM,
    "GRU" : GRU,
    "Bi-LSTM" : BiLSTM
}

def create_sliding_window(train_data: pd.Series, validation_data: pd.Series, test_data: pd.Series, window_size: int):
    # Mengubah Series menjadi numpy array dan menggabungkannya
    if not isinstance(train_data, pd.Series):
        train_data = pd.Series(train_data.reshape(-1))
        validation_data = pd.Series(validation_data.reshape(-1))
        test_data = pd.Series(test_data.reshape(-1))

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

def build_lstm_model(model_type: str,input_shape: Tuple[int, int], lstm_units: List[int], dropout_rate: float = 0.2):  
    model = Sequential()
    
    # --- Tambahkan lapisan-lapisan LSTM secara dinamis ---
    num_lstm_layers = len(lstm_units)
    for i, units in enumerate(lstm_units):
        is_last_lstm_layer = (i == num_lstm_layers - 1)
        
        # return_sequences=True untuk semua lapisan kecuali lapisan LSTM terakhir
        return_sequences = not is_last_lstm_layer
        
        if i == 0:
            # Lapisan pertama memerlukan input_shape
            model.add(MODEL_MAP[model_type](units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(MODEL_MAP[model_type](units=units, return_sequences=return_sequences))
            
        # Tambahkan Dropout setelah setiap lapisan LSTM
        model.add(Dropout(dropout_rate))
        
    # --- Tambahkan lapisan output ---
    model.add(Dense(units=1))
    
    return model

def forecast_lstm(model, initial_sequence, n_steps_to_predict):
    if initial_sequence.ndim != 2:
        raise ValueError(f"initial_sequence must be 2D (timesteps, features), but got shape {initial_sequence.shape}")

    predictions = []
    
    # Reshape the initial sequence to match the model's expected input shape: (1, timesteps, features)
    n_timesteps, n_features = initial_sequence.shape
    current_batch = initial_sequence.reshape(1, n_timesteps, n_features)

    for _ in range(n_steps_to_predict):
        # 1. Get the prediction for the next step (it will be scaled)
        # model.predict returns shape (1, 1) or (1, n_output_features), so we get the first element.
        predicted_value = model.predict(current_batch, verbose=0)[0]
        
        # 2. Store the scaled prediction
        predictions.append(predicted_value)
        
        # 3. Update the batch for the next prediction
        # - Remove the first time step from the current batch.
        # - Append the new predicted value at the end.
        # The new predicted value needs to be reshaped to (1, 1, n_features) to be appended correctly.
        new_batch_entry = predicted_value.reshape(1, 1, n_features)
        current_batch = np.append(current_batch[:, 1:, :], new_batch_entry, axis=1)
        
    return np.array(predictions)