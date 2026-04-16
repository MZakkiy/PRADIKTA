# app/analysis/data_processor.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import os

METHOD_MAP = {'Forward' : 'ffill',
              'Bacward' : 'bfill',
              'Linear' : 'linear',
              'Spline' : 'spline',
              'Nearest' : 'nearest',
              'Akima' : 'akima',
              'Pchip' : 'pchip'}

def import_data(file_path):
    if not file_path:
        return None, "Tidak ada file yang dipilih."

    # Memeriksa apakah file ada
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File tidak ditemukan di '{file_path}'")
    
    # Mendapatkan ekstensi file 
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
        
    try:
        if extension == '.csv':
            df = pd.read_csv(file_path)
            # df = set_datetime_index(df)
            return df, None
        
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            # df.drop("Tanggal", axis=1, inplace=True)
            # df = set_datetime_index(df)
            return df, None
            
        elif extension == '.json':
            df = pd.read_json(file_path)
            # df = set_datetime_index(df)
            return df, None
            
        else:
            # Memberikan pesan error jika format tidak didukung
            raise ValueError(f"Format file '{extension}' tidak didukung. Gunakan .csv, .xlsx, atau .json.")
            
    except ValueError as e:
        print(f"Terjadi kesalahan saat membaca file: {e}")
        return None, e
    
def count_na(dataframe, column_name):
    return dataframe[column_name].isna().sum()

def set_datetime_index(dataframe):
    for column in dataframe.columns:
        try:
            dataframe[column] = pd.to_datetime(dataframe[column])
            # d1 = pd.to_datetime(data, format='%d-%m-%Y', errors='coerce')

            # # For the remaining NaT values, try another format
            # d2 = pd.to_datetime(data, format='%B %d, %Y', errors='coerce')

            # # And another one...
            # d3 = pd.to_datetime(data, format='%Y/%m/%d', errors='coerce')

            dataframe.set_index(column, inplace=True)
            return dataframe
        except Exception:
            continue 

def data_imputation(dataframe, method):
    if method == 'Forward':
        return dataframe.ffill()
    elif method == 'Backward':
        return dataframe.bfill()
    else:
        return dataframe.interpolate(method=METHOD_MAP[method])
    
def remove_random_data(dataframe, sample_ratio):
    np.random.seed(74)
    n_data = len(dataframe)
    random_valid_indices = dataframe[dataframe.notna()].sample(n=int(sample_ratio * n_data)).index
    dataframe[random_valid_indices] = np.nan
    return dataframe, random_valid_indices

def MAE(actual, predicted):
    return mean_absolute_error(actual, predicted)

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def data_separation(dataframe, train_ratio, valid_ratio):
    n_data = len(dataframe)
    train_split_index = int(n_data * train_ratio)
    validation_split_index = int(n_data * (train_ratio + valid_ratio))
    
    train_data = dataframe[:train_split_index]
    validation_data = dataframe[train_split_index:validation_split_index]
    test_data = dataframe[validation_split_index:]
    return train_data, validation_data, test_data

def feature_scaling(train_data, validation_data, test_data):
    if isinstance(train_data, pd.Series):
        train_data = train_data.to_numpy()
        validation_data = validation_data.to_numpy()
        test_data = test_data.to_numpy()

    if train_data.ndim == 1:
        # Reshape 1D arrays to 2D for the scaler
        train_data = train_data.reshape(-1, 1)
        validation_data = validation_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    train_data_scaled = scaler.transform(train_data)
    validation_data_scaled = scaler.transform(validation_data)
    test_data_scaled = scaler.transform(test_data)

    return train_data_scaled, validation_data_scaled, test_data_scaled, scaler


def MAPE(actual, predicted):
    """Mean Absolute Percentage Error"""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def RMSE(actual, predicted):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actual, predicted))

def RSE(actual, predicted):
    """Relative Squared Error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)