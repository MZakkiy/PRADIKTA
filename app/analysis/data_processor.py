# app/analysis/data_processor.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import os

method_map = {'Forward' : 'ffill',
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
            df = set_datetime_index(df)
            return df, None
        
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            df = set_datetime_index(df)
            return df, None
            
        elif extension == '.json':
            df = pd.read_json(file_path)
            df = set_datetime_index(df)
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
            dataframe.set_index(column, inplace=True)
            return dataframe
        except Exception:
            continue 
    raise ValueError(f"Tidak ada kolom waktu")

def data_imputation(dataframe, column_name, method):
    if method == 'Forward':
        return dataframe[column_name].ffill()
    elif method == 'Backward':
        return dataframe[column_name].bfill()
    else:
        return dataframe[column_name].interpolate(method=method_map[method])
    
def remove_random_data(dataframe, column_name, n_sample):
    np.random.seed(74)
    random_valid_indices = dataframe[column_name][dataframe[column_name].notna()].sample(n=n_sample).index
    dataframe.loc[random_valid_indices, column_name] = np.nan
    return dataframe, random_valid_indices

def MAE(actual, predicted):
    return mean_absolute_error(actual, predicted)

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def data_separation(dataframe, rasio_train):
    # ... kode untuk memisahkan data ...
    pass