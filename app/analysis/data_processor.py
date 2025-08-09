# app/analysis/data_processor.py

import pandas as pd
import os

def import_data(file_path):
    """
    Mencoba memuat data dari file CSV yang diberikan.
    Mengembalikan DataFrame jika berhasil, atau None jika gagal.
    """
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
            return pd.read_csv(file_path), None
        
        elif extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path), None
            
        elif extension == '.json':
            return pd.read_json(file_path), None
            
        else:
            # Memberikan pesan error jika format tidak didukung
            raise ValueError(f"Format file '{extension}' tidak didukung. Gunakan .csv, .xlsx, atau .json.")
            
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file: {e}")
        return None, e


# --- Anda akan menambahkan fungsi lain di sini nanti ---

def data_imputation(dataframe, metode):
    # ... kode untuk imputasi ...
    pass

def data_separation(dataframe, rasio_train):
    # ... kode untuk memisahkan data ...
    pass