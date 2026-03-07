# PRADIKTA

PRADIKTA is a desktop application with a Graphical User Interface (GUI) designed for time series prediction, focusing specifically on predicting the Peat Fire Vulnerability Index (PFVI) and other drought indices.

This application provides an end-to-end pipeline, starting from data preprocessing and building Machine Learning (Deep Learning) models, to calculating and forecasting peatland fire danger levels.

## ✨ Key Features

### 1. Data Preparation
* **Data Import:** Supports `.csv`, `.xlsx`, `.xls`, and `.json` file formats.
* **Exploratory Data Analysis (Summary):** Displays descriptive statistical summaries and data distribution visualizations using Boxplots.
* **Data Separation:** Splits the dataset into Train, Validation, and Test sets based on user-defined ratios.
* **Data Imputation:** Handles missing values using various interpolation methods such as Forward, Backward, Linear, Akima, and Pchip. It also includes a "Random Check" feature to evaluate the imputation error (MSE/MAE).
* **Feature Scaling:** Automatically normalizes data using `MinMaxScaler`.

### 2. Machine Learning Modeling (Time Series)
* **Model Building (Sliding Window):** Constructs Recurrent Neural Network (RNN) architectures for sequential data.
* **Supported Algorithms:** **LSTM** and **GRU**.
* **Hyperparameter Customization:** Users can adjust the Window Size, Number of Hidden Layers, Neurons per layer, Dropout Rate, Epochs, Batch Size, Optimizer (Adam, RMSprop, SGD), and Loss Function metric.
* **Model Evaluation:** Displays evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE). It also provides interactive visualizations for the Loss Function and Actual vs. Predicted plots.

### 3. Fire Index Calculation & Forecasting
* **Drought Index Calculation:** Capable of computing various indices, including:
  * **KBDI** (Keetch-Byram Drought Index)
  * **KBDI(adj)**
  * **mKBDI**
  * **PFVI** (Peat Fire Vulnerability Index)
* **Parameter Optimization:** Utilizes the Nelder-Mead algorithm (via SciPy) to calibrate the intrinsic parameters of the peatland hydrological model (`aH`, `bH`, `n`, `alpha`).
* **Forecasting:** Predicts future index values for *n* forecast steps using the trained ML model.
* **Fire Danger Rating:** Automatically classifies the fire danger level into *Low*, *Moderate*, *High*, and *Extreme* categories on each interactive data point.

## 🛠️ Technologies Used
* **GUI Framework:** PySide6
* **Data Manipulation & Math:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-Learn, TensorFlow / Keras
* **Data Visualization:** Matplotlib, mplcursors

## 🚀 How to Run the Application

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MZakkiy/PRADIKTA.git
   cd pradikta
   ```
2. **Install Dependencies** \
   Ensure you have Python 3.8+ installed. Install the required libraries by running:
   ```bash
   pip install PySide6 pandas numpy scipy scikit-learn tensorflow matplotlib mplcursors
   ```
3. **Run the Application** \
   Navigate to the project directory and run the application entry point:
   ```bash
   python app/main.py
   ```

## 📂 Directory Structure

```
PRADIKTA/
│
├── app/
│   ├── analysis/
│   │   ├── data_processor.py   # Data imputation, scaling, and splitting logic
│   │   ├── fire_predict.py     # PFVI/KBDI hydrology logic & parameter optimization
│   │   └── model.py            # Neural Network architectures (LSTM, GRU, Bi-LSTM)
│   ├── main.py                 # Application entry point (QApplication initialization)
│   ├── ui_main.py              # Main UI layout and logic (Tabs, Plots)
│   └── widgets.py              # Custom widgets (Matplotlib Canvas, Pandas Table)
│
├── .gitignore
└── README.md
```
