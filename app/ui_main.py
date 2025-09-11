# app/ui_main.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox, 
    QProgressBar, QComboBox, QSpinBox, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt

from widgets import SummaryWindow, MplCanvas

import warnings

warnings.filterwarnings('ignore')

import re

from analysis.data_processor import (
    import_data, count_na, data_imputation, remove_random_data, 
    MAE, MSE, data_separation, feature_scaling
)
from analysis.lstm import (
    create_sliding_window, build_lstm_model
)

class UIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Prediksi Time Series")
        self.setGeometry(100, 100, 1200, 800) 

        # Widget utama dan layout utama
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Membuat dan Menambahkan Panel Atas (Pengaturan & Tab) ---
        top_panel_layout = QHBoxLayout()
        
        # Panel Kiri: Update Settings
        # update_settings_panel = self.create_update_settings_panel()
        # top_panel_layout.addWidget(update_settings_panel)

        # Panel Kanan: Tab Kontrol
        main_tabs = self.create_main_tabs()
        top_panel_layout.addWidget(main_tabs, 1)

        # --- Membuat dan Menambahkan Panel Bawah (Plot) ---
        plot_panel = self.create_plot_panel()

        # Menambahkan semua bagian ke layout utama
        main_layout.addLayout(top_panel_layout)
        main_layout.addWidget(plot_panel, 1)
        
        # self.create_menu_bar()

        self.dataframe = None
        self.summary_win = None

    # def create_menu_bar(self):
    #     menu_bar = self.menuBar()
    #     file_menu = menu_bar.addMenu("File")
    #     exit_action = file_menu.addAction("Exit")
    #     exit_action.triggered.connect(self.close)

    def create_update_settings_panel(self):
        group_box = QGroupBox("Update Settings")
        layout = QVBoxLayout()

        auto_update_checkbox = QCheckBox("Auto Update")
        update_all_button = QPushButton("UPDATE ALL")
        
        processing_label = QLabel("Processing (%)")
        progress_bar = QProgressBar()
        
        # Layout untuk progress bar dan angka
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(progress_bar)
        progress_layout.addWidget(QLabel("0"))

        # Indikator status (lingkaran hijau)
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(16, 16)
        self.status_indicator.setStyleSheet("background-color: #4CAF50; border-radius: 8px;")

        layout.addWidget(auto_update_checkbox)
        layout.addWidget(update_all_button)
        layout.addStretch() 
        layout.addWidget(processing_label)
        layout.addLayout(progress_layout)
        layout.addWidget(self.status_indicator, alignment=Qt.AlignCenter)
        
        group_box.setLayout(layout)
        return group_box

    def create_main_tabs(self):
        tab_widget = QTabWidget()
        tab_widget.setObjectName("mainTabWidget")
        
        # Membuat setiap tab
        data_tab = self.create_data_tab()
        arima_tab = self.create_arima_tab()
        lstm_tab = self.create_lstm_tab()
        
        # Menambahkan tab
        tab_widget.addTab(data_tab, "Data")
        tab_widget.addTab(arima_tab, "ARIMA SARIMA")
        tab_widget.addTab(lstm_tab, "LSTM")
        
        # Contoh isi untuk tab placeholder
        arima_tab.setLayout(QVBoxLayout())
        arima_tab.layout()
        
        lstm_tab.setLayout(QVBoxLayout())
        lstm_tab.layout()

        return tab_widget
        
    def create_data_tab(self):
        data_tab_widget = QWidget()
        main_layout = QHBoxLayout(data_tab_widget)
        
        # --- Kolom 1: Input Data ---
        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()

        import_data_button = QPushButton("Import Data")
        import_data_button.setObjectName("ImportButton")
        import_data_button.clicked.connect(self.handle_import_data)
        
        self.summary_button = QPushButton("Summary")
        self.summary_button.setObjectName("SummaryButton")
        self.summary_button.clicked.connect(self.handle_summary)
        self.summary_button.setEnabled(False)

        self.variable_combobox = QComboBox()
        self.variable_combobox.currentIndexChanged.connect(self.on_variable_column_selected)

        input_layout.addWidget(import_data_button, 0, 0)
        input_layout.addWidget(self.summary_button, 0, 1)
        input_layout.addWidget(QLabel("Variable"), 1, 0)
        input_layout.addWidget(self.variable_combobox, 2, 0, 1, 2)
        input_layout.addWidget(QLabel("NaN Data"), 3, 0)

        self.nan_data_line = QLineEdit("0")
        self.nan_data_line.setReadOnly(True)
        self.nan_data_line.setAlignment(Qt.AlignmentFlag.AlignRight)
        input_layout.addWidget(self.nan_data_line, 4, 0, 1, 2)

        input_group.setLayout(input_layout)

        # --- Kolom 2: Data Separation ---
        self.train_percentage = QSpinBox()
        self.valid_percentage = QSpinBox()
        self.test_percentage = QSpinBox()

        self.train_total = QLineEdit("0")
        self.valid_total = QLineEdit("0")
        self.test_total = QLineEdit("0")
        self.train_total.setReadOnly(True)
        self.valid_total.setReadOnly(True)
        self.test_total.setReadOnly(True)
        
        self.show_separation = QPushButton("Show")
        self.show_separation.clicked.connect(self.handle_show_separation)
        self.show_separation.setEnabled(False)

        separation_group = QGroupBox("Data Separation")
        separation_layout = QGridLayout()
        separation_layout.setSpacing(10)

        separation_layout.addWidget(QLabel("Train (%)"), 0, 0)
        separation_layout.addWidget(self.train_percentage, 0, 1)
        separation_layout.addWidget(self.train_total, 0, 2) # (#)
        separation_layout.addWidget(QLabel("Validation (%)"), 1, 0)
        separation_layout.addWidget(self.valid_percentage, 1, 1)
        separation_layout.addWidget(self.valid_total, 1, 2) # (#)
        separation_layout.addWidget(QLabel("Test (%)"), 2, 0)
        separation_layout.addWidget(self.test_percentage, 2, 1)
        separation_layout.addWidget(self.test_total, 2, 2) # (#)
        separation_layout.addWidget(self.show_separation, 3, 0, 1, 3)
        separation_group.setLayout(separation_layout)

        # --- Kolom 3: Data Imputation ---
        self.imputation_method = QComboBox()
        self.imputation_method.setPlaceholderText("Choose Method")
        self.imputation_method.addItems(['Forward', 'Backward', 'Linear', 'Nearest', 'Akima', 'Pchip'])
        self.imputation_method.currentIndexChanged.connect(self.on_method_selected)
        self.imputation_method.setEnabled(False)

        self.random_check = QCheckBox("Random Check (%)")
        self.random_check.stateChanged.connect(self.on_random_check_state_changed)
        self.random_check.setEnabled(False)

        self.sample_percentage_random_check = QSpinBox()
        self.sample_percentage_random_check.valueChanged.connect(self.on_n_sample_random_check_changed)
        self.sample_percentage_random_check.setEnabled(False)

        imputation_group = QGroupBox("Data Imputation")
        imputation_layout = QGridLayout()

        # imputation_layout.addWidget(QLabel("Method"), 0, 0, 1, 2)
        imputation_layout.addWidget(self.imputation_method, 1, 0, 1, 4)
        imputation_layout.addWidget(self.random_check, 2, 0, 1, 2)
        imputation_layout.addWidget(self.sample_percentage_random_check, 2, 2, 1, 2)
        
        # Menambahkan metrik
        imputation_layout.addWidget(QLabel("MSE"), 3, 0)
        self.mse_random_check = QLineEdit("0")
        self.mse_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.mse_random_check, 3, 1)

        imputation_layout.addWidget(QLabel("MAE"), 3, 2)
        self.mae_random_check = QLineEdit("0")
        self.mae_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.mae_random_check, 3, 3)
        
        imputation_layout.addWidget(QLabel("Var Before"), 4, 0)
        self.var_before_random_check = QLineEdit("0")
        self.var_before_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.var_before_random_check, 4, 1)

        imputation_layout.addWidget(QLabel("Var After"), 4, 2)
        self.var_after_random_check = QLineEdit("0")
        self.var_after_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.var_after_random_check, 4, 3)

        imputation_group.setLayout(imputation_layout)
        
        # --- Kolom 4: Feature Scaling ---
        preprocess_group = QGroupBox("Feature Scaling")
        preprocess_layout = QVBoxLayout()
        preprocess_layout.addWidget(QCheckBox("Standardize"))
        preprocess_group.setLayout(preprocess_layout)

        # Menambahkan semua grup ke layout utama tab data
        main_layout.addWidget(input_group)
        main_layout.addWidget(separation_group)
        main_layout.addWidget(imputation_group)
        main_layout.addWidget(preprocess_group)
        
        return data_tab_widget

    def create_arima_tab(self):
        """Fungsi utama untuk membuat semua konten di dalam tab ARIMA SARIMA."""
        arima_tab_widget = QWidget()
        main_layout = QHBoxLayout(arima_tab_widget)
        # main_layout.setContentsMargins(15, 15, 15, 15)
        # main_layout.setSpacing(15)

        # Membuat setiap kolom menggunakan fungsi pembantu
        controls_col = self.create_arima_controls_column()
        acf_pacf_col = self.create_acf_pacf_column()
        params_col = self.create_parameter_column()
        eval_col = self.create_evaluation_column()
        forecast_col = self.create_forecast_column()

        # Menambahkan setiap kolom ke layout utama
        main_layout.addWidget(controls_col, 1)
        main_layout.addWidget(acf_pacf_col, 3) # Stretch factor 1
        main_layout.addWidget(params_col, 2)
        main_layout.addWidget(eval_col, 2)
        main_layout.addWidget(forecast_col, 1)

        return arima_tab_widget

    def create_arima_controls_column(self):
        """Membuat kolom pertama: Kontrol dan Uji Stasioneritas."""
        group_box = QGroupBox("Stationer")
        layout = QGridLayout()
        # layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        layout.addWidget(QLabel("Differentiate"), 0, 0)
        self.n_differentiate = QSpinBox()
        layout.addWidget(self.n_differentiate, 0, 1)
        
        layout.addWidget(QLabel("ADF Test"), 1, 0)
        self.adf_pvalue_line = QLineEdit("N/A")
        self.adf_pvalue_line.setReadOnly(True)
        layout.addWidget(self.adf_pvalue_line, 1, 1)
        
        layout.addWidget(QLabel("Show Lags"), 2, 0)
        self.lags_spinbox = QSpinBox()
        self.lags_spinbox.setRange(10, 100)
        self.lags_spinbox.setValue(30)
        layout.addWidget(self.lags_spinbox, 2, 1)
        
        group_box.setLayout(layout)
        return group_box

    def create_acf_pacf_column(self):
        """Membuat kolom kedua: Plot ACF dan PACF."""
        # Menggunakan QVBoxLayout untuk menumpuk plot secara vertikal
        plot_container = QWidget()
        layout = QVBoxLayout(plot_container)
        
        self.acf_canvas = MplCanvas(self, width=4, height=1, dpi=30)
        self.pacf_canvas = MplCanvas(self, width=4, height=1, dpi=30)
        
        # Inisialisasi dengan placeholder
        # self.create_placeholder_plot(self.acf_canvas, "ACF Plot")
        # self.create_placeholder_plot(self.pacf_canvas, "PACF Plot")
        
        layout.addWidget(self.acf_canvas)
        layout.addWidget(self.pacf_canvas)
        
        return plot_container

    def create_parameter_column(self):
        """Membuat kolom ketiga: Parameter Model SARIMA."""
        group_box = QGroupBox("Model Parameter")
        layout = QGridLayout()
        # layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Parameter (p, d, q)
        layout.addWidget(QLabel("AR"), 0, 0)
        self.p_spinbox = QSpinBox()
        layout.addWidget(self.p_spinbox, 0, 1)
        
        # layout.addWidget(QLabel("I"), 0, 2)
        # self.d_spinbox = QSpinBox()
        # layout.addWidget(self.d_spinbox, 0, 3)
        
        layout.addWidget(QLabel("MA"), 0, 2)
        self.q_spinbox = QSpinBox()
        layout.addWidget(self.q_spinbox, 0, 3)
        
        # Parameter Musiman (P, D, Q, m)
        layout.addWidget(QLabel("SAR"), 1, 0)
        self.P_spinbox = QSpinBox()
        layout.addWidget(self.P_spinbox, 1, 1)

        # layout.addWidget(QLabel("SI"), 1, 2)
        # self.D_spinbox = QSpinBox()
        # layout.addWidget(self.D_spinbox, 1, 3)

        layout.addWidget(QLabel("SMA"), 1, 2)
        self.Q_spinbox = QSpinBox()
        layout.addWidget(self.Q_spinbox, 1, 3)

        layout.addWidget(QLabel("Seasonality"), 2, 0)
        self.m_spinbox = QSpinBox()
        layout.addWidget(self.m_spinbox, 2, 1)
        
        layout.addWidget(QPushButton("Fit Model"), 2, 2, 1, 2)
        
        group_box.setLayout(layout)
        return group_box

    def create_evaluation_column(self):
        """Membuat kolom keempat: Metrik Evaluasi."""
        group_box = QGroupBox("Model Parameter")
        # QFormLayout ideal untuk pasangan label-field
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.aic_line = QLineEdit("N/A")
        self.aic_line.setReadOnly(True)
        
        self.bic_line = QLineEdit("N/A")
        self.bic_line.setReadOnly(True)
        
        self.ljung_box_line = QLineEdit("N/A")
        self.ljung_box_line.setReadOnly(True)

        self.aic_line = QLineEdit("N/A")
        self.aic_line.setReadOnly(True)

        layout.addWidget(QLabel("AIC"), 0, 0)
        layout.addWidget(self.aic_line, 1, 0)

        layout.addWidget(QLabel("BIC"), 0, 1)
        layout.addWidget(self.bic_line, 1, 1)
        
        layout.addWidget(QLabel("Ljung-Box Test"), 0, 2)
        layout.addWidget(self.ljung_box_line, 1, 2)
        
        group_box.setLayout(layout)
        return group_box
        
    def create_forecast_column(self):
        """Membuat kolom kelima: Prediksi."""
        group_box = QGroupBox("Forecasting")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Forecast"))
        self.forecast_steps_spinbox = QSpinBox()
        self.forecast_steps_spinbox.setMinimum(1)
        self.forecast_steps_spinbox.setValue(12)
        layout.addWidget(self.forecast_steps_spinbox)
        
        layout.addWidget(QPushButton("Forecast"))
        
        layout.addStretch()
        group_box.setLayout(layout)
        return group_box

    def create_lstm_tab(self):
        """Fungsi utama untuk membuat semua konten di dalam tab LSTM."""
        lstm_tab_widget = QWidget()
        main_layout = QHBoxLayout(lstm_tab_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Membuat setiap kolom menggunakan fungsi pembantu
        build_col = self.create_lstm_build_column()
        train_col = self.create_lstm_train_column()
        eval_col = self.create_lstm_eval_column()
        forecast_col = self.create_lstm_forecast_column()

        # Menambahkan setiap kolom ke layout utama dengan stretch factor
        main_layout.addWidget(build_col, 2)
        main_layout.addWidget(train_col, 2)
        main_layout.addWidget(eval_col, 1)
        main_layout.addWidget(forecast_col, 1) # Kolom prediksi lebih lebar

        return lstm_tab_widget

    def create_lstm_build_column(self):
        """Membuat kolom pertama: Arsitektur Model LSTM."""
        group_box = QGroupBox("Model Architecture")
        layout = QGridLayout()
        
        self.lstm_window_size_spinbox = QSpinBox()
        self.lstm_window_size_spinbox.setRange(1, 5)
        self.lstm_window_size_spinbox.setValue(1)
        
        self.lstm_layers_spinbox = QSpinBox()
        self.lstm_layers_spinbox.setRange(1, 5)
        self.lstm_layers_spinbox.setValue(1)
        
        self.lstm_units_spinbox = QSpinBox()
        self.lstm_units_spinbox.setMinimum(1)
        self.lstm_units_spinbox.setValue(50)

        self.lstm_dropout_spinbox = QDoubleSpinBox()
        self.lstm_dropout_spinbox.setRange(0.0, 0.9)
        self.lstm_dropout_spinbox.setSingleStep(0.1)
        self.lstm_dropout_spinbox.setValue(0.2)

        self.build_model_button = QPushButton("Build Model")
        self.build_model_button.clicked.connect(self.handle_build_model_lstm)
        
        layout.addWidget(QLabel("Window Size"), 0, 0)
        layout.addWidget(self.lstm_window_size_spinbox, 0, 1) 
        layout.addWidget(QLabel("Number of Hidden Layer"), 1, 0)       
        layout.addWidget(self.lstm_layers_spinbox, 1, 1)
        layout.addWidget(QLabel("Neuron per Layer"), 0, 2)
        layout.addWidget(self.lstm_units_spinbox, 0, 3)
        layout.addWidget(QLabel("Dropout Rate"), 1, 2)
        layout.addWidget(self.lstm_dropout_spinbox, 1, 3)
        layout.addWidget(self.build_model_button, 2, 1, 1, 2)
        
        group_box.setLayout(layout)
        return group_box

    def create_lstm_train_column(self):
        """Membuat kolom kedua: Hyperparameter Pelatihan."""
        group_box = QGroupBox("Model Training")
        layout = QGridLayout()

        self.lstm_epochs_spinbox = QSpinBox()
        self.lstm_epochs_spinbox.setRange(10, 1000)
        self.lstm_epochs_spinbox.setValue(100)

        self.lstm_batch_spinbox = QSpinBox()
        self.lstm_batch_spinbox.setRange(8, 128)
        self.lstm_batch_spinbox.setValue(32)

        self.lstm_optimizer_combo = QComboBox()
        self.lstm_optimizer_combo.addItems(["Adam", "RMSprop", "SGD"])

        self.lstm_loss_combo = QComboBox()
        self.lstm_loss_combo.addItems(["Mean Squared Error", "Mean Absolute Error"])
        
        self.train_progress_bar = QProgressBar()

        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self.handle_train_model_lstm)

        layout.addWidget(QLabel("Epochs"), 0, 0)
        layout.addWidget(self.lstm_epochs_spinbox, 0, 1)
        layout.addWidget(QLabel("Batch Size"), 1, 0)
        layout.addWidget(self.lstm_batch_spinbox, 1, 1)
        layout.addWidget(QLabel("Optimizer"), 2, 0)
        layout.addWidget(self.lstm_optimizer_combo, 2, 1)
        layout.addWidget(QLabel("Loss Function"), 0, 2)
        layout.addWidget(self.lstm_loss_combo, 0, 3)
        layout.addWidget(self.train_model_button, 1, 2, 1, 2)
        layout.addWidget(QLabel("Progress"), 2, 2)
        layout.addWidget(self.train_progress_bar, 2, 3)

        group_box.setLayout(layout)
        return group_box

    def create_lstm_eval_column(self):
        """Membuat kolom ketiga: Metrik Evaluasi."""
        group_box = QGroupBox("Model Evaluation")
        layout = QFormLayout()
        
        self.lstm_mse_line = QLineEdit("N/A")
        self.lstm_mse_line.setReadOnly(True)
        
        # self.lstm_rmse_line = QLineEdit("N/A")
        # self.lstm_rmse_line.setReadOnly(True)
        
        self.lstm_mae_line = QLineEdit("N/A")
        self.lstm_mae_line.setReadOnly(True)
        
        layout.addRow("MSE", self.lstm_mse_line)
        # layout.addRow("RMSE (Root MSE):", self.lstm_rmse_line)
        layout.addRow("MAE", self.lstm_mae_line)
        
        group_box.setLayout(layout)
        return group_box

    def create_lstm_forecast_column(self):
        """Membuat kolom keempat: Prediksi."""
        group_box = QGroupBox("Forecasting")
        layout = QVBoxLayout()
        
        # Kontrol di bagian atas
        control_widget = QWidget()
        control_layout = QFormLayout(control_widget)
        
        self.lstm_forecast_steps_spinbox = QSpinBox()
        self.lstm_forecast_steps_spinbox.setMinimum(1)
        self.lstm_forecast_steps_spinbox.setValue(12)
        
        control_layout.addRow("Forecast Steps:", self.lstm_forecast_steps_spinbox)
        control_layout.addRow(QPushButton("Forecast"))
        
        # Plot di bagian bawah
        # self.lstm_forecast_canvas = MplCanvas(self)
        # self.create_placeholder_plot(self.lstm_forecast_canvas, "Plot Hasil Prediksi LSTM")
        
        layout.addWidget(control_widget)
        # layout.addWidget(self.lstm_forecast_canvas) # Plot mengisi sisa ruang
        
        group_box.setLayout(layout)
        return group_box

    def create_plot_panel(self):
        group_box = QGroupBox("Main Plot")
        layout = QVBoxLayout()

        # Placeholder untuk kanvas Matplotlib
        self.main_plot_canvas = MplCanvas(self, width=100, height=80, dpi=50)

        # Tombol di bawah plot
        show_button = QPushButton("Show Forecast")
        show_button.setEnabled(False)
        show_button.setFixedWidth(100)
        
        layout.addWidget(self.main_plot_canvas)
        layout.addWidget(show_button, alignment=Qt.AlignCenter)
        
        group_box.setLayout(layout)
        return group_box

    def on_variable_column_selected(self, index):
        if index >= 0:
            variable_col= self.variable_combobox.currentText()
            self.nan_data_line.setText(str(count_na(self.dataframe, variable_col))) 
            self.na_marker = self.dataframe[variable_col].isna()
            self.imputation_method.setCurrentIndex(-1)
            self.handle_show_plot()
        else:
            self.nan_data_line.setText("0")
            self.main_plot_canvas.axes.cla()
    
    def on_method_selected(self, index):
        if index >= 0:
            self.random_check.setEnabled(True)
            self.sample_percentage_random_check.setValue(0)
            # variable_col = self.variable_combobox.currentText()
            
            if self.na_marker_train.sum() != 0:
                self.train_plot.remove()
                self.imputed_train_data = data_imputation(self.imputed_train_data, self.imputation_method.currentText())
                self.train_plot, = self.main_plot_canvas.axes.plot(self.imputed_train_data.index, self.imputed_train_data, label='Actual Train', c='blue')

            if self.na_marker_validation.sum() != 0:            
                self.validation_plot.remove()
                self.imputed_validation_data = data_imputation(self.imputed_validation_data, self.imputation_method.currentText())
                self.validation_plot, = self.main_plot_canvas.axes.plot(self.imputed_validation_data.index, self.imputed_validation_data, label='Actual Val', c='orange')
            
            if self.na_marker_test.sum() != 0:
                self.test_plot.remove()
                self.imputed_test_data = data_imputation(self.imputed_test_data, self.imputation_method.currentText())
                self.test_plot, = self.main_plot_canvas.axes.plot(self.imputed_test_data.index, self.imputed_test_data, label='Actual Test', c='yellow')

            self.nan_data_line.setText("0")

            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()

    def on_random_check_state_changed(self, state):
        if state == 2:
            self.sample_percentage_random_check.setEnabled(True)
        else:
            self.sample_percentage_random_check.setValue(0)
            self.sample_percentage_random_check.setEnabled(False)

    def on_n_sample_random_check_changed(self, value):
        if value > 0:
            variable_col = self.variable_combobox.currentText()
            random_check_dataframe, random_valid_indices = remove_random_data(self.dataframe[variable_col].copy(), self.sample_percentage_random_check.value() / 100)
            random_check_dataframe = data_imputation(random_check_dataframe, method=self.imputation_method.currentText())

            actual = random_check_dataframe[random_valid_indices]
            predicted = self.dataframe.loc[random_valid_indices, variable_col]

            self.mse_random_check.setText(f"{MSE(actual, predicted):.2E}")
            self.mae_random_check.setText(f"{MAE(actual, predicted):.2E}")
            self.var_before_random_check.setText(f"{self.dataframe[variable_col].var():.2E}")
            self.var_after_random_check.setText(f"{random_check_dataframe.var():.2E}")

            self.random_check_imputed_plot.remove()
            self.random_check_actual_plot.remove()

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter(random_valid_indices, predicted, label='Imputed Check Data', c='red')
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter(random_valid_indices, actual, label='Check Data', c='green')
            
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()               
        else:
            self.mse_random_check.setText("0")
            self.mae_random_check.setText("0")
            self.var_before_random_check.setText("0")
            self.var_after_random_check.setText("0")

            self.random_check_imputed_plot.remove()
            self.random_check_actual_plot.remove()

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()
    
    # Belum jadi
    def on_scaler_button_state_changed(self, state):
        if state == 2:
            self.train_data_scaled, self.validation_data_scaled, self.test_data_scaled = feature_scaling(self.imputed_train_data.copy(), self.imputed_validation_data.copy(), self.imputed_test_data.copy())

            self.train_plot.remove()
            self.validation_plot.remove()
            self.test_plot.remove()

            self.train_plot, = self.main_plot_canvas.axes.plot(self.train_data_scaled.index, self.train_data_scaled, label='Actual Train', c='blue')
        else:
            self.sample_percentage_random_check.setValue(0)
            self.sample_percentage_random_check.setEnabled(False)

    def handle_import_data(self):
        filter_file = "Semua File Data (*.csv *.xlsx *.xls *.json);;File CSV (*.csv);;File Excel (*.xlsx *.xls);;File JSON (*.json);;Semua File (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih File Data", "", filter_file)

        df, error_message = import_data(file_path)

        if error_message:
            QMessageBox.critical(self, "Error", error_message)
            self.dataframe = None
            self.variable_combobox.clear()
            self.summary_button.setEnabled(False)
            self.show_separation.setEnabled(False)
            self.imputation_method.setEnabled(False)
        else:
            self.dataframe = df
            QMessageBox.information(self, "Sukses", f"Data berhasil dimuat dengan {len(self.dataframe)} baris.")

            column_list = self.dataframe.columns.tolist()
            
            self.variable_combobox.clear()
            self.variable_combobox.addItems(column_list)
            self.variable_combobox.setCurrentIndex(0)

            self.imputation_method.setCurrentIndex(-1)

            self.summary_button.setEnabled(True)
            self.show_separation.setEnabled(True)
            self.imputation_method.setEnabled(True)
    
    def handle_summary(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Data Tidak Ditemukan", "Tidak ada data untuk diringkas.")
            return
        
        self.summary_win = SummaryWindow(self.dataframe)
        self.summary_win.show()

    def handle_show_plot(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Error", "Data belum dimuat!")
            return

        variable_col = self.variable_combobox.currentText()

        try:
            # Konversi kolom waktu ke format datetime (sangat penting untuk plot)
            plot_df = self.dataframe.copy()

            # Bersihkan plot sebelumnya
            self.main_plot_canvas.axes.cla()
            
            # Gambar plot deret waktu
            self.main_plot, = self.main_plot_canvas.axes.plot(plot_df.index, plot_df[variable_col], label='Actual Data', c='blue')
            # self.imputation_plot = self.main_plot_canvas.axes.scatter([], [])
            
            # Atur properti plot agar lebih informatif
            self.main_plot_canvas.axes.set_xlabel(plot_df.index.name)
            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            
            # Segarkan kanvas untuk menampilkan plot baru
            self.main_plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error Plotting", f"Gagal membuat plot: {e}")
            print(f"Error plotting: {e}")
    
    def handle_show_separation(self):
        if self.train_percentage.value() + self.valid_percentage.value() + self.test_percentage.value() != 100:
            QMessageBox.warning(self, "Error", "Jumlah rasio dari train, validation, dan test harus 100%!")
            return
        
        variable_col = self.variable_combobox.currentText()

        self.train_data, self.validation_data, self.test_data = data_separation(self.dataframe[variable_col].copy(), self.train_percentage.value() / 100, self.valid_percentage.value() / 100)
        
        self.train_total.setText(str(len(self.train_data)))
        self.valid_total.setText(str(len(self.validation_data)))
        self.test_total.setText(str(len(self.test_data)))

        self.imputed_train_data = self.train_data.copy()
        self.imputed_validation_data = self.validation_data.copy()
        self.imputed_test_data = self.test_data.copy()

        self.na_marker_train = self.train_data.isna()
        self.na_marker_validation = self.validation_data.isna()
        self.na_marker_test = self.test_data.isna()

        self.main_plot_canvas.axes.cla()

        self.train_plot, = self.main_plot_canvas.axes.plot(self.train_data.index, self.train_data, label="Actual Train", c='blue')
        self.validation_plot, = self.main_plot_canvas.axes.plot(self.validation_data.index, self.validation_data, label="Actual Val", c='orange')
        self.test_plot, = self.main_plot_canvas.axes.plot(self.test_data.index, self.test_data, label="Actual Test", c='yellow')
        self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
        self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

        self.main_plot_canvas.axes.set_xlabel(self.train_data.index.name)
        self.main_plot_canvas.axes.set_ylabel(variable_col)
        self.main_plot_canvas.axes.legend()
        self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.main_plot_canvas.figure.autofmt_xdate()
        self.main_plot_canvas.figure.tight_layout()
        
        # Segarkan kanvas untuk menampilkan plot baru
        self.main_plot_canvas.draw()    

    def handle_build_model_lstm(self):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = create_sliding_window(self.imputed_train_data.copy(), self.imputed_validation_data.copy(), self.imputed_test_data.copy(), self.lstm_window_size_spinbox.value())
    
        lstm_units = [self.lstm_units_spinbox.value() for _ in range(self.lstm_layers_spinbox.value())] 

        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        self.lstm_model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=self.lstm_dropout_spinbox.value()
        )

    def handle_train_model_lstm(self):
        self.lstm_model.compile(optimizer=self.lstm_optimizer_combo.currentText().lower(), loss=re.sub(r'[\s]', '_', self.lstm_loss_combo.currentText().lower()))

        self.history_lstm = self.lstm_model.fit(
            self.X_train, 
            self.y_train, 
            epochs=self.lstm_epochs_spinbox.value(), 
            batch_size=self.lstm_batch_spinbox.value(), 
            validation_data=(self.X_val, self.y_val),
            verbose = 0
        )

        QMessageBox.information(self, "Sukses", "Model sudah dilatih.")

        y_predict = self.lstm_model.predict(self.X_test, verbose = 0)
        self.lstm_mae_line.setText(f"{MAE(self.y_test, y_predict):.2E}")
        self.lstm_mse_line.setText(f"{MSE(self.y_test, y_predict):.2E}")