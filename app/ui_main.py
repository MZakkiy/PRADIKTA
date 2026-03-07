# app/ui_main.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox, 
    QProgressBar, QComboBox, QSpinBox, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox, QFormLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt

from widgets import SummaryWindow, MplCanvas, LossFunctionWindow

import re

import pandas as pd
import numpy as np

import mplcursors

from analysis.data_processor import (
    import_data, count_na, data_imputation, remove_random_data, 
    MAE, MSE, data_separation, feature_scaling
)

from analysis.model import (
    create_sliding_window, build_lstm_model, forecast_lstm
)

from analysis.fire_predict import (
    fire_predict, calculate_pfvi, calculate_kdbi, calculate_kdbi_adj,
    calculate_mkdbi, fire_danger
)

class UIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Prediction Application")
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
        # arima_tab = self.create_arima_tab()
        lstm_tab = self.create_lstm_tab()
        fire_index_tab = self.create_fire_index_tab()
        
        # Menambahkan tab
        tab_widget.addTab(data_tab, "Data Preparation")
        # tab_widget.addTab(arima_tab, "ARIMA SARIMA")
        tab_widget.addTab(lstm_tab, "Build ML Model")
        tab_widget.addTab(fire_index_tab, "Fire Index")

        
        # Contoh isi untuk tab placeholder
        # arima_tab.setLayout(QVBoxLayout())
        # arima_tab.layout()
        
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
        self.show_separation.clicked.connect(self.handle_separation)
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
        self.imputation_method.addItems(['Forward', 'Backward', 'Linear', 'Akima', 'Pchip'])
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
        
        # imputation_layout.addWidget(QLabel("Var Before"), 4, 0)
        # self.var_before_random_check = QLineEdit("0")
        # self.var_before_random_check.setReadOnly(True)
        # imputation_layout.addWidget(self.var_before_random_check, 4, 1)

        # imputation_layout.addWidget(QLabel("Var After"), 4, 2)
        # self.var_after_random_check = QLineEdit("0")
        # self.var_after_random_check.setReadOnly(True)
        # imputation_layout.addWidget(self.var_after_random_check, 4, 3)

        imputation_group.setLayout(imputation_layout)
        
        # --- Kolom 4: Feature Scaling ---
        preprocess_group = QGroupBox("Feature Scaling")
        preprocess_layout = QVBoxLayout()
        self.scale_button = QCheckBox("Scale Data")
        self.scale_button.stateChanged.connect(self.on_scaler_button_state_changed)

        preprocess_layout.addWidget(self.scale_button)
        preprocess_group.setLayout(preprocess_layout)

        # Menambahkan semua grup ke layout utama tab data
        main_layout.addWidget(input_group)
        main_layout.addWidget(separation_group)
        main_layout.addWidget(imputation_group)
        main_layout.addWidget(preprocess_group)
        
        return data_tab_widget

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
        # forecast_col = self.create_lstm_forecast_column()

        # Menambahkan setiap kolom ke layout utama dengan stretch factor
        main_layout.addWidget(build_col, 2)
        main_layout.addWidget(train_col, 2)
        main_layout.addWidget(eval_col, 2)
        # main_layout.addWidget(forecast_col, 1) # Kolom prediksi lebih lebar

        return lstm_tab_widget

    def create_lstm_build_column(self):
        """Membuat kolom pertama: Arsitektur Model LSTM."""
        group_box = QGroupBox("Model Architecture")
        layout = QGridLayout()
        
        self.lstm_window_size_spinbox = QSpinBox()
        self.lstm_window_size_spinbox.setMinimum(1)
        self.lstm_window_size_spinbox.setMaximum(1000)
        # self.lstm_window_size_spinbox.setRange(1, 5)
        self.lstm_window_size_spinbox.setValue(1)
        
        self.lstm_layers_spinbox = QSpinBox()
        self.lstm_layers_spinbox.setMinimum(1)
        self.lstm_layers_spinbox.setMaximum(1000)
        self.lstm_layers_spinbox.setValue(1)
        
        self.lstm_units_spinbox = QSpinBox()
        self.lstm_units_spinbox.setMinimum(1)
        self.lstm_units_spinbox.setMaximum(1000)
        self.lstm_units_spinbox.setValue(32)

        self.lstm_dropout_spinbox = QDoubleSpinBox()
        self.lstm_dropout_spinbox.setRange(0.0, 0.9)
        self.lstm_dropout_spinbox.setSingleStep(0.1)
        self.lstm_dropout_spinbox.setValue(0.2)

        self.lstm_type_combo = QComboBox()
        self.lstm_type_combo.addItems(["LSTM", "GRU"])

        self.lstm_build_model_button = QPushButton("Build Model")
        self.lstm_build_model_button.clicked.connect(self.handle_build_model_lstm)
        self.lstm_build_model_button.setEnabled(False)
        
        layout.addWidget(QLabel("Window Size"), 0, 0)
        layout.addWidget(self.lstm_window_size_spinbox, 0, 1) 
        layout.addWidget(QLabel("Number of Hidden Layer"), 1, 0)       
        layout.addWidget(self.lstm_layers_spinbox, 1, 1)
        layout.addWidget(QLabel("Neuron per Layer"), 0, 2)
        layout.addWidget(self.lstm_units_spinbox, 0, 3)
        layout.addWidget(QLabel("Dropout Rate"), 1, 2)
        layout.addWidget(self.lstm_dropout_spinbox, 1, 3)
        layout.addWidget(QLabel("Model Type"), 2, 0)
        layout.addWidget(self.lstm_type_combo, 2, 1)
        layout.addWidget(self.lstm_build_model_button, 2, 2, 1, 2)
        
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
        self.lstm_batch_spinbox.setMinimum(1)
        self.lstm_batch_spinbox.setValue(32)

        self.lstm_optimizer_combo = QComboBox()
        self.lstm_optimizer_combo.addItems(["Adam", "RMSprop", "SGD"])

        self.lstm_loss_combo = QComboBox()
        self.lstm_loss_combo.addItems(["Mean Squared Error", "Mean Absolute Error"])
        
        self.train_progress_bar = QProgressBar()

        self.lstm_train_model_button = QPushButton("Train Model")
        self.lstm_train_model_button.clicked.connect(self.handle_train_model_lstm)
        self.lstm_train_model_button.setEnabled(False)

        layout.addWidget(QLabel("Epochs"), 0, 0)
        layout.addWidget(self.lstm_epochs_spinbox, 0, 1)
        layout.addWidget(QLabel("Batch Size"), 1, 0)
        layout.addWidget(self.lstm_batch_spinbox, 1, 1)
        layout.addWidget(QLabel("Optimizer"), 2, 0)
        layout.addWidget(self.lstm_optimizer_combo, 2, 1)
        layout.addWidget(QLabel("Loss Function"), 0, 2)
        layout.addWidget(self.lstm_loss_combo, 0, 3)
        layout.addWidget(self.lstm_train_model_button, 1, 2, 1, 2)
        layout.addWidget(QLabel("Progress"), 2, 2)
        layout.addWidget(self.train_progress_bar, 2, 3)

        group_box.setLayout(layout)
        return group_box

    def create_lstm_eval_column(self):
        """Membuat kolom ketiga: Metrik Evaluasi."""
        group_box = QGroupBox("Model Evaluation")
        layout = QGridLayout()
        
        self.lstm_mse_line = QLineEdit("N/A")
        self.lstm_mse_line.setReadOnly(True)
        
        # self.lstm_rmse_line = QLineEdit("N/A")
        # self.lstm_rmse_line.setReadOnly(True)
        
        self.lstm_mae_line = QLineEdit("N/A")
        self.lstm_mae_line.setReadOnly(True)

        self.lstm_show_loss_plot_button = QPushButton("Loss Plot")
        self.lstm_show_loss_plot_button.clicked.connect(self.handle_loss_function)
        self.lstm_show_loss_plot_button.setEnabled(False)
        
        layout.addWidget(QLabel("MSE"), 0, 0)
        layout.addWidget(self.lstm_mse_line, 0, 1)
        layout.addWidget(QLabel("MAE"), 1, 0)
        layout.addWidget(self.lstm_mae_line, 1, 1)
        layout.addWidget(self.lstm_show_loss_plot_button, 2, 0, 1, 2)
        
        group_box.setLayout(layout)
        return group_box
    
    def create_fire_index_tab(self):
        fire_index_tab_widget = QWidget()
        main_layout = QHBoxLayout(fire_index_tab_widget)
        # main_layout.setContentsMargins(15, 15, 15, 15)
        # main_layout.setSpacing(15)

        # Membuat setiap kolom menggunakan fungsi pembantu
        fire_index_col = self.create_fire_index_column()
        predict_fire_index_col = self.create_predict_fire_index_column()

        # Menambahkan setiap kolom ke layout utama dengan stretch factor
        main_layout.addWidget(fire_index_col, 1, Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(predict_fire_index_col, 1, Qt.AlignmentFlag.AlignLeft)

        return fire_index_tab_widget

    def create_fire_index_column(self):
        group_box = QGroupBox("Calculate Drought Index")
        layout = QGridLayout()

        self.drought_index_combo = QComboBox()
        self.drought_index_combo.setPlaceholderText("Choose Drought Index")
        self.drought_index_combo.addItems(["KBDI", "KBDI(adj)", "mKBDI", "PFVI"])
        self.drought_index_combo.currentIndexChanged.connect(self.on_drought_index_selected)
        self.drought_index_combo.setEnabled(False)
        
        layout.addWidget(QLabel("Drought Index"), 0, 0)
        layout.addWidget(self.drought_index_combo, 0, 1)

        group_box.setLayout(layout)

        return group_box
    
    def create_predict_fire_index_column(self):
        group_box = QGroupBox("Calculate Fire Index")
        layout = QGridLayout()

        self.forecast_steps_spinbox = QSpinBox()
        self.forecast_steps_spinbox.setMinimum(1)
        self.forecast_steps_spinbox.setValue(12)

        self.forecast_button = QPushButton("Forecast")
        self.forecast_button.setEnabled(False)
        self.forecast_button.clicked.connect(self.handle_predict_drought_index) 

        layout.addWidget(QLabel("Forecast Steps:"), 0, 0)
        layout.addWidget(self.forecast_steps_spinbox, 0, 1)
        layout.addWidget(self.forecast_button, 1, 0, 1, 2)

        group_box.setLayout(layout)

        return group_box

    def create_plot_panel(self):
        group_box = QGroupBox("Main Plot")
        layout = QVBoxLayout()

        # Placeholder untuk kanvas Matplotlib
        self.main_plot_canvas = MplCanvas(self, width=100, height=80, dpi=120)

        # Tombol di bawah plot
        button_layout = QHBoxLayout()
        self.download_image_button = QPushButton("Download Image")
        self.download_image_button.clicked.connect(self.handle_download_image)
        self.download_image_button.setFixedWidth(150)
        button_layout.addWidget(self.download_image_button)
        button_layout.addStretch()
        
        layout.addWidget(self.main_plot_canvas)
        layout.addLayout(button_layout)
        
        group_box.setLayout(layout)
        return group_box

    def on_variable_column_selected(self, index):
        if index >= 0:
            if self.imputation_method.isEnabled():
                if self.drought_index_combo.isEnabled():
                    self.handle_show_plot_predict()
                elif self.imputation_method.currentIndex() > -1:
                    self.handle_show_plot_imputed()
                else:
                    self.handle_show_plot_separation()
            else:
                variable_col= self.variable_combobox.currentText()
                n_nan = count_na(self.dataframe, variable_col)

                if n_nan == 0:
                    self.lstm_build_model_button.setEnabled(True)

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
            variable_col = self.variable_combobox.currentText()
            
            for column in self.imputed_train_data:
                if self.na_marker_train[column].sum() != 0:
                    self.imputed_train_data[column] = data_imputation(self.train_data[column].copy(), self.imputation_method.currentText())

                if self.na_marker_validation[column].sum() != 0:            
                    self.imputed_validation_data[column] = data_imputation(self.validation_data[column].copy(), self.imputation_method.currentText())
                
                if self.na_marker_test[column].sum() != 0:
                    self.imputed_test_data[column] = data_imputation(self.test_data[column].copy(), self.imputation_method.currentText())
            
            self.train_plot.remove()
            self.validation_plot.remove()
            self.test_plot.remove()
            self.plot_cursor.remove()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.train_plot, = self.main_plot_canvas.axes.plot(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy(), label='Actual Train')
            self.validation_plot, = self.main_plot_canvas.axes.plot(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy(), label='Actual Val')
            self.test_plot, = self.main_plot_canvas.axes.plot(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy(), label='Actual Test')

            self.train_scatter = self.main_plot_canvas.axes.scatter(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy())
            self.validation_scatter = self.main_plot_canvas.axes.scatter(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy())
            self.test_scatter = self.main_plot_canvas.axes.scatter(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy())

            self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter], hover=True)
            
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
                )
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)

            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            self.main_plot_canvas.draw()

            self.lstm_build_model_button.setEnabled(True)
            self.nan_data_line.setText("0")

    def on_drought_index_selected(self, index):
        self.handle_optimize_drought_index_params(index)

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

            self.random_check_imputed_plot.remove()
            self.random_check_actual_plot.remove()

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter(random_valid_indices, predicted, label='Imputed Check Data', c='red')
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter(random_valid_indices, actual, label='Check Data', c='green')
            
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()               
        else:
            self.mse_random_check.setText("0")
            self.mae_random_check.setText("0")

            self.random_check_imputed_plot.remove()
            self.random_check_actual_plot.remove()

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()
    
    def on_scaler_button_state_changed(self, state):
        if state == 2:
            self.train_data_scaled = {}
            self.validation_data_scaled  = {}
            self.test_data_scaled = {}

            self.scaler = {}

            for column in self.imputed_train_data:
                self.train_data_scaled[column], self.validation_data_scaled[column], self.test_data_scaled[column], self.scaler[column] = feature_scaling(self.imputed_train_data[column].copy(), self.imputed_validation_data[column].copy(), self.imputed_test_data[column].copy())

        else:
            self.sample_percentage_random_check.setValue(0)
            self.sample_percentage_random_check.setEnabled(False)

    def handle_import_data(self):
        filter_file = "All Data Files (*.csv *.xlsx *.xls *.json);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;JSON Files (*.json);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", filter_file)

        df, error_message = import_data(file_path)

        if error_message:
            QMessageBox.critical(self, "Error", error_message)
            # self.dataframe = None
            # self.variable_combobox.clear()
            # self.summary_button.setEnabled(False)
            # self.show_separation.setEnabled(False)
            # self.imputation_method.setEnabled(False)
            # self.main_plot_canvas.axes.cla()

        else:
            self.reset_ui()
            self.dataframe = df
            QMessageBox.information(self, "Success", f"Data successfully loaded with {len(self.dataframe)} rows.")

            column_list = self.dataframe.columns.tolist()
            
            self.variable_combobox.clear()
            self.variable_combobox.addItems(column_list)
            self.variable_combobox.setCurrentIndex(0)

            self.imputation_method.setCurrentIndex(-1)

            self.summary_button.setEnabled(True)
            self.show_separation.setEnabled(True)
    
    def handle_summary(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Data Not Found", "There is no data to summarize.")
            return
        
        self.summary_win = SummaryWindow(self.dataframe)
        self.summary_win.show()

    def handle_show_plot(self):
        if self.dataframe is None:
            QMessageBox.warning(self, "Error", "Data has not been loaded!")
            return

        variable_col = self.variable_combobox.currentText()

        try:
            # Konversi kolom waktu ke format datetime (sangat penting untuk plot)
            plot_df = self.dataframe.copy()

            # Bersihkan plot sebelumnya
            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)
            
            # Gambar plot deret waktu
            self.main_plot, = self.main_plot_canvas.axes.plot(plot_df.index, plot_df[variable_col], label='Actual Data')
            self.main_scatter = self.main_plot_canvas.axes.scatter(plot_df.index, plot_df[variable_col], label='Actual Data', alpha=0)
            # self.imputation_plot = self.main_plot_canvas.axes.scatter([], [])
            
            # Atur properti plot agar lebih informatif
            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()

            self.plot_cursor = mplcursors.cursor(self.main_scatter, hover=True)
            
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
                )
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            
            # Segarkan kanvas untuk menampilkan plot baru
            self.main_plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error Plotting", f"Failed to create plot: {e}")
            print(f"Error plotting: {e}")
    
    def handle_separation(self):
        if self.train_percentage.value() + self.valid_percentage.value() + self.test_percentage.value() != 100:
            QMessageBox.warning(self, "Error", "The sum of the ratios of train, validation, and test must be 100%!")
            return
        
        variable_col = self.variable_combobox.currentText()

        self.train_data = {}
        self.test_data = {}
        self.validation_data = {}

        self.na_marker_train = {}
        self.na_marker_validation = {}
        self.na_marker_test = {}

        for column in self.dataframe.columns:   
            self.train_data[column], self.validation_data[column], self.test_data[column] = data_separation(self.dataframe[column].copy(), self.train_percentage.value() / 100, self.valid_percentage.value() / 100)

            self.na_marker_train[column] = self.train_data[column].isna()
            self.na_marker_validation[column] = self.validation_data[column].isna()
            self.na_marker_test[column] = self.test_data[column].isna()
        
        self.imputed_train_data = self.train_data.copy()
        self.imputed_validation_data = self.validation_data.copy()
        self.imputed_test_data = self.test_data.copy()
        
        self.train_total.setText(str(len(self.train_data[column])))
        self.valid_total.setText(str(len(self.validation_data[column])))
        self.test_total.setText(str(len(self.test_data[column])))

        self.main_plot_canvas.axes.cla()

        self.plot_cursor.remove()

        self.train_plot, = self.main_plot_canvas.axes.plot(self.train_data[variable_col].index, self.train_data[variable_col], label="Actual Train")
        self.validation_plot, = self.main_plot_canvas.axes.plot(self.validation_data[variable_col].index, self.validation_data[variable_col], label="Actual Val")
        self.test_plot, = self.main_plot_canvas.axes.plot(self.test_data[variable_col].index, self.test_data[variable_col], label="Actual Test")

        self.train_scatter = self.main_plot_canvas.axes.scatter(self.train_data[variable_col].index, self.train_data[variable_col])
        self.validation_scatter = self.main_plot_canvas.axes.scatter(self.validation_data[variable_col].index, self.validation_data[variable_col])
        self.test_scatter = self.main_plot_canvas.axes.scatter(self.test_data[variable_col].index, self.test_data[variable_col])

        self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter], hover=True)
            
        @self.plot_cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(
                f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
            )
            sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)

        self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
        self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

        self.main_plot_canvas.axes.set_xlabel("Time (day)")
        self.main_plot_canvas.axes.set_ylabel(variable_col)
        self.main_plot_canvas.axes.legend()
        self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.main_plot_canvas.figure.autofmt_xdate()
        self.main_plot_canvas.figure.tight_layout()
        
        # Segarkan kanvas untuk menampilkan plot baru
        self.main_plot_canvas.draw() 

        self.imputation_method.setEnabled(True)   

    def handle_show_plot_separation(self):
        variable_col = self.variable_combobox.currentText()

        try:
            # Konversi kolom waktu ke format datetime (sangat penting untuk plot)
            train_plot_df = self.train_data[variable_col].copy()
            validation_plot_df = self.validation_data[variable_col].copy()
            test_plot_df = self.test_data[variable_col].copy()

            # Bersihkan plot sebelumnya
            self.main_plot_canvas.axes.cla()

            self.plot_cursor.remove()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.main_plot_canvas.axes.set_prop_cycle(None)
            
            self.train_plot, = self.main_plot_canvas.axes.plot(train_plot_df.index, train_plot_df, label="Actual Train")
            self.validation_plot, = self.main_plot_canvas.axes.plot(validation_plot_df.index, validation_plot_df, label="Actual Val")
            self.test_plot, = self.main_plot_canvas.axes.plot(test_plot_df.index, test_plot_df, label="Actual Test")

            self.train_scatter = self.main_plot_canvas.axes.scatter(train_plot_df.index, train_plot_df)
            self.validation_scatter = self.main_plot_canvas.axes.scatter(validation_plot_df.index, validation_plot_df)
            self.test_scatter = self.main_plot_canvas.axes.scatter(test_plot_df.index, test_plot_df)

            self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter], hover=True)
            
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'Drought Index:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
                )
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)

            # self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
            # self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            
            # Segarkan kanvas untuk menampilkan plot baru
            self.main_plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error Plotting", f"Failed to create plot: {e}")
            print(f"Error plotting: {e}")

    def handle_show_plot_imputed(self):
        variable_col = self.variable_combobox.currentText()

        try:
            self.main_plot_canvas.axes.cla()

            self.plot_cursor.remove()

            self.main_plot_canvas.axes.set_prop_cycle(None)
            
            self.train_plot, = self.main_plot_canvas.axes.plot(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy(), label='Actual Train')
            self.validation_plot, = self.main_plot_canvas.axes.plot(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy(), label='Actual Val')
            self.test_plot, = self.main_plot_canvas.axes.plot(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy(), label='Actual Test')

            self.predict_plot, = self.main_plot_canvas.axes.plot([], [], label="Predict Test")

            self.train_scatter = self.main_plot_canvas.axes.scatter(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy())
            self.validation_scatter = self.main_plot_canvas.axes.scatter(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy())
            self.test_scatter = self.main_plot_canvas.axes.scatter(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy())

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])

            self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter], hover=True)
            
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
                )
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            
            # Segarkan kanvas untuk menampilkan plot baru
            self.main_plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error Plotting", f"Failed to create plot: {e}")
            print(f"Error plotting: {e}")
    
    def handle_show_plot_predict(self):
        variable_col = self.variable_combobox.currentText()

        try:
            if self.scale_button.isChecked():
                self.lstm_mae_line.setText(f"{MAE(self.scaler[variable_col].inverse_transform(self.y_test[variable_col].copy().reshape(-1, 1)), self.scaler[variable_col].inverse_transform(self.y_predict[variable_col].copy())):.2E}")
                self.lstm_mse_line.setText(f"{MSE(self.scaler[variable_col].inverse_transform(self.y_test[variable_col].copy().reshape(-1, 1)), self.scaler[variable_col].inverse_transform(self.y_predict[variable_col].copy())):.2E}")
            else:
                self.lstm_mae_line.setText(f"{MAE(self.y_test[variable_col].copy(), self.y_predict[variable_col].copy()):.2E}")
                self.lstm_mse_line.setText(f"{MSE(self.y_test[variable_col].copy(), self.y_predict[variable_col].copy()):.2E}")

            self.main_plot_canvas.axes.cla()

            self.plot_cursor.remove()

            self.main_plot_canvas.axes.set_prop_cycle(None)
            
            self.train_plot, = self.main_plot_canvas.axes.plot(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy(), label='Actual Train')
            self.validation_plot, = self.main_plot_canvas.axes.plot(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy(), label='Actual Val')
            self.test_plot, = self.main_plot_canvas.axes.plot(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy(), label='Actual Test')

            self.train_scatter = self.main_plot_canvas.axes.scatter(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy())
            self.validation_scatter = self.main_plot_canvas.axes.scatter(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy())
            self.test_scatter = self.main_plot_canvas.axes.scatter(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy())
            
            if self.scale_button.isChecked():
                self.predict_plot, = self.main_plot_canvas.axes.plot(self.test_data[variable_col].index, self.scaler[variable_col].inverse_transform(self.y_predict[variable_col]), label="Predict Test")
                self.predict_scatter = self.main_plot_canvas.axes.scatter(self.test_data[variable_col].index, self.scaler[variable_col].inverse_transform(self.y_predict[variable_col]))
            else:
                self.predict_plot, = self.main_plot_canvas.axes.plot(self.test_data[variable_col].index, self.y_predict[variable_col], label="Predict Test")
                self.predict_scatter = self.main_plot_canvas.axes.scatter(self.test_data[variable_col].index, self.y_predict[variable_col])

            self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter, self.predict_scatter], hover=True)
                
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
                )
                sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
                sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel(variable_col)
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            
            # Segarkan kanvas untuk menampilkan plot baru
            self.main_plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error Plotting", f"Failed to create plot: {e}")
            print(f"Error plotting: {e}")
    

    def handle_build_model_lstm(self):
        self.X_train = {}
        self.y_train = {}
        self.X_val = {}
        self.y_val = {}
        self.X_test = {}
        self.y_test = {}

        self.ml_model = {}

        for column in self.imputed_train_data:
            if self.scale_button.isChecked():
                self.X_train[column], self.y_train[column], self.X_val[column], self.y_val[column], self.X_test[column], self.y_test[column] = create_sliding_window(self.train_data_scaled[column].copy(), self.validation_data_scaled[column].copy(), self.test_data_scaled[column].copy(), self.lstm_window_size_spinbox.value())
            else:    
                self.X_train[column], self.y_train[column], self.X_val[column], self.y_val[column], self.X_test[column], self.y_test[column] = create_sliding_window(self.imputed_train_data[column].copy(), self.imputed_validation_data[column].copy(), self.imputed_test_data[column].copy(), self.lstm_window_size_spinbox.value())
            
            lstm_units = [self.lstm_units_spinbox.value() for _ in range(self.lstm_layers_spinbox.value())] 

            input_shape = (self.X_train[column].shape[1], self.X_train[column].shape[2])

            self.ml_model[column] = build_lstm_model(
                model_type=self.lstm_type_combo.currentText(),
                input_shape=input_shape,
                lstm_units=lstm_units,
                dropout_rate=self.lstm_dropout_spinbox.value()
            )

        QMessageBox.information(self, "Success", "Build Success.")

        self.lstm_train_model_button.setEnabled(True)

    def handle_train_model_lstm(self):
        self.y_predict = {}
        self.history = {}

        for column in self.ml_model:
            self.ml_model[column].compile(optimizer=self.lstm_optimizer_combo.currentText().lower(), loss=re.sub(r'[\s]', '_', self.lstm_loss_combo.currentText().lower()))

            self.history[column] = self.ml_model[column].fit(
                self.X_train[column], 
                self.y_train[column], 
                epochs=self.lstm_epochs_spinbox.value(), 
                batch_size=self.lstm_batch_spinbox.value(), 
                validation_data=(self.X_val[column], self.y_val[column]),
                verbose = 0
            )

            self.lstm_show_loss_plot_button.setEnabled(True)

            self.y_predict[column] = self.ml_model[column].predict(self.X_test[column], verbose = 0)
        
        variable_col = self.variable_combobox.currentText()

        if self.scale_button.isChecked():
            self.lstm_mae_line.setText(f"{MAE(self.scaler[variable_col].inverse_transform(self.y_test[variable_col].copy().reshape(-1, 1)), self.scaler[variable_col].inverse_transform(self.y_predict[variable_col].copy())):.2E}")
            self.lstm_mse_line.setText(f"{MSE(self.scaler[variable_col].inverse_transform(self.y_test[variable_col].copy().reshape(-1, 1)), self.scaler[variable_col].inverse_transform(self.y_predict[variable_col].copy())):.2E}")
        else:
            self.lstm_mae_line.setText(f"{MAE(self.y_test[variable_col].copy(), self.y_predict[variable_col].copy()):.2E}")
            self.lstm_mse_line.setText(f"{MSE(self.y_test[variable_col].copy(), self.y_predict[variable_col].copy()):.2E}")
        
        self.main_plot_canvas.axes.cla()

        self.plot_cursor.remove()

        self.main_plot_canvas.axes.set_prop_cycle(None)
        
        self.train_plot, = self.main_plot_canvas.axes.plot(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy(), label='Actual Train')
        self.validation_plot, = self.main_plot_canvas.axes.plot(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy(), label='Actual Val')
        self.test_plot, = self.main_plot_canvas.axes.plot(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy(), label='Actual Test')

        self.train_scatter = self.main_plot_canvas.axes.scatter(self.imputed_train_data[variable_col].copy().index, self.imputed_train_data[variable_col].copy())
        self.validation_scatter = self.main_plot_canvas.axes.scatter(self.imputed_validation_data[variable_col].copy().index, self.imputed_validation_data[variable_col].copy())
        self.test_scatter = self.main_plot_canvas.axes.scatter(self.imputed_test_data[variable_col].copy().index, self.imputed_test_data[variable_col].copy())
        
        if self.scale_button.isChecked():
            self.predict_plot, = self.main_plot_canvas.axes.plot(self.test_data[variable_col].index, self.scaler[variable_col].inverse_transform(self.y_predict[variable_col]), label="Predict Test")
            self.predict_scatter = self.main_plot_canvas.axes.scatter(self.test_data[variable_col].index, self.scaler[variable_col].inverse_transform(self.y_predict[variable_col]))
        else:
            self.predict_plot, = self.main_plot_canvas.axes.plot(self.test_data[variable_col].index, self.y_predict[variable_col], label="Predict Test")
            self.predict_scatter = self.main_plot_canvas.axes.scatter(self.test_data[variable_col].index, self.y_predict[variable_col])

        self.plot_cursor = mplcursors.cursor([self.train_scatter, self.validation_scatter, self.test_scatter, self.predict_scatter], hover=True)
            
        @self.plot_cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(
                f'{variable_col}:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}'
            )
            sel.annotation.get_bbox_patch().set(facecolor='lightblue', alpha=0.7)
            sel.annotation.arrow_patch.set(arrowstyle="simple", facecolor="white", alpha=0.7)
            
        self.main_plot_canvas.axes.set_xlabel("Time (day)")
        self.main_plot_canvas.axes.set_ylabel(variable_col)
        self.main_plot_canvas.axes.legend()
        self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.main_plot_canvas.figure.autofmt_xdate()
        self.main_plot_canvas.figure.tight_layout()
        self.main_plot_canvas.draw()  

        self.drought_index_combo.setEnabled(True)

        QMessageBox.information(self, "Success", "Training Success.")

    def handle_loss_function(self):
        self.loss_function_win = LossFunctionWindow(self.history[self.variable_combobox.currentText()])
        self.loss_function_win.show()

    def handle_optimize_drought_index_params(self, index):
        features = []

        for column in self.dataframe.columns:
            feature_past = np.concatenate([
                self.imputed_train_data[column].values, 
                self.imputed_validation_data[column].values, 
                self.imputed_test_data[column].values
            ]) 

            feature_past = pd.Series(feature_past)

            features.append(feature_past)
        
        if index == 0:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            self.drought_index_values = calculate_kdbi(Temp=features[0], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            self.drought_index_values = np.clip(self.drought_index_values, 0, 203)

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='KBDI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            self.drought_index_predicted_plot = self.main_plot_canvas.axes.plot([], [])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("KBDI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        elif index == 1:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            self.drought_index_values = calculate_kdbi_adj(Temp=features[0], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            self.drought_index_values = np.clip(self.drought_index_values, 0, 203)

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='KBDI(adj)')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            self.drought_index_predicted_plot = self.main_plot_canvas.axes.plot([], [])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("KBDI(adj)")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        elif index == 2:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            self.drought_index_values, self.params = fire_predict(Temp=features[0], WT=features[1], SM=features[2], Rf=features[3], drought_index="mKBDI")
        
            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='mKBDI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            self.drought_index_predicted_plot = self.main_plot_canvas.axes.plot([], [])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("mKBDI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        else:
            self.drought_index_values, self.params = fire_predict(Temp=features[0], WT=features[1], SM=features[2], Rf=features[3])

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='PFVI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            self.drought_index_predicted_plot = self.main_plot_canvas.axes.plot([], [])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("PFVI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 305])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        self.plot_cursor.remove()

        self.plot_cursor = mplcursors.cursor(self.drought_index_scatter, hover=True)

        if index == 3:
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'Drought Index:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}\nFire Danger:{fire_danger(sel.target[1], "PFVI")}'
                )
                
        else:    
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'Drought Index:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}\nFire Danger:{fire_danger(sel.target[1], "KBDI")}'
                )

        self.main_plot_canvas.draw()
    
        self.forecast_button.setEnabled(True)
    
    def handle_predict_drought_index(self):
        features = []

        for column in self.dataframe.columns:
            feature_past = np.concatenate([
                self.imputed_train_data[column].values, 
                self.imputed_validation_data[column].values, 
                self.imputed_test_data[column].values
            ]) 

            feature_past = pd.Series(feature_past.reshape(-1))

            last_known_sequence = self.X_test[column][-1]

            future_forecast = forecast_lstm(
                model=self.ml_model[column],
                initial_sequence=last_known_sequence,
                n_steps_to_predict=self.forecast_steps_spinbox.value()
            )

            if self.scale_button.isChecked():
                feature_future = self.scaler[column].inverse_transform(future_forecast)
                feature_future = pd.Series(feature_future.reshape(-1))

                feature = pd.concat([feature_past, feature_future], ignore_index=True)

                features.append(feature)
            else:
                feature_future = pd.Series(future_forecast.reshape(-1))
                
                feature = pd.concat([feature_past, feature_future], ignore_index=True)

                features.append(feature)
        
        if self.drought_index_combo.currentIndex() == 0:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            drought_index_values = calculate_kdbi(Temp=features[0], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            drought_index_values = np.clip(drought_index_values, 0, 203)
            print(drought_index_values)
            self.main_plot_canvas.axes.cla()
        
            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='KBDI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            last_index = self.dataframe.index[-1]
            predicted_index = [t for t in range(last_index+1, last_index+1+self.forecast_steps_spinbox.value())]
            self.drought_index_predicted_plot, = self.main_plot_canvas.axes.plot(predicted_index, drought_index_values[predicted_index], label="KBDI Predicted")
            self.drought_index_predicted_scatter = self.main_plot_canvas.axes.scatter(predicted_index, drought_index_values[predicted_index])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("KBDI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        elif self.drought_index_combo.currentIndex() == 1:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            drought_index_values = calculate_kdbi_adj(Temp=features[0], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            drought_index_values = np.clip(drought_index_values, 0, 203)

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='KBDI(adj)')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            last_index = self.dataframe.index[-1]
            predicted_index = [t for t in range(last_index+1, last_index+1+self.forecast_steps_spinbox.value())]
            self.drought_index_predicted_plot, = self.main_plot_canvas.axes.plot(predicted_index, drought_index_values[predicted_index], label="KBDI(adj) Predicted")
            self.drought_index_predicted_scatter = self.main_plot_canvas.axes.scatter(predicted_index, drought_index_values[predicted_index])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("KBDI(adj)")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
        
        elif self.drought_index_combo.currentIndex() == 2:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            drought_index_values = calculate_mkdbi(self.params, Temp=features[0], WT=features[1], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            drought_index_values = np.clip(drought_index_values, 0, 203)

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='mKBDI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            last_index = self.dataframe.index[-1]
            predicted_index = [t for t in range(last_index+1, last_index+1+self.forecast_steps_spinbox.value())]
            self.drought_index_predicted_plot, = self.main_plot_canvas.axes.plot(predicted_index, drought_index_values[predicted_index], label="mKBDI Predicted")
            self.drought_index_predicted_scatter = self.main_plot_canvas.axes.scatter(predicted_index, drought_index_values[predicted_index])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("mKBDI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 208])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()
            
        else:
            Rf_b = np.roll(features[3], 1)
            Rf_b[0] = np.nan

            drought_index_values = calculate_pfvi(params=self.params, Temp=features[0], WT=features[1], SM=features[2], Rf=features[3], Rf_b=Rf_b, R0=3000, dt=1)
            drought_index_values = np.clip(drought_index_values, 0, 300)

            self.main_plot_canvas.axes.cla()

            self.main_plot_canvas.axes.set_prop_cycle(None)

            self.drought_index_plot, = self.main_plot_canvas.axes.plot(self.dataframe.index, self.drought_index_values[self.dataframe.index], label='PFVI')
            self.drought_index_scatter = self.main_plot_canvas.axes.scatter(self.dataframe.index, self.drought_index_values[self.dataframe.index])

            last_index = self.dataframe.index[-1]
            predicted_index = [t for t in range(last_index+1, last_index+1+self.forecast_steps_spinbox.value())]
            self.drought_index_predicted_plot, = self.main_plot_canvas.axes.plot(predicted_index, drought_index_values[predicted_index], label="PFVI Predicted")
            self.drought_index_predicted_scatter = self.main_plot_canvas.axes.scatter(predicted_index, drought_index_values[predicted_index])

            self.main_plot_canvas.axes.set_xlabel("Time (day)")
            self.main_plot_canvas.axes.set_ylabel("PFVI")
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.axes.set_ylim([-5, 305])
            self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.main_plot_canvas.figure.autofmt_xdate()
            self.main_plot_canvas.figure.tight_layout()

        self.plot_cursor.remove()

        self.plot_cursor = mplcursors.cursor([self.drought_index_scatter, self.drought_index_predicted_scatter], hover=True)

        if self.drought_index_combo.currentIndex() == 3:
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'Drought Index:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}\nFire Danger:{fire_danger(sel.target[1], "PFVI")}'
                )
                
        else:    
            @self.plot_cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(
                    f'Drought Index:{sel.target[1]:.2f}\nt:{sel.target[0]:.2f}\nFire Danger:{fire_danger(sel.target[1], "KBDI")}'
                )
                    
        self.main_plot_canvas.draw()

    def reset_ui(self):
        """Reset UI to initial state and clear all data variables"""
        # Clear dataframes and data storage
        self.dataframe = None
        self.train_data = {}
        self.validation_data = {}
        self.test_data = {}
        self.imputed_train_data = {}
        self.imputed_validation_data = {}
        self.imputed_test_data = {}
        self.train_data_scaled = {}
        self.validation_data_scaled = {}
        self.test_data_scaled = {}
        self.scaler = {}
        self.na_marker_train = {}
        self.na_marker_validation = {}
        self.na_marker_test = {}
        
        # Clear model related variables
        self.lstm_model = None
        self.history = None
        self.predictions_lstm = None
        self.test_predictions = None
        self.drought_index_values = None
        self.params = {}
        
        # Clear plot references
        self.train_plot = None
        self.validation_plot = None
        self.test_plot = None
        self.train_scatter = None
        self.validation_scatter = None
        self.test_scatter = None
        self.random_check_imputed_plot = None
        self.random_check_actual_plot = None
        self.drought_index_plot = None
        self.drought_index_scatter = None
        self.drought_index_predicted_plot = None
        self.drought_index_predicted_scatter = None
        self.plot_cursor = None
        
        # Clear plot canvas
        self.main_plot_canvas.axes.clear()
        self.main_plot_canvas.draw()
        
        # Reset UI input fields
        self.variable_combobox.clear()
        self.nan_data_line.setText("0")
        self.train_percentage.setValue(0)
        self.valid_percentage.setValue(0)
        self.test_percentage.setValue(0)
        self.train_total.setText("0")
        self.valid_total.setText("0")
        self.test_total.setText("0")
        self.mse_random_check.setText("0")
        self.mae_random_check.setText("0")
        self.sample_percentage_random_check.setValue(0)
        self.random_check.setChecked(False)
        self.scale_button.setChecked(False)
        
        # Reset LSTM settings
        self.lstm_window_size_spinbox.setValue(0)
        self.lstm_layers_spinbox.setValue(0)
        self.lstm_units_spinbox.setValue(0)
        self.lstm_dropout_spinbox.setValue(0)
        self.lstm_epochs_spinbox.setValue(0)
        self.lstm_batch_spinbox.setValue(0)
        
        # Disable buttons that require data
        self.imputation_method.setEnabled(False)
        self.random_check.setEnabled(False)
        self.sample_percentage_random_check.setEnabled(False)
        self.lstm_build_model_button.setEnabled(False)
        self.lstm_train_model_button.setEnabled(False)
        self.lstm_show_loss_plot_button.setEnabled(False)
        self.forecast_button.setEnabled(False)
        self.drought_index_combo.setEnabled(False)
        self.show_separation.setEnabled(False)
        
        # Close summary window if open
        if self.summary_win is not None:
            self.summary_win.close()
            self.summary_win = None

    def handle_download_image(self):
        """Save the current plot as an image file."""
        try:
            # Open file dialog to choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot Image",
                "",
                "PNG Images (*.png);;JPEG Images (*.jpg);;PDF Documents (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Save the figure
                self.main_plot_canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Image saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
            print(f"Error saving image: {e}")