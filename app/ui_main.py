# app/ui_main.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox, 
    QProgressBar, QComboBox, QSpinBox, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

from widgets import SummaryWindow, MplCanvas
from analysis.data_processor import import_data, count_na, data_imputation, remove_random_data, MAE, MSE, data_separation

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
        update_settings_panel = self.create_update_settings_panel()
        top_panel_layout.addWidget(update_settings_panel)

        # Panel Kanan: Tab Kontrol
        main_tabs = self.create_main_tabs()
        top_panel_layout.addWidget(main_tabs, 1)

        # --- Membuat dan Menambahkan Panel Bawah (Plot) ---
        plot_panel = self.create_plot_panel()

        # Menambahkan semua bagian ke layout utama
        main_layout.addLayout(top_panel_layout)
        main_layout.addWidget(plot_panel)
        
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
        arima_tab = QWidget() # Placeholder
        lstm_tab = QWidget() # Placeholder
        
        # Menambahkan tab
        tab_widget.addTab(data_tab, "Data")
        tab_widget.addTab(arima_tab, "ARIMA SARIMA")
        tab_widget.addTab(lstm_tab, "LSTM")
        
        # Contoh isi untuk tab placeholder
        arima_tab.setLayout(QVBoxLayout())
        arima_tab.layout().addWidget(QLabel("Pengaturan untuk ARIMA/SARIMA akan ada di sini."))
        
        lstm_tab.setLayout(QVBoxLayout())
        lstm_tab.layout().addWidget(QLabel("Pengaturan untuk LSTM akan ada di sini."))

        return tab_widget
        
    def create_data_tab(self):
        data_tab_widget = QWidget()
        main_layout = QHBoxLayout(data_tab_widget)
        
        # --- Kolom 1: Input Data ---
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()

        import_data_button = QPushButton("Import Data")
        import_data_button.setObjectName("ImportButton")
        import_data_button.clicked.connect(self.handle_import_data)
        
        self.summary_button = QPushButton("Summary")
        self.summary_button.setObjectName("SummaryButton")
        self.summary_button.clicked.connect(self.handle_summary)
        self.summary_button.setEnabled(False)

        self.variable_combobox = QComboBox()
        self.variable_combobox.currentIndexChanged.connect(self.on_variable_column_selected)

        input_layout.addWidget(import_data_button)
        input_layout.addWidget(self.summary_button)
        input_layout.addWidget(QLabel("Variable"))
        input_layout.addWidget(self.variable_combobox)
        input_layout.addWidget(QLabel("NaN Data"))
        self.nan_data_line = QLineEdit("0")
        self.nan_data_line.setReadOnly(True)
        input_layout.addWidget(self.nan_data_line)
        input_layout.addStretch()
        input_group.setLayout(input_layout)

        # --- Kolom 2: Data Imputation ---
        self.imputation_method = QComboBox()
        self.imputation_method.setPlaceholderText("Choose Method")
        self.imputation_method.addItems(['Forward', 'Backward', 'Linear', 'Nearest', 'Akima', 'Pchip'])
        self.imputation_method.currentIndexChanged.connect(self.on_method_selected)
        self.imputation_method.setEnabled(False)

        self.random_check = QCheckBox("Random Check (%)")
        self.random_check.stateChanged.connect(self.on_random_check_state_changed)
        self.random_check.setEnabled(False)

        self.n_sample_random_check = QSpinBox()
        self.n_sample_random_check.valueChanged.connect(self.on_n_sample_random_check_changed)
        self.n_sample_random_check.setEnabled(False)

        imputation_group = QGroupBox("Data Imputation")
        imputation_layout = QGridLayout()
        imputation_layout.addWidget(QLabel("Method"), 0, 0, 1, 2)
        imputation_layout.addWidget(self.imputation_method, 1, 0, 1, 2)
        imputation_layout.addWidget(self.random_check, 2, 0)
        imputation_layout.addWidget(self.n_sample_random_check, 2, 1)
        
        # Menambahkan metrik
        imputation_layout.addWidget(QLabel("MSE"), 3, 0)
        self.mse_random_check = QLineEdit("0")
        self.mse_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.mse_random_check, 3, 1)

        imputation_layout.addWidget(QLabel("MAE"), 4, 0)
        self.mae_random_check = QLineEdit("0")
        self.mae_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.mae_random_check, 4, 1)
        
        imputation_layout.addWidget(QLabel("Var Before"), 5, 0)
        self.var_before_random_check = QLineEdit("0")
        self.var_before_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.var_before_random_check, 5, 1)

        imputation_layout.addWidget(QLabel("Var After"), 6, 0)
        self.var_after_random_check = QLineEdit("0")
        self.var_after_random_check.setReadOnly(True)
        imputation_layout.addWidget(self.var_after_random_check, 6, 1)

        imputation_group.setLayout(imputation_layout)
        
        # --- Kolom 3 & 4: Preprocessing & Data Separation ---
        right_column_layout = QVBoxLayout()
        
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout()
        preprocess_layout.addWidget(QCheckBox("Standardize"))
        preprocess_group.setLayout(preprocess_layout)

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

        separation_group = QGroupBox("Data Separation")
        separation_layout = QGridLayout()
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
        
        right_column_layout.addWidget(preprocess_group)
        right_column_layout.addWidget(separation_group)
        
        # Menambahkan semua grup ke layout utama tab data
        main_layout.addWidget(input_group)
        main_layout.addWidget(imputation_group)
        main_layout.addLayout(right_column_layout)
        
        return data_tab_widget

    def create_plot_panel(self):
        group_box = QGroupBox("Main Plot")
        layout = QVBoxLayout()

        # Placeholder untuk kanvas Matplotlib
        self.main_plot_canvas = MplCanvas(self, width=10, height=8, dpi=100)

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
            self.n_sample_random_check.setValue(0)
            variable_col = self.variable_combobox.currentText()
            self.imputed_dataframe = self.dataframe.copy()
            self.imputed_dataframe[variable_col] = data_imputation(self.imputed_dataframe, variable_col, self.imputation_method.currentText())

            self.nan_data_line.setText("0")

            self.main_plot.remove()
            self.imputation_plot.remove()
            self.main_plot, = self.main_plot_canvas.axes.plot(self.imputed_dataframe.index, self.imputed_dataframe[variable_col], label='Actual Data', c='blue')
            self.imputation_plot = self.main_plot_canvas.axes.scatter(self.imputed_dataframe.index[self.na_marker], self.imputed_dataframe[variable_col][self.na_marker], label='Imputed Data', c='red')
            
            self.main_plot_canvas.axes.legend()
            self.main_plot_canvas.draw()

    def on_random_check_state_changed(self, state):
        if state == 2:
            self.n_sample_random_check.setEnabled(True)
        else:
            self.n_sample_random_check.setValue(0)
            self.n_sample_random_check.setEnabled(False)

    def on_n_sample_random_check_changed(self, value):
        if value > 0:
            variable_col = self.variable_combobox.currentText()
            random_check_dataframe, random_valid_indices = remove_random_data(self.dataframe.copy(), variable_col, self.n_sample_random_check.value())
            random_check_dataframe[variable_col] = data_imputation(random_check_dataframe, variable_col, method=self.imputation_method.currentText())

            actual = random_check_dataframe[variable_col].loc[random_valid_indices]
            predicted = self.dataframe[variable_col].loc[random_valid_indices]

            self.mse_random_check.setText(str(MSE(actual, predicted)))
            self.mae_random_check.setText(str(MAE(actual, predicted)))
            self.var_before_random_check.setText(str(self.dataframe[variable_col].var()))
            self.var_after_random_check.setText(str(random_check_dataframe[variable_col].var()))

            self.random_check_imputed_plot.remove()
            self.random_check_actual_plot.remove()

            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter(random_valid_indices, predicted, label='Imputed Check Data', c='orange')
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

    def handle_import_data(self):
        filter_file = "Semua File Data (*.csv *.xlsx *.xls *.json);;File CSV (*.csv);;File Excel (*.xlsx *.xls);;File JSON (*.json);;Semua File (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih File Data", "", filter_file)

        df, error_message = import_data(file_path)

        if error_message:
            QMessageBox.critical(self, "Error", error_message)
            self.dataframe = None
            self.variable_combobox.clear()
            self.summary_button.setEnabled(False)
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
            self.imputation_plot = self.main_plot_canvas.axes.scatter([], [])
            self.random_check_imputed_plot = self.main_plot_canvas.axes.scatter([], [])
            self.random_check_actual_plot = self.main_plot_canvas.axes.scatter([], [])
            
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

        self.train_data, self.validation_data, self.test_data = data_separation(self.imputed_dataframe, self.train_percentage.value() / 100, self.valid_percentage.value() / 100)

        self.main_plot_canvas.axes.cla()

        variable_col = self.variable_combobox.currentText()

        self.train_plot, = self.main_plot_canvas.axes.plot(self.train_data.index, self.train_data[variable_col], label="Train Data")
        self.validation_plot, = self.main_plot_canvas.axes.plot(self.validation_data.index, self.validation_data[variable_col], label="Validation Data")
        self.test_plot, = self.main_plot_canvas.axes.plot(self.test_data.index, self.test_data[variable_col], label="Test Data")

        self.main_plot_canvas.axes.set_xlabel(self.train_data.index.name)
        self.main_plot_canvas.axes.set_ylabel(variable_col)
        self.main_plot_canvas.axes.legend()
        self.main_plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.main_plot_canvas.figure.autofmt_xdate()
        self.main_plot_canvas.figure.tight_layout()
        
        # Segarkan kanvas untuk menampilkan plot baru
        self.main_plot_canvas.draw()
        