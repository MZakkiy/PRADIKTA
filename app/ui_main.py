# app/ui_main.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QGroupBox, QLabel, QPushButton, QCheckBox, 
    QProgressBar, QComboBox, QSpinBox, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

from analysis.data_processor import import_data

class UIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikasi Prediksi Time Series")
        self.setGeometry(100, 100, 1200, 800) # Ukuran awal jendela

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
        top_panel_layout.addWidget(main_tabs, 1) # Angka 1 memberikan stretch factor

        # --- Membuat dan Menambahkan Panel Bawah (Plot) ---
        plot_panel = self.create_plot_panel()

        # Menambahkan semua bagian ke layout utama
        main_layout.addLayout(top_panel_layout)
        main_layout.addWidget(plot_panel)
        
        self.create_menu_bar()

        self.dataframe = None

        self.tombol_import_data = self.findChild(QPushButton, "ImportButton")
        if self.tombol_import_data:
            self.tombol_import_data.clicked.connect(self.handle_import_data)

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

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
        layout.addStretch() # Menambahkan ruang fleksibel
        layout.addWidget(processing_label)
        layout.addLayout(progress_layout)
        layout.addWidget(self.status_indicator, alignment=Qt.AlignCenter)
        
        group_box.setLayout(layout)
        return group_box

    def create_main_tabs(self):
        tab_widget = QTabWidget()
        
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
        input_layout.addWidget(import_data_button)
        summary_button = QPushButton("Summary")
        summary_button.setObjectName("SummaryButton")
        input_layout.addWidget(summary_button)
        input_layout.addWidget(QLabel("Variable"))
        input_layout.addWidget(QComboBox())
        input_layout.addWidget(QLabel("NaN Data"))
        nan_data_line = QLineEdit("0")
        nan_data_line.setReadOnly(True)
        input_layout.addWidget(nan_data_line)
        input_layout.addStretch()
        input_group.setLayout(input_layout)

        # --- Kolom 2: Data Imputation ---
        imputation_group = QGroupBox("Data Imputation")
        imputation_layout = QGridLayout()
        imputation_layout.addWidget(QLabel("Method"), 0, 0, 1, 2)
        imputation_layout.addWidget(QComboBox(), 1, 0, 1, 2)
        imputation_layout.addWidget(QCheckBox("Random Check (%)"), 2, 0)
        imputation_layout.addWidget(QSpinBox(), 2, 1)
        # Menambahkan metrik
        metrics = {"MSE": "0.0e+00", "MAE": "0.0e+00", "Var Before": "0.0e+00", "Var After": "0.0e+00"}
        row = 3
        for label, value in metrics.items():
            imputation_layout.addWidget(QLabel(label), row, 0)
            line_edit = QLineEdit(value)
            line_edit.setReadOnly(True)
            imputation_layout.addWidget(line_edit, row, 1)
            row += 1
        imputation_group.setLayout(imputation_layout)
        
        # --- Kolom 3 & 4: Preprocessing & Data Separation ---
        right_column_layout = QVBoxLayout()
        
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout()
        preprocess_layout.addWidget(QCheckBox("Standardize"))
        preprocess_group.setLayout(preprocess_layout)
        
        separation_group = QGroupBox("Data Separation")
        separation_layout = QGridLayout()
        separation_layout.addWidget(QLabel("Train (%)"), 0, 0)
        separation_layout.addWidget(QSpinBox(), 0, 1)
        separation_layout.addWidget(QLineEdit("0"), 0, 2) # (#)
        separation_layout.addWidget(QLabel("Validation (%)"), 1, 0)
        separation_layout.addWidget(QSpinBox(), 1, 1)
        separation_layout.addWidget(QLineEdit("0"), 1, 2) # (#)
        separation_layout.addWidget(QLabel("Test (%)"), 2, 0)
        separation_layout.addWidget(QSpinBox(), 2, 1)
        separation_layout.addWidget(QLineEdit("0"), 2, 2) # (#)
        separation_layout.addWidget(QPushButton("Show"), 3, 0, 1, 3)
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
        plot_canvas = QLabel("Kanvas Plot Utama (Matplotlib akan diintegrasikan di sini)")
        plot_canvas.setAlignment(Qt.AlignCenter)
        plot_canvas.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        plot_canvas.setSizePolicy(
            plot_canvas.sizePolicy().horizontalPolicy(),
            plot_canvas.sizePolicy().verticalPolicy()
        )
        plot_canvas.setMinimumHeight(300)

        # Tombol di bawah plot
        show_button = QPushButton("Show Forecast")
        show_button.setEnabled(False) # Awalnya nonaktif seperti di gambar
        show_button.setFixedWidth(100)
        
        layout.addWidget(plot_canvas)
        layout.addWidget(show_button, alignment=Qt.AlignCenter)
        
        group_box.setLayout(layout)
        return group_box

    def handle_import_data(self):
        """
        Ini adalah SLOT. Fungsi ini akan menjadi 'pelayan' yang
        menangani event klik tombol.
        """
        # 1. Buka dialog untuk memilih file
        filter_file = "Semua File Data (*.csv *.xlsx *.xls *.json);;File CSV (*.csv);;File Excel (*.xlsx *.xls);;File JSON (*.json);;Semua File (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih File Data", "", filter_file)

        # 2. Panggil fungsi backend dari 'dapur' untuk memproses file
        df, error_message = import_data(file_path)

        # 3. Tampilkan hasilnya ke pengguna
        if error_message:
            # Tampilkan pesan error jika ada masalah
            QMessageBox.critical(self, "Error", error_message)
            self.dataframe = None
        else:
            # Simpan dataframe dan update UI
            self.dataframe = df
            QMessageBox.information(self, "Sukses", f"Data berhasil dimuat dengan {len(self.dataframe)} baris.")
            # Di sini Anda akan memperbarui tabel pratinjau, jumlah NaN, dll.
            print(self.dataframe.head())