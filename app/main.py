# app/main.py

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

# Mengimpor kelas jendela utama dari file ui_main.py
from ui_main import UIMainWindow

def apply_light_theme(app):
    """Apply light theme to the application"""
    
    # Light stylesheet
    light_stylesheet = """
    QWidget {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    QMainWindow {
        background-color: #FFFFFF;
    }
    
    QGroupBox {
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        color: #000000;
        font-weight: bold;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 3px 5px;
    }
    
    QPushButton {
        background-color: #F0F0F0;
        color: #000000;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        padding: 5px;
        font-weight: bold;
    }
    
    QPushButton:hover {
        background-color: #E0E0E0;
        border: 1px solid #A0A0A0;
    }
    
    QPushButton:pressed {
        background-color: #D0D0D0;
    }
    
    QPushButton:disabled {
        background-color: #F5F5F5;
        color: #A0A0A0;
    }
    
    QLineEdit {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        padding: 4px;
    }
    
    QLineEdit:focus {
        border: 2px solid #4CAF50;
    }
    
    QLineEdit:read-only {
        background-color: #F5F5F5;
        color: #666666;
    }
    
    QComboBox {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        padding: 4px;
    }
    
    QComboBox:focus {
        border: 2px solid #4CAF50;
    }
    
    QComboBox::drop-down {
        border-left: 1px solid #D0D0D0;
    }
    
    QComboBox QAbstractItemView {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #D0D0D0;
        selection-background-color: #4CAF50;
    }
    
    QSpinBox, QDoubleSpinBox {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        padding: 4px;
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 2px solid #4CAF50;
    }
    
    QCheckBox {
        color: #000000;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    
    QCheckBox::indicator:unchecked {
        background-color: #FFFFFF;
        border: 1px solid #D0D0D0;
    }
    
    QCheckBox::indicator:checked {
        background-color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    
    QProgressBar {
        background-color: #F0F0F0;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        text-align: center;
        color: #000000;
    }
    
    QProgressBar::chunk {
        background-color: #4CAF50;
    }
    
    QTabWidget::pane {
        border: 1px solid #D0D0D0;
    }
    
    QTabBar::tab {
        background-color: #F5F5F5;
        color: #000000;
        padding: 8px 20px;
        border: 1px solid #D0D0D0;
        border-bottom: 2px solid #D0D0D0;
    }
    
    QTabBar::tab:selected {
        background-color: #FFFFFF;
        border-bottom: 3px solid #4CAF50;
        color: #000000;
    }
    
    QTabBar::tab:hover {
        background-color: #F0F0F0;
    }
    
    QTableView {
        background-color: #FFFFFF;
        alternate-background-color: #F5F5F5;
        color: #000000;
        border: 1px solid #D0D0D0;
    }
    
    QHeaderView::section {
        background-color: #F0F0F0;
        color: #000000;
        padding: 5px;
        border: 1px solid #D0D0D0;
    }
    
    QScrollBar:vertical {
        background-color: #F5F5F5;
        border: 1px solid #D0D0D0;
    }
    
    QScrollBar::handle:vertical {
        background-color: #BDBDBD;
        border-radius: 4px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #9E9E9E;
    }
    
    QScrollBar:horizontal {
        background-color: #F5F5F5;
        border: 1px solid #D0D0D0;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #BDBDBD;
        border-radius: 4px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #9E9E9E;
    }
    """
    
    app.setStyle('Fusion')
    app.setStyleSheet(light_stylesheet)
    
    # Set light palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(palette)

if __name__ == "__main__":
    # 1. Membuat instance aplikasi
    # Setiap aplikasi PySide6 harus memiliki satu QApplication.
    app = QApplication(sys.argv)

    # 2. Menerapkan tema terang
    apply_light_theme(app)

    # 3. Membuat instance dari jendela utama kita
    # Ini akan memanggil __init__ di kelas UIMainWindow dan membangun seluruh GUI.
    window = UIMainWindow()

    # 4. Menampilkan jendela ke layar
    window.show()

    # 5. Memulai event loop dan memastikan aplikasi ditutup dengan bersih
    # app.exec() memulai loop yang mendengarkan event seperti klik tombol.
    # sys.exit() menangani proses keluar dari aplikasi.
    sys.exit(app.exec())