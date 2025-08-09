# app/main.py

import sys
from PySide6.QtWidgets import QApplication

# Mengimpor kelas jendela utama dari file ui_main.py
from ui_main import UIMainWindow

if __name__ == "__main__":
    # 1. Membuat instance aplikasi
    # Setiap aplikasi PySide6 harus memiliki satu QApplication.
    app = QApplication(sys.argv)

    # 2. Membuat instance dari jendela utama kita
    # Ini akan memanggil __init__ di kelas UIMainWindow dan membangun seluruh GUI.
    window = UIMainWindow()

    # 3. Menampilkan jendela ke layar
    window.show()

    # 4. Memulai event loop dan memastikan aplikasi ditutup dengan bersih
    # app.exec() memulai loop yang mendengarkan event seperti klik tombol.
    # sys.exit() menangani proses keluar dari aplikasi.
    sys.exit(app.exec())