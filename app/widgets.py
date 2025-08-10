# app/widgets.py

from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableView, QSplitter

import matplotlib
matplotlib.use('QtAgg') # Set backend Matplotlib agar kompatibel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np # Kita akan butuh numpy

# =======================================================
# KELAS KANVAS MATPLOTLIB (BARU)
# =======================================================
class MplCanvas(FigureCanvas):
    """Widget kanvas Matplotlib yang bisa diintegrasikan ke PySide6."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

# =======================================================
# KELAS MODEL UNTUK MENJEMBATANI PANDAS DAN QTABLEVIEW
# =======================================================
class PandasModel(QAbstractTableModel):
    """
    Kelas Model untuk mengubah Pandas DataFrame menjadi sumber data
    yang bisa dibaca oleh QTableView.
    """
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])
        return None

# =======================================================
# KELAS JENDELA KUSTOM UNTUK MENAMPILKAN RINGKASAN
# =======================================================

class SummaryWindow(QWidget):
    """
    Jendela terpisah yang sekarang menampilkan box plot DAN tabel ringkasan.
    """
    def __init__(self, dataframe):
        super().__init__()
        
        self.dataframe = dataframe
        self.setWindowTitle("Ringkasan Visual & Statistik")
        self.setGeometry(200, 200, 800, 700) # Perbesar ukuran jendela
        
        # Gunakan layout utama untuk menampung splitter
        main_layout = QVBoxLayout(self)
        
        # 1. Buat QSplitter untuk membagi area secara vertikal
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 2. Buat dan tambahkan kanvas plot ke splitter
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        splitter.addWidget(self.plot_canvas)
        
        # 3. Buat dan tambahkan tabel ke splitter
        self.table_view = QTableView()
        splitter.addWidget(self.table_view)
        
        # Atur ukuran awal splitter (misal: 60% plot, 40% tabel)
        splitter.setSizes([400, 300])
        
        # 4. Panggil fungsi untuk mengisi konten
        self.plot_boxplot()
        self.tampilkan_ringkasan_tabel()
        
    def plot_boxplot(self):
        """Membuat dan menampilkan box plot untuk kolom numerik."""
        try:
            # Pilih hanya kolom numerik untuk di-plot
            numeric_df = self.dataframe.select_dtypes(include=np.number)
            
            if numeric_df.empty:
                self.plot_canvas.axes.text(0.5, 0.5, 'Tidak ada data numerik untuk di-plot.', 
                                           horizontalalignment='center', verticalalignment='center')
                return

            # Bersihkan plot sebelumnya
            self.plot_canvas.axes.cla()
            
            # Buat box plot langsung dari DataFrame ke axes kanvas kita
            numeric_df.plot(kind='box', ax=self.plot_canvas.axes, patch_artist=True)
            
            self.plot_canvas.axes.set_title('Distribusi Data Numerik (Box Plot)')
            self.plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.plot_canvas.figure.tight_layout() # Merapikan layout plot
            self.plot_canvas.draw() # Menggambar ulang kanvas
            
        except Exception as e:
            print(f"Gagal membuat box plot: {e}")
            self.plot_canvas.axes.text(0.5, 0.5, f'Error: {e}', 
                                       horizontalalignment='center', verticalalignment='center')

    def tampilkan_ringkasan_tabel(self):
        """Menampilkan DataFrame ringkasan di QTableView."""
        # ... (fungsi ini tidak perlu diubah) ...
        try:
            summary_df = self.dataframe.describe().round(4)
            model = PandasModel(summary_df)
            self.table_view.setModel(model)
            self.table_view.resizeColumnsToContents()
        except Exception as e:
            print(f"Tidak dapat membuat tabel ringkasan: {e}")