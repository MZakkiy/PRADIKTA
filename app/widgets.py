# app/widgets.py

from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableView, QSplitter, QPushButton, QHBoxLayout, QFileDialog, QMessageBox

import matplotlib
matplotlib.use('QtAgg') # Set backend Matplotlib agar kompatibel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np 
import pandas as pd 

class MplCanvas(FigureCanvas):
    """Widget kanvas Matplotlib yang bisa diintegrasikan ke PySide6."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, nrows=1, ncols=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.nrows = nrows
        self.ncols = ncols
        
        if nrows == 1 and ncols == 1:
            self.axes = fig.add_subplot(111)
        else:
            # Untuk multiple subplots, buat grid of axes
            self.axes = fig.subplots(nrows, ncols)
            
        self.figure = fig
        super(MplCanvas, self).__init__(fig)

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

class SummaryWindow(QWidget):
    """
    Jendela terpisah yang menampilkan box plot (setiap kolom punya y-axis sendiri) DAN tabel ringkasan.
    """
    def __init__(self, dataframe):
        super().__init__()
        
        self.dataframe = dataframe
        self.setWindowTitle("Visual & Statistical Summary")
        
        # Hitung jumlah kolom numerik untuk menentukan ukuran canvas
        numeric_df = self.dataframe.select_dtypes(include=np.number)
        num_cols = len(numeric_df.columns) if not numeric_df.empty else 1
        
        # Tentukan ukuran window yang sesuai
        canvas_width = max(10, num_cols * 3)  # 3 inch per kolom
        self.setGeometry(200, 200, max(800, num_cols * 200 + 100), 700)
        
        # Gunakan layout utama untuk menampung splitter
        main_layout = QVBoxLayout(self)
        
        # 1. Buat QSplitter untuk membagi area secara vertikal
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 2. Buat dan tambahkan kanvas plot ke splitter dengan jumlah subplot yang tepat
        self.plot_canvas = MplCanvas(self, width=canvas_width, height=5, dpi=100, nrows=1, ncols=num_cols)
        splitter.addWidget(self.plot_canvas)
        
        # 3. Buat dan tambahkan tabel ke splitter
        self.table_view = QTableView()
        splitter.addWidget(self.table_view)
        
        # Atur ukuran awal splitter (misal: 60% plot, 40% tabel)
        splitter.setSizes([400, 300])
        
        # 4. Buat layout untuk tombol download
        button_layout = QHBoxLayout()
        self.download_boxplot_button = QPushButton("Download Boxplot")
        self.download_boxplot_button.clicked.connect(self.handle_download_boxplot)
        button_layout.addWidget(self.download_boxplot_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # 5. Panggil fungsi untuk mengisi konten
        self.plot_boxplot()
        self.show_summary_table()
        
    def plot_boxplot(self):
        """Membuat dan menampilkan box plot untuk setiap kolom numerik dengan y-axis terpisah."""
        try:
            # Pilih hanya kolom numerik untuk di-plot
            numeric_df = self.dataframe.select_dtypes(include=np.number)
            
            if numeric_df.empty:
                if self.plot_canvas.nrows == 1 and self.plot_canvas.ncols == 1:
                    self.plot_canvas.axes.text(0.5, 0.5, 'There is no numeric data to plot..', 
                                               horizontalalignment='center', verticalalignment='center')
                return

            # Get axes array
            if self.plot_canvas.nrows == 1 and self.plot_canvas.ncols == 1:
                axes_list = [self.plot_canvas.axes]
            else:
                axes_list = self.plot_canvas.axes.flatten() if isinstance(self.plot_canvas.axes, np.ndarray) else [self.plot_canvas.axes]
            
            # Buat box plot untuk setiap kolom pada axes terpisah
            for idx, column in enumerate(numeric_df.columns):
                if idx < len(axes_list):
                    ax = axes_list[idx]
                    ax.axvline  # Clear any previous content
                    
                    # Membuat boxplot untuk single column
                    bp = ax.boxplot([numeric_df[column].dropna()], vert=True, patch_artist=True)
                    
                    # Styling
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.7)
                    
                    ax.set_title(f'{column}', fontsize=11, fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
                    ax.set_xticklabels([''])  # Remove x-axis label since it's just one box
            
            # Hide any unused subplots
            for idx in range(len(numeric_df.columns), len(axes_list)):
                axes_list[idx].axis('off')
            
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.draw()
            
        except Exception as e:
            print(f"Gagal membuat box plot: {e}")
            if self.plot_canvas.nrows == 1 and self.plot_canvas.ncols == 1:
                self.plot_canvas.axes.text(0.5, 0.5, f'Error: {e}', 
                                           horizontalalignment='center', verticalalignment='center')

    def show_summary_table(self):
        """Menampilkan DataFrame ringkasan di QTableView."""
        # ... (fungsi ini tidak perlu diubah) ...
        try:
            summary_df = self.dataframe.describe().round(4)
            model = PandasModel(summary_df)
            self.table_view.setModel(model)
            self.table_view.resizeColumnsToContents()
        except Exception as e:
            print(f"Unable to create summary table: {e}")
    
    def handle_download_boxplot(self):
        """Save the boxplot as an image file."""
        try:
            # Open file dialog to choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Boxplot",
                "",
                "PNG Images (*.png);;JPEG Images (*.jpg);;PDF Documents (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Save the figure
                self.plot_canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Boxplot saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save boxplot: {e}")
            print(f"Error saving boxplot: {e}")

class LossFunctionWindow(QWidget):
    """
    Jendela terpisah yang menampilkan loss function plot dengan fitur download.
    """
    def __init__(self, history):
        super().__init__()
        
        self.history = history
        self.setWindowTitle("Loss Function")
        self.setGeometry(200, 200, 500, 500) # Perbesar ukuran jendela
        
        # Gunakan layout utama untuk menampung splitter
        main_layout = QVBoxLayout(self)
        
        # 2. Buat dan tambahkan kanvas plot ke splitter
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        main_layout.addWidget(self.plot_canvas)
        
        # 3. Buat layout untuk tombol
        button_layout = QHBoxLayout()
        self.download_loss_button = QPushButton("Download Loss Plot")
        self.download_loss_button.clicked.connect(self.handle_download_loss_plot)
        button_layout.addWidget(self.download_loss_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # 4. Panggil fungsi untuk mengisi konten
        self.plot_loss_function()
        
    def plot_loss_function(self):
        try:
            # Bersihkan plot sebelumnya
            self.plot_canvas.axes.cla()
            
            self.plot_canvas.axes.plot(self.history.history['loss'], label='Training Loss')
            self.plot_canvas.axes.plot(self.history.history['val_loss'], label='Validation Loss')
            self.plot_canvas.axes.set_xlabel('Epochs')
            self.plot_canvas.axes.set_ylabel('Loss')
            self.plot_canvas.axes.set_title('Loss Function Plot')
            self.plot_canvas.axes.legend()
            self.plot_canvas.axes.grid(True, linestyle='--', alpha=0.6)
            self.plot_canvas.figure.tight_layout() # Merapikan layout plot
            self.plot_canvas.draw() # Menggambar ulang kanvas
            
        except Exception as e:
            print(f"Failed to create loss function plot: {e}")
            self.plot_canvas.axes.text(0.5, 0.5, f'Error: {e}', 
                                       horizontalalignment='center', verticalalignment='center')
    
    def handle_download_loss_plot(self):
        """Save the loss function plot as an image file."""
        try:
            # Open file dialog to choose save location
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Loss Function Plot",
                "",
                "PNG Images (*.png);;JPEG Images (*.jpg);;PDF Documents (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Save the figure
                self.plot_canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Loss function plot saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save loss function plot: {e}")
            print(f"Error saving loss function plot: {e}")