# app/widgets.py

from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableView

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
    Sebuah kelas jendela terpisah untuk menampilkan ringkasan DataFrame
    dalam format tabel.
    """
    def __init__(self, dataframe):
        super().__init__()
        
        self.setWindowTitle("Ringkasan Data Statistik")
        self.setGeometry(200, 200, 700, 400)
        
        layout = QVBoxLayout(self)
        table_view = QTableView()
        layout.addWidget(table_view)

        try:
            summary_df = dataframe.describe().round(4)
            model = PandasModel(summary_df)
            table_view.setModel(model)
            table_view.resizeColumnsToContents()
        except Exception as e:
            print(f"Error saat membuat tabel ringkasan: {e}")