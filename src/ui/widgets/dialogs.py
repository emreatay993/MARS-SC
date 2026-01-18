"""
Dialog widgets for MARS-SC (Solution Combination).

Contains dialog windows used throughout the application for result displays.
"""

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QLabel, QTableView, QVBoxLayout
)


class HotspotDialog(QDialog):
    """
    A dialog window to display and interact with hotspot analysis results.
    
    This dialog presents a table of nodes with the highest scalar values,
    allowing the user to click on a specific node to highlight it in the
    main 3D visualization window.
    """
    
    # Signal to be emitted when a node is selected from the table
    node_selected = pyqtSignal(int)
    
    def __init__(self, hotspot_df, parent=None):
        """
        Initialize the hotspot dialog.
        
        Args:
            hotspot_df: Pandas DataFrame containing hotspot analysis results.
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        self.setWindowTitle("Hotspot Analysis Results")
        self.setMinimumSize(300, 300)
        
        self.table_view = QTableView()
        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)
        
        # Make the table non-editable and select whole rows at a time
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Populate the table with the data
        self._populate_table(hotspot_df)
        
        # When a row is clicked, trigger our handler
        self.table_view.clicked.connect(self._on_row_clicked)
        
        layout = QVBoxLayout()
        layout.addWidget(
            QLabel("Click a row to navigate to the node in the Display tab.")
        )
        layout.addWidget(self.table_view)
        self.setLayout(layout)
    
    def _populate_table(self, df):
        """
        Populates the table, formatting floats to 4 decimal places.
        
        Args:
            df: Pandas DataFrame with hotspot data.
        """
        self.model.setHorizontalHeaderLabels(df.columns)
        
        for index, row in df.iterrows():
            items = []
            for col_name, val in row.items():
                # Keep Rank and NodeID as integers
                if col_name in ['Rank', 'NodeID']:
                    items.append(QStandardItem(str(int(float(val)))))
                # Format all other columns as floats with 4 decimal places
                else:
                    items.append(QStandardItem(f"{val:.4f}"))
            self.model.appendRow(items)
        
        self.table_view.resizeColumnsToContents()
    
    @pyqtSlot('QModelIndex')
    def _on_row_clicked(self, index):
        """
        Handle row click event - emit signal with selected node ID.
        
        Args:
            index: QModelIndex of the clicked cell.
        """
        # Get the row of the clicked cell
        row = index.row()
        # Assume 'NodeID' is the second column (index 1)
        node_id_item = self.model.item(row, 1)
        if node_id_item:
            node_id = int(float(node_id_item.text()))
            # Emit the signal with the node ID
            self.node_selected.emit(node_id)
