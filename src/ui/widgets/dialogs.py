"""
Dialog widgets for MARS-SC (Solution Combination).

Contains dialog windows used throughout the application, including settings and result displays.
"""

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QGridLayout, QGroupBox, QLabel, QSpinBox, QTableView, QVBoxLayout
)

# Import constants module for dynamic access to runtime values
from utils import constants
from ui.styles.style_constants import DIALOG_STYLE, DIALOG_GROUP_BOX_STYLE


class AdvancedSettingsDialog(QDialog):
    """
    A dialog for configuring advanced global settings for the solver engine.
    
    This dialog allows the user to view the current solver settings and modify
    key performance parameters such as RAM allocation, numerical precision,
    and GPU acceleration.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setMinimumWidth(400)
        
        # Define fonts for different elements
        main_font = QFont()
        main_font.setPointSize(10)
        
        group_title_font = QFont()
        group_title_font.setPointSize(10)
        
        # Current values for reference
        global_settings_text = (
            f"Current settings:\n"
            f"- Precision: {constants.DEFAULT_PRECISION}\n"
            f"- RAM Limit: {constants.RAM_PERCENT * 100:.0f}%\n"
            f"- GPU Acceleration: {'Enabled' if constants.IS_GPU_ACCELERATION_ENABLED else 'Disabled'}"
        )
        self.current_settings_label = QLabel(global_settings_text)
        self.current_settings_label.setProperty("class", "currentSettingsLabel")
        
        # Create widgets for modification
        self.ram_label = QLabel("Set RAM Allocation (%):")
        self.ram_spinbox = QSpinBox()
        self.ram_spinbox.setRange(10, 95)
        self.ram_spinbox.setValue(int(constants.RAM_PERCENT * 100))
        self.ram_spinbox.setToolTip(
            "Set the maximum percentage of available RAM the solver can use. "
            "It will based on allowable free memory."
        )
        
        self.precision_label = QLabel("Set Solver Precision:")
        self.precision_combobox = QComboBox()
        self.precision_combobox.addItems(["Single", "Double"])
        self.precision_combobox.setCurrentText(constants.DEFAULT_PRECISION)
        self.precision_combobox.setToolTip(
            "Single precision is faster and uses less memory.\n"
            "Double precision is more accurate but slower."
        )
        
        self.gpu_checkbox = QCheckBox(
            "Enable GPU Acceleration (Only works if NVIDIA CUDA is installed in PC)"
        )
        self.gpu_checkbox.setChecked(constants.IS_GPU_ACCELERATION_ENABLED)
        self.gpu_checkbox.setToolTip(
            "Uses the GPU for matrix multiplication if a compatible NVIDIA GPU "
            "is found and CUDA is installed in the system."
        )
        
        # Apply font to the widgets
        self.ram_label.setFont(main_font)
        self.ram_spinbox.setFont(main_font)
        self.precision_label.setFont(main_font)
        self.precision_combobox.setFont(main_font)
        self.gpu_checkbox.setFont(main_font)
        
        # Layout
        layout = QGridLayout()
        layout.setSpacing(15)
        layout.addWidget(self.ram_label, 0, 0)
        layout.addWidget(self.ram_spinbox, 0, 1)
        layout.addWidget(self.precision_label, 1, 0)
        layout.addWidget(self.precision_combobox, 1, 1)
        layout.addWidget(self.gpu_checkbox, 2, 0, 1, 2)
        
        # GroupBox to hold the settings
        settings_group = QGroupBox("Modify Global Parameters")
        settings_group.setFont(group_title_font)
        settings_group.setLayout(layout)
        
        # OK and Cancel buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.setFont(main_font)
        
        # Main layout for the dialog
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.current_settings_label)
        main_layout.addWidget(settings_group)
        main_layout.addStretch()
        main_layout.addWidget(self.buttons)
        self.setLayout(main_layout)

        # Apply styles using direct CSS constants
        self.setStyleSheet(DIALOG_STYLE)
        settings_group.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
    
    def get_settings(self):
        """
        Returns the selected settings from the dialog widgets.
        
        Returns:
            dict: Dictionary containing 'ram_percent', 'precision', and 'gpu_acceleration'.
        """
        return {
            "ram_percent": self.ram_spinbox.value() / 100.0,
            "precision": self.precision_combobox.currentText(),
            "gpu_acceleration": self.gpu_checkbox.isChecked(),
        }


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
