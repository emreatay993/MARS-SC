"""
Plotting widgets for MARS-SC (Solution Combination).

Contains matplotlib- and plotly-based widgets for displaying analysis results.

Key widgets:
- MatplotlibWidget: Time history / combination history plots with data table
- PlotlyWidget: Modal coordinate plots (legacy, kept for compatibility)
- PlotlyMaxWidget: Envelope plots with data table (max/min over combinations)
"""

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QKeySequence, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QShortcut, QSizePolicy,
    QSplitter, QTableView, QVBoxLayout, QWidget, QHeaderView
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import MaxNLocator

import plotly.graph_objects as go


class MatplotlibWidget(QWidget):
    """
    Widget for displaying matplotlib plots with an interactive data table.
    
    Features:
    - Matplotlib canvas with navigation toolbar
    - Interactive legend (click to hide/show traces)
    - Hover annotations for data points
    - Side-by-side data table
    - Clipboard copy support (Ctrl+C)
    - Auto-resizing splitter
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Attributes for interactivity
        self.ax = None
        self.annot = None
        self.plotted_lines = []
        self.legend_map = {}  # Used to map legend items to plot lines
        
        # Matplotlib canvas on the left
        self.figure = plt.Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # Add the Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Data table on the right
        self.table = QTableView(self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Time [s]", "Value"])
        self.table.setModel(self.model)
        
        # Ctrl+C to copy the selected block
        copy_sc = QShortcut(QKeySequence.Copy, self.table)
        copy_sc.activated.connect(self.copy_selection)
        
        # Split view
        self.splitter = QSplitter(Qt.Horizontal, self)
        
        # Create a container for plot and toolbar
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        self.splitter.addWidget(plot_container)
        self.splitter.addWidget(self.table)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        self.setLayout(layout)
        
        # Connect events
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.canvas.mpl_connect('pick_event', self.on_legend_pick)
    
    def showEvent(self, event):
        """Called when the widget is shown."""
        super().showEvent(event)
        QTimer.singleShot(50, self.adjust_splitter_size)
    
    def resizeEvent(self, event):
        """Called when the widget is resized."""
        super().resizeEvent(event)
        self.adjust_splitter_size()
    
    def adjust_splitter_size(self):
        """Calculates and sets optimal table width."""
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        v_scrollbar = self.table.verticalScrollBar()
        scrollbar_width = v_scrollbar.width() if v_scrollbar.isVisible() else 0
        
        required_width = (header.length() +
                         self.table.verticalHeader().width() +
                         self.table.frameWidth() * 2 +
                         scrollbar_width)
        
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        total_width = self.splitter.width()
        plot_width = max(450, total_width - required_width)
        new_table_width = total_width - plot_width
        
        self.splitter.setSizes([int(plot_width), int(new_table_width)])
    
    def hover(self, event):
        """Show annotation when hovering over a data point."""
        if not event.inaxes or self.ax is None or self.annot is None:
            return
        
        visible = self.annot.get_visible()
        
        for line in self.plotted_lines:
            cont, ind = line.contains(event)
            if cont:
                pos = line.get_xydata()[ind["ind"][0]]
                x_coord, y_coord = pos[0], pos[1]
                
                self.annot.xy = (x_coord, y_coord)
                self.annot.set_text(f"Time: {x_coord:.4f}\\nValue: {y_coord:.4f}")
                
                if not visible:
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                return
        
        if visible:
            self.annot.set_visible(False)
            self.canvas.draw_idle()
    
    def on_legend_pick(self, event):
        """Toggle line visibility when legend is clicked."""
        artist = event.artist
        if artist in self.legend_map:
            original_line = self.legend_map[artist]
            is_visible = not original_line.get_visible()
            original_line.set_visible(is_visible)
            
            for leg_artist, line in self.legend_map.items():
                if line == original_line:
                    leg_artist.set_alpha(1.0 if is_visible else 0.2)
            
            # Rescale Y-axis to fit visible data
            min_y, max_y = np.inf, -np.inf
            any_line_visible = False
            for line in self.plotted_lines:
                if line.get_visible():
                    any_line_visible = True
                    y_data = line.get_ydata()
                    if len(y_data) > 0:
                        min_y = min(min_y, np.min(y_data))
                        max_y = max(max_y, np.max(y_data))
            
            if any_line_visible and self.ax:
                if np.isclose(min_y, max_y):
                    margin = 1.0
                    self.ax.set_ylim(min_y - margin, max_y + margin)
                else:
                    margin = (max_y - min_y) * 0.05
                    self.ax.set_ylim(min_y - margin, max_y + margin)
            
            self.canvas.draw()
    
    def update_plot(self, x, y, node_id=None,
                   is_max_principal_stress=False,
                   is_min_principal_stress=False,
                   is_von_mises=False,
                   is_deformation=False,
                   is_velocity=False,
                   is_acceleration=False,
                   plasticity_overlay=None):
        """Update the plot and table with new data."""
        # Reset state
        self.figure.clear()
        self.plotted_lines.clear()
        self.legend_map.clear()
        
        self.ax = self.figure.add_subplot(1, 1, 1)
        
        # Define plotting styles
        styles = {
            'Magnitude': {'color': 'black', 'linestyle': '-', 'linewidth': 2},
            'X': {'color': 'red', 'linestyle': '--', 'linewidth': 1},
            'Y': {'color': 'green', 'linestyle': '--', 'linewidth': 1},
            'Z': {'color': 'blue', 'linestyle': '--', 'linewidth': 1},
        }
        
        self.model.clear()
        textstr = ""
        
        # Handle dict data (multi-component)
        if isinstance(y, dict):
            if is_velocity:
                prefix, units = "Velocity", "(mm/s)"
            elif is_acceleration:
                prefix, units = "Acceleration", "(mm/s²)"
            else:
                prefix, units = "Deformation", "(mm)"
            
            self.ax.set_title(f"{prefix} (Node ID: {node_id})", fontsize=8)
            self.ax.set_ylabel(f"{prefix} {units}", fontsize=8)
            self.model.setHorizontalHeaderLabels(
                ["Time [s]", f"Mag {units}", f"X {units}", f"Y {units}", f"Z {units}"])
            
            for component, data in y.items():
                style = styles.get(component, {})
                line, = self.ax.plot(x, data, label=f'{prefix} ({component})', **style)
                self.plotted_lines.append(line)
            
            for i in range(len(x)):
                items = [
                    QStandardItem(f"{x[i]:.5f}"),
                    QStandardItem(f"{y['Magnitude'][i]:.5f}"),
                    QStandardItem(f"{y['X'][i]:.5f}"),
                    QStandardItem(f"{y['Y'][i]:.5f}"),
                    QStandardItem(f"{y['Z'][i]:.5f}")
                ]
                self.model.appendRow(items)
            
            max_y_value = np.max(y['Magnitude'])
            time_of_max = x[np.argmax(y['Magnitude'])]
            textstr = f'Max Magnitude: {max_y_value:.4f}\nTime of Max: {time_of_max:.5f} s'
        
        # Handle array data (single component)
        else:
            if is_min_principal_stress:
                self.model.setHorizontalHeaderLabels(["Time [s]", "σ3 [MPa]"])
                for xi, yi in zip(x, y):
                    self.model.appendRow([QStandardItem(f"{xi:.5f}"), QStandardItem(f"{yi:.5f}")])

                line, = self.ax.plot(x, y, label=r'$\sigma_3$', color='green')
                self.plotted_lines.append(line)
                self.ax.set_title(f"Min Principal Stress (Node ID: {node_id})" if node_id else "Min Principal Stress", fontsize=8)
                self.ax.set_ylabel(r'$\sigma_3$ [MPa]', fontsize=8)
                min_y_value = np.min(y)
                time_of_min = x[np.argmin(y)]
                textstr = f'Min Magnitude: {min_y_value:.4f}\nTime of Min: {time_of_min:.5f} s'
            else:
                # For plot labels (LaTeX renders)
                title, plot_label, color = "Stress", "Value", 'blue'
                # For table headers (plain text)
                table_label = "Value"
                
                if is_max_principal_stress:
                    title, plot_label, color = "Max Principal Stress", r'$\sigma_1$', 'red'
                    table_label = "σ1"
                elif is_von_mises:
                    title, plot_label, color = "Von Mises Stress", r'$\sigma_{VM}$', 'blue'
                    table_label = "σ_VM"
                
                headers = ["Time [s]", f"{table_label} [MPa]"]
                if plasticity_overlay and 'corrected_vm' in plasticity_overlay:
                    corrected = np.asarray(plasticity_overlay['corrected_vm'], dtype=float)
                    strain = np.asarray(plasticity_overlay.get('plastic_strain', []), dtype=float)

                    elastic_line, = self.ax.plot(x, y, label=f"{plot_label} (Elastic)", color=color)
                    corrected_line, = self.ax.plot(x, corrected, label=f"{plot_label} (Corrected)", color='orange')
                    self.plotted_lines.extend([elastic_line, corrected_line])

                    headers.append("Corrected [MPa]")
                    if strain.size == corrected.size:
                        headers.append("Plastic Strain")

                    self.model.setHorizontalHeaderLabels(headers)
                    for idx, xi in enumerate(x):
                        row_items = [
                            QStandardItem(f"{xi:.5f}"),
                            QStandardItem(f"{y[idx]:.5f}"),
                            QStandardItem(f"{corrected[idx]:.5f}")
                        ]
                        if strain.size == corrected.size:
                            row_items.append(QStandardItem(f"{strain[idx]:.6e}"))
                        self.model.appendRow(row_items)

                    if corrected.size > 0 and np.any(np.isfinite(corrected)):
                        max_corr = np.nanmax(corrected)
                        time_of_max = x[np.nanargmax(corrected)]
                        textstr = f'Max Corrected: {max_corr:.4f}\nTime of Max: {time_of_max:.5f} s'
                else:
                    line, = self.ax.plot(x, y, label=plot_label, color=color)
                    self.plotted_lines.append(line)
                    self.model.setHorizontalHeaderLabels(headers)
                    for xi, yi in zip(x, y):
                        self.model.appendRow([QStandardItem(f"{xi:.5f}"), QStandardItem(f"{yi:.5f}")])

                    if len(y) > 0 and np.any(y):
                        max_y_value = np.max(y)
                        time_of_max = x[np.argmax(y)]
                        textstr = f'Max Magnitude: {max_y_value:.4f}\nTime of Max: {time_of_max:.5f} s'

                self.ax.set_title(f"{title} (Node ID: {node_id})" if node_id else title, fontsize=8)
                self.ax.set_ylabel(f'{plot_label} [MPa]', fontsize=8)

                # Optional diagnostics overlay (Δεp, εp) on secondary axis
                if plasticity_overlay and plasticity_overlay.get('show_diagnostics'):
                    eps = np.asarray(plasticity_overlay.get('plastic_strain', []), dtype=float)
                    deps = np.asarray(plasticity_overlay.get('delta_plastic_strain', []), dtype=float)
                    if eps.size == len(x) or deps.size == len(x):
                        ax2 = self.ax.twinx()
                        ax2.grid(False)
                        if eps.size == len(x):
                            line_eps, = ax2.plot(x, eps, label='εp (cumulative)', color='purple', linestyle='--', linewidth=1)
                            self.plotted_lines.append(line_eps)
                        if deps.size == len(x):
                            line_deps, = ax2.plot(x, deps, label='Δεp (per step)', color='brown', linestyle=':', linewidth=1)
                            self.plotted_lines.append(line_deps)
                        ax2.set_ylabel('Plastic Strain', fontsize=8)
        
        # Apply common styling
        self.ax.set_xlabel('Time [seconds]', fontsize=8)
        self.ax.set_xlim(np.min(x), np.max(x))
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        self.ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Create interactive legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            leg = self.ax.legend(handles, labels, fontsize=7, loc='upper right')
            for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), self.plotted_lines):
                legline.set_picker(True)
                legline.set_pickradius(5)
                self.legend_map[legline] = origline
                legtext.set_picker(True)
                self.legend_map[legtext] = origline
        
        # Add annotation
        if textstr:
            self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Finalize
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)

    def update_plot_rotational_acceleration(self, time_values, alpha_x, alpha_y, alpha_z, alpha_mag, ref_point=None):
        """
        Update plot with rotational acceleration data (4 components).
        
        Args:
            time_values: Array of time points
            alpha_x: Angular acceleration about X axis (rad/s²)
            alpha_y: Angular acceleration about Y axis (rad/s²)
            alpha_z: Angular acceleration about Z axis (rad/s²)
            alpha_mag: Magnitude of angular acceleration (rad/s²)
            ref_point: Reference point tuple (x, y, z)
        """
        # Reset state
        self.figure.clear()
        self.plotted_lines.clear()
        self.legend_map.clear()
        
        self.ax = self.figure.add_subplot(1, 1, 1)
        
        x = np.asarray(time_values)
        ax_data = np.asarray(alpha_x)
        ay_data = np.asarray(alpha_y)
        az_data = np.asarray(alpha_z)
        mag_data = np.asarray(alpha_mag)
        
        # Plot all 4 traces
        line_mag, = self.ax.plot(x, mag_data, label='|α|', color='black', linewidth=2)
        line_x, = self.ax.plot(x, ax_data, label='αx', color='red', linestyle='--', linewidth=1)
        line_y, = self.ax.plot(x, ay_data, label='αy', color='green', linestyle='--', linewidth=1)
        line_z, = self.ax.plot(x, az_data, label='αz', color='blue', linestyle='--', linewidth=1)
        
        self.plotted_lines = [line_mag, line_x, line_y, line_z]
        
        # Title and labels
        ref_str = f"Ref: ({ref_point[0]:.2f}, {ref_point[1]:.2f}, {ref_point[2]:.2f})" if ref_point else ""
        self.ax.set_title(f"Rotational Acceleration (Rigid Body) {ref_str}", fontsize=8)
        self.ax.set_xlabel('Time [seconds]', fontsize=8)
        self.ax.set_ylabel('Angular Acceleration [rad/s²]', fontsize=8)
        self.ax.set_xlim(np.min(x), np.max(x))
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        self.ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Interactive legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            leg = self.ax.legend(handles, labels, fontsize=7, loc='upper right')
            for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), self.plotted_lines):
                legline.set_picker(True)
                legline.set_pickradius(5)
                self.legend_map[legline] = origline
                legtext.set_picker(True)
                self.legend_map[legtext] = origline
        
        # Max annotation
        max_mag = np.max(mag_data)
        time_of_max = x[np.argmax(mag_data)]
        textstr = f'Max |α|: {max_mag:.4e} rad/s²\nTime of Max: {time_of_max:.5f} s'
        self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Annotation setup
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        
        # Populate table
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Time [s]", "|α| [rad/s²]", "αx [rad/s²]", "αy [rad/s²]", "αz [rad/s²]"])
        for i in range(len(x)):
            items = [
                QStandardItem(f"{x[i]:.5f}"),
                QStandardItem(f"{mag_data[i]:.6e}"),
                QStandardItem(f"{ax_data[i]:.6e}"),
                QStandardItem(f"{ay_data[i]:.6e}"),
                QStandardItem(f"{az_data[i]:.6e}")
            ]
            self.model.appendRow(items)
        
        self.table.resizeColumnsToContents()
        self.canvas.draw()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    def clear_plot(self):
        """Clear the plot and table."""
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.set_title("Time History (No Data)", fontsize=8)
        ax.set_xlabel('Time [seconds]', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=8)
        self.canvas.draw()
        
        self.model.removeRows(0, self.model.rowCount())
        self.model.setHorizontalHeaderLabels(["Time [s]", "Value"])
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    def update_combination_history_plot(self, combination_indices, stress_values, node_id=None,
                                        combination_names=None, stress_type="von_mises",
                                        plasticity_overlay=None):
        """
        Update the plot with combination history data (stress vs combination number).
        
        This is the MARS-SC specific version that shows stress values across combinations
        for a single node.
        
        Args:
            combination_indices: Array of combination indices (0, 1, 2, ...)
            stress_values: Array of stress values at each combination
            node_id: Optional node ID for the title
            combination_names: Optional list of combination names for x-axis labels
            stress_type: Type of stress ("von_mises", "max_principal", "min_principal")
            plasticity_overlay: Optional dict with corrected values
        """
        # Reset state
        self.figure.clear()
        self.plotted_lines.clear()
        self.legend_map.clear()
        
        self.ax = self.figure.add_subplot(1, 1, 1)
        
        # Convert to numpy arrays
        x = np.asarray(combination_indices)
        y = np.asarray(stress_values)
        
        # Determine styling based on stress type
        stress_config = {
            'von_mises': {'title': 'Von Mises Stress', 'label': r'$\sigma_{VM}$', 
                        'table_label': 'σ_VM', 'color': 'blue'},
            'max_principal': {'title': 'Max Principal Stress', 'label': r'$\sigma_1$', 
                             'table_label': 'σ1', 'color': 'red'},
            'min_principal': {'title': 'Min Principal Stress', 'label': r'$\sigma_3$', 
                             'table_label': 'σ3', 'color': 'green'},
        }
        config = stress_config.get(stress_type, stress_config['von_mises'])
        
        # Clear and setup table
        self.model.clear()
        
        # Plot main line
        if plasticity_overlay and 'corrected_vm' in plasticity_overlay:
            corrected = np.asarray(plasticity_overlay['corrected_vm'], dtype=float)
            strain = np.asarray(plasticity_overlay.get('plastic_strain', []), dtype=float)
            
            elastic_line, = self.ax.plot(x, y, label=f"{config['label']} (Elastic)", 
                                         color=config['color'], marker='o', markersize=4)
            corrected_line, = self.ax.plot(x, corrected, label=f"{config['label']} (Corrected)", 
                                           color='orange', marker='s', markersize=4)
            self.plotted_lines.extend([elastic_line, corrected_line])
            
            headers = ["Combo #", "Name", f"{config['table_label']} [MPa]", "Corrected [MPa]"]
            if strain.size == corrected.size:
                headers.append("Plastic Strain")
            
            self.model.setHorizontalHeaderLabels(headers)
            for idx in range(len(x)):
                name = combination_names[idx] if combination_names and idx < len(combination_names) else ""
                row_items = [
                    # Display 1-based combination number for user-friendliness
                    QStandardItem(f"{int(x[idx]) + 1}"),
                    QStandardItem(name),
                    QStandardItem(f"{y[idx]:.5f}"),
                    QStandardItem(f"{corrected[idx]:.5f}")
                ]
                if strain.size == corrected.size:
                    row_items.append(QStandardItem(f"{strain[idx]:.6e}"))
                self.model.appendRow(row_items)
            
            if corrected.size > 0 and np.any(np.isfinite(corrected)):
                max_corr = np.nanmax(corrected)
                combo_of_max = x[np.nanargmax(corrected)]
                # Display 1-based combination number for user-friendliness
                textstr = f'Max Corrected: {max_corr:.4f} MPa\nAt Combination: {int(combo_of_max) + 1}'
        else:
            line, = self.ax.plot(x, y, label=config['label'], color=config['color'], 
                                marker='o', markersize=4)
            self.plotted_lines.append(line)
            
            headers = ["Combo #", "Name", f"{config['table_label']} [MPa]"]
            self.model.setHorizontalHeaderLabels(headers)
            for idx in range(len(x)):
                name = combination_names[idx] if combination_names and idx < len(combination_names) else ""
                self.model.appendRow([
                    # Display 1-based combination number for user-friendliness
                    QStandardItem(f"{int(x[idx]) + 1}"),
                    QStandardItem(name),
                    QStandardItem(f"{y[idx]:.5f}")
                ])
            
            if len(y) > 0:
                max_y = np.max(y)
                combo_of_max = x[np.argmax(y)]
                # Display 1-based combination number for user-friendliness
                textstr = f'Max: {max_y:.4f} MPa\nAt Combination: {int(combo_of_max) + 1}'
            else:
                textstr = ""
        
        # Styling
        title = f"{config['title']} vs Combination"
        if node_id is not None:
            title = f"{config['title']} (Node ID: {node_id})"
        self.ax.set_title(title, fontsize=8)
        self.ax.set_xlabel('Combination #', fontsize=8)
        self.ax.set_ylabel(f'{config["label"]} [MPa]', fontsize=8)
        self.ax.set_xlim(np.min(x) - 0.5, np.max(x) + 0.5)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        self.ax.grid(True, which='both', linestyle='-', linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Interactive legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            leg = self.ax.legend(handles, labels, fontsize=7, loc='upper right')
            for legline, legtext, origline in zip(leg.get_lines(), leg.get_texts(), self.plotted_lines):
                legline.set_picker(True)
                legline.set_pickradius(5)
                self.legend_map[legline] = origline
                legtext.set_picker(True)
                self.legend_map[legtext] = origline
        
        # Annotation box
        if textstr:
            self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Hover annotation
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        
        self.table.resizeColumnsToContents()
        self.canvas.draw()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    @pyqtSlot()
    def copy_selection(self):
        """Copy selected cells to clipboard as TSV."""
        sel = self.table.selectedIndexes()
        if not sel:
            return
        
        rows = sorted(idx.row() for idx in sel)
        cols = sorted(idx.column() for idx in sel)
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        
        headers = [self.model.headerData(c, Qt.Horizontal) for c in range(c0, c1 + 1)]
        lines = ['\\t'.join(headers)]
        
        for r in range(r0, r1 + 1):
            row_data = []
            for c in range(c0, c1 + 1):
                text = self.model.index(r, c).data() or ""
                row_data.append(text)
            lines.append('\\t'.join(row_data))
        
        QApplication.clipboard().setText('\\n'.join(lines))


class PlotlyWidget(QWidget):
    """
    Simple widget for displaying Plotly plots in a web view.
    
    Used for displaying modal coordinates over time.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.web_view = QWebEngineView(self)
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        self.setLayout(layout)
        
        # Store last used data for refresh
        self.last_time_values = None
        self.last_modal_coord = None
    
    def update_plot(self, time_values, modal_coord):
        """Update the plot with modal coordinate data."""
        self.last_time_values = time_values
        self.last_modal_coord = modal_coord
        
        fig = go.Figure()
        num_modes = modal_coord.shape[0]
        for i in range(num_modes):
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=modal_coord[i, :],
                mode='lines',
                name=f'Mode {i + 1}',
                opacity=0.7
            ))
        
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Modal Coordinate Value",
            template="plotly_white",
            font=dict(size=7),
            margin=dict(l=40, r=40, t=10, b=0),
            legend=dict(font=dict(size=7))
        )
        
        # Display
        main_win = self.window()
        main_win.plotting_handler.load_fig_to_webview(fig, self.web_view)
    
    def clear_plot(self):
        """Clear the plot."""
        self.web_view.setHtml("")
        self.last_time_values = None
        self.last_modal_coord = None


class PlotlyMaxWidget(QWidget):
    """
    Widget for displaying Plotly plots with a data table.
    
    Features:
    - Plotly web view for interactive plots
    - Side-by-side data table
    - Clipboard copy support (Ctrl+C)
    - Auto-resizing splitter
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Plotly web view
        self.web_view = QWebEngineView(self)
        
        # Data table
        self.table = QTableView(self)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        self.model = QStandardItemModel(self)
        self.model.setHorizontalHeaderLabels(["Time [s]", "Data Value"])
        self.table.setModel(self.model)
        
        # Ctrl+C shortcut
        copy_sc = QShortcut(QKeySequence.Copy, self.table)
        copy_sc.activated.connect(self.copy_selection)
        
        # Splitter
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.web_view)
        self.splitter.addWidget(self.table)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        self.setLayout(layout)
    
    def showEvent(self, event):
        """Called when widget is shown."""
        super().showEvent(event)
        QTimer.singleShot(50, self.adjust_splitter_size)
    
    def resizeEvent(self, event):
        """Called when widget is resized."""
        super().resizeEvent(event)
        self.adjust_splitter_size()
    
    def adjust_splitter_size(self):
        """Calculate and set optimal table width."""
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        v_scrollbar = self.table.verticalScrollBar()
        scrollbar_width = v_scrollbar.width() if v_scrollbar.isVisible() else 0
        
        required_width = (header.length() +
                         self.table.verticalHeader().width() +
                         self.table.frameWidth() * 2 +
                         scrollbar_width)
        
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        total_width = self.splitter.width()
        plot_width = max(450, total_width - required_width)
        
        self.splitter.setSizes([int(plot_width), int(total_width - plot_width)])
    
    def update_plot(self, time_values, traces=None):
        """
        Update plot with multiple data traces.
        
        Args:
            time_values: Array of time values.
            traces: List of dicts with 'name' and 'data' keys.
        """
        if traces is None:
            traces = []
        
        # Build figure
        fig = go.Figure()
        for trace_info in traces:
            fig.add_trace(go.Scattergl(
                x=time_values,
                y=trace_info['data'],
                mode='lines',
                name=trace_info['name']
            ))
        
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Value",
            template="plotly_white",
            font=dict(size=7),
            margin=dict(l=40, r=40, t=10, b=0),
            legend=dict(font=dict(size=7))
        )
        
        # Display
        main_win = self.window()
        main_win.plotting_handler.load_fig_to_webview(fig, self.web_view)
        
        # Populate table
        headers = ["Time [s]"] + [trace['name'] for trace in traces]
        self.model.setHorizontalHeaderLabels(headers)
        self.model.removeRows(0, self.model.rowCount())
        
        for i, t in enumerate(time_values):
            row_items = [QStandardItem(f"{t:.5f}")]
            for trace in traces:
                row_items.append(QStandardItem(f"{trace['data'][i]:.6f}"))
            self.model.appendRow(row_items)
        
        # Finalize
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    @pyqtSlot()
    def copy_selection(self):
        """Copy selected cells to clipboard as TSV."""
        sel = self.table.selectedIndexes()
        if not sel:
            return
        
        rows = sorted(idx.row() for idx in sel)
        cols = sorted(idx.column() for idx in sel)
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        
        headers = [self.model.headerData(c, Qt.Horizontal) for c in range(c0, c1 + 1)]
        lines = ['\\t'.join(headers)]
        
        for r in range(r0, r1 + 1):
            row_data = []
            for c in range(c0, c1 + 1):
                idx = self.model.index(r, c)
                text = idx.data() or ""
                row_data.append(text)
            lines.append('\\t'.join(row_data))
        
        QApplication.clipboard().setText('\\n'.join(lines))
    
    def update_envelope_plot(self, node_ids, max_values=None, min_values=None,
                            combo_of_max=None, combo_of_min=None,
                            stress_type="von_mises", combination_names=None,
                            show_top_n=50):
        """
        Update plot with envelope results (max/min over combinations).
        
        This is the MARS-SC specific method for displaying envelope results
        that show the maximum or minimum stress values across all combinations.
        
        Args:
            node_ids: Array of node IDs
            max_values: Array of maximum stress values over all combinations
            min_values: Array of minimum stress values over all combinations
            combo_of_max: Array of combination indices where max occurred
            combo_of_min: Array of combination indices where min occurred
            stress_type: Type of stress ("von_mises", "max_principal", "min_principal")
            combination_names: Optional list of combination names
            show_top_n: Number of top nodes to display in the plot (default 50)
        """
        # Stress type configuration
        stress_config = {
            'von_mises': {'title': 'Von Mises Envelope', 'label': 'σ_VM [MPa]'},
            'max_principal': {'title': 'Max Principal Envelope', 'label': 'σ1 [MPa]'},
            'min_principal': {'title': 'Min Principal Envelope', 'label': 'σ3 [MPa]'},
        }
        config = stress_config.get(stress_type, stress_config['von_mises'])
        
        # Build traces for the plot
        fig = go.Figure()
        
        traces_added = False
        
        # Add max values trace (bar chart of top N nodes)
        if max_values is not None and len(max_values) > 0:
            # Sort by max values descending and take top N
            sorted_indices = np.argsort(max_values)[::-1][:show_top_n]
            sorted_node_ids = np.array(node_ids)[sorted_indices]
            sorted_max = np.array(max_values)[sorted_indices]
            
            hover_text = []
            for i, idx in enumerate(sorted_indices):
                text = f"Node: {sorted_node_ids[i]}<br>Max: {sorted_max[i]:.4f} MPa"
                if combo_of_max is not None:
                    combo_idx = int(combo_of_max[idx])
                    # Display 1-based combination number for user-friendliness
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else f"#{combo_idx + 1}"
                    text += f"<br>At: {combo_name}"
                hover_text.append(text)
            
            fig.add_trace(go.Bar(
                x=[f"Node {nid}" for nid in sorted_node_ids],
                y=sorted_max,
                name=f'Max {config["label"]}',
                marker_color='red',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        # Add min values trace if provided
        if min_values is not None and len(min_values) > 0:
            # Sort by min values ascending and take top N (most negative)
            sorted_indices = np.argsort(min_values)[:show_top_n]
            sorted_node_ids = np.array(node_ids)[sorted_indices]
            sorted_min = np.array(min_values)[sorted_indices]
            
            hover_text = []
            for i, idx in enumerate(sorted_indices):
                text = f"Node: {sorted_node_ids[i]}<br>Min: {sorted_min[i]:.4f} MPa"
                if combo_of_min is not None:
                    combo_idx = int(combo_of_min[idx])
                    # Display 1-based combination number for user-friendliness
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else f"#{combo_idx + 1}"
                    text += f"<br>At: {combo_name}"
                hover_text.append(text)
            
            fig.add_trace(go.Bar(
                x=[f"Node {nid}" for nid in sorted_node_ids],
                y=sorted_min,
                name=f'Min {config["label"]}',
                marker_color='blue',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        if not traces_added:
            fig.add_annotation(
                text="No envelope data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=f'{config["title"]} - Top {show_top_n} Nodes',
            xaxis_title="Node",
            yaxis_title=config["label"],
            template="plotly_white",
            font=dict(size=10),
            margin=dict(l=50, r=50, t=50, b=100),
            legend=dict(font=dict(size=9)),
            barmode='group',
            xaxis_tickangle=-45
        )
        
        # Display
        try:
            main_win = self.window()
            if hasattr(main_win, 'plotting_handler') and main_win.plotting_handler:
                main_win.plotting_handler.load_fig_to_webview(fig, self.web_view)
            else:
                # Fallback: render HTML directly with embedded plotly.js (offline compatible)
                self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        except Exception as e:
            print(f"Error rendering envelope plot: {e}")
            # Use embedded plotly.js for offline compatibility
            self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        
        # Populate table with all data (not just top N)
        headers = ["Node ID"]
        if max_values is not None:
            headers.extend([f"Max {config['label']}", "Combo of Max"])
        if min_values is not None:
            headers.extend([f"Min {config['label']}", "Combo of Min"])
        
        self.model.setHorizontalHeaderLabels(headers)
        self.model.removeRows(0, self.model.rowCount())
        
        for i, nid in enumerate(node_ids):
            row_items = [QStandardItem(f"{int(nid)}")]
            if max_values is not None:
                row_items.append(QStandardItem(f"{max_values[i]:.6f}"))
                if combo_of_max is not None:
                    combo_idx = int(combo_of_max[i])
                    # Display 1-based combination number for user-friendliness
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else str(combo_idx + 1)
                    row_items.append(QStandardItem(combo_name))
                else:
                    row_items.append(QStandardItem(""))
            if min_values is not None:
                row_items.append(QStandardItem(f"{min_values[i]:.6f}"))
                if combo_of_min is not None:
                    combo_idx = int(combo_of_min[i])
                    # Display 1-based combination number for user-friendliness
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else str(combo_idx + 1)
                    row_items.append(QStandardItem(combo_name))
                else:
                    row_items.append(QStandardItem(""))
            self.model.appendRow(row_items)
        
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    def update_forces_envelope_plot(self, node_ids, max_magnitude=None, min_magnitude=None,
                                     combo_of_max=None, combo_of_min=None,
                                     combination_names=None, force_unit="N",
                                     show_top_n=50):
        """
        Update plot with nodal forces envelope results (max/min magnitude over combinations).
        
        Args:
            node_ids: Array of node IDs
            max_magnitude: Array of maximum force magnitudes over all combinations
            min_magnitude: Array of minimum force magnitudes over all combinations
            combo_of_max: Array of combination indices where max occurred
            combo_of_min: Array of combination indices where min occurred
            combination_names: Optional list of combination names
            force_unit: Force unit string (e.g., "N")
            show_top_n: Number of top nodes to display in the plot (default 50)
        """
        # Build traces for the plot
        fig = go.Figure()
        
        traces_added = False
        
        # Add max magnitude trace (bar chart of top N nodes)
        if max_magnitude is not None and len(max_magnitude) > 0:
            # Sort by max magnitude descending and take top N
            sorted_indices = np.argsort(max_magnitude)[::-1][:show_top_n]
            sorted_node_ids = np.array(node_ids)[sorted_indices]
            sorted_max = np.array(max_magnitude)[sorted_indices]
            
            hover_text = []
            for i, idx in enumerate(sorted_indices):
                text = f"Node: {sorted_node_ids[i]}<br>Max |F|: {sorted_max[i]:.4f} {force_unit}"
                if combo_of_max is not None:
                    combo_idx = int(combo_of_max[idx])
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else f"#{combo_idx + 1}"
                    text += f"<br>At: {combo_name}"
                hover_text.append(text)
            
            fig.add_trace(go.Bar(
                x=[f"Node {nid}" for nid in sorted_node_ids],
                y=sorted_max,
                name=f'Max |F| [{force_unit}]',
                marker_color='darkred',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        # Add min magnitude trace if provided
        if min_magnitude is not None and len(min_magnitude) > 0:
            # Sort by min magnitude ascending and take top N
            sorted_indices = np.argsort(min_magnitude)[:show_top_n]
            sorted_node_ids = np.array(node_ids)[sorted_indices]
            sorted_min = np.array(min_magnitude)[sorted_indices]
            
            hover_text = []
            for i, idx in enumerate(sorted_indices):
                text = f"Node: {sorted_node_ids[i]}<br>Min |F|: {sorted_min[i]:.4f} {force_unit}"
                if combo_of_min is not None:
                    combo_idx = int(combo_of_min[idx])
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else f"#{combo_idx + 1}"
                    text += f"<br>At: {combo_name}"
                hover_text.append(text)
            
            fig.add_trace(go.Bar(
                x=[f"Node {nid}" for nid in sorted_node_ids],
                y=sorted_min,
                name=f'Min |F| [{force_unit}]',
                marker_color='darkblue',
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        if not traces_added:
            fig.add_annotation(
                text="No nodal forces envelope data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=f'Nodal Forces Envelope - Top {show_top_n} Nodes',
            xaxis_title="Node",
            yaxis_title=f"Force Magnitude [{force_unit}]",
            template="plotly_white",
            font=dict(size=10),
            margin=dict(l=50, r=50, t=50, b=100),
            legend=dict(font=dict(size=9)),
            barmode='group',
            xaxis_tickangle=-45
        )
        
        # Display
        try:
            main_win = self.window()
            if hasattr(main_win, 'plotting_handler') and main_win.plotting_handler:
                main_win.plotting_handler.load_fig_to_webview(fig, self.web_view)
            else:
                self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        except Exception as e:
            print(f"Error rendering forces envelope plot: {e}")
            self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        
        # Populate table with all data
        headers = ["Node ID", f"Max |F| [{force_unit}]", "Combo of Max", f"Min |F| [{force_unit}]", "Combo of Min"]
        
        self.model.setHorizontalHeaderLabels(headers)
        self.model.removeRows(0, self.model.rowCount())
        
        for i, nid in enumerate(node_ids):
            row_items = [QStandardItem(f"{int(nid)}")]
            
            if max_magnitude is not None:
                row_items.append(QStandardItem(f"{max_magnitude[i]:.6f}"))
                if combo_of_max is not None:
                    combo_idx = int(combo_of_max[i])
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else str(combo_idx + 1)
                    row_items.append(QStandardItem(combo_name))
                else:
                    row_items.append(QStandardItem(""))
            else:
                row_items.extend([QStandardItem(""), QStandardItem("")])
            
            if min_magnitude is not None:
                row_items.append(QStandardItem(f"{min_magnitude[i]:.6f}"))
                if combo_of_min is not None:
                    combo_idx = int(combo_of_min[i])
                    combo_name = combination_names[combo_idx] if combination_names and combo_idx < len(combination_names) else str(combo_idx + 1)
                    row_items.append(QStandardItem(combo_name))
                else:
                    row_items.append(QStandardItem(""))
            else:
                row_items.extend([QStandardItem(""), QStandardItem("")])
            
            self.model.appendRow(row_items)
        
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    def update_max_over_combinations_plot(self, combination_indices, max_values_per_combo,
                                          min_values_per_combo=None, combination_names=None,
                                          stress_type="von_mises"):
        """
        Plot envelope max/min values vs combination number.
        
        This shows how the maximum (or minimum) stress across ALL nodes changes
        for each combination - similar to original MARS "Maximum Over Time" plots.
        
        Args:
            combination_indices: Array of combination indices (0, 1, 2, ...)
            max_values_per_combo: Array of max values (one per combination - max across all nodes)
            min_values_per_combo: Optional array of min values (one per combination)
            combination_names: Optional list of combination names for x-axis labels
            stress_type: Type of stress for axis label ("von_mises", "max_principal", "min_principal")
        """
        # Stress type configuration
        stress_config = {
            'von_mises': {'title': 'Maximum Over Combinations', 'label': 'σ_VM [MPa]', 'short': 'Von Mises'},
            'max_principal': {'title': 'Maximum Over Combinations', 'label': 'σ1 [MPa]', 'short': 'Max Principal'},
            'min_principal': {'title': 'Minimum Over Combinations', 'label': 'σ3 [MPa]', 'short': 'Min Principal'},
        }
        config = stress_config.get(stress_type, stress_config['von_mises'])
        
        # Build figure
        fig = go.Figure()
        
        x_values = np.array(combination_indices)
        
        # Create x-axis labels (1-based for user-friendliness)
        if combination_names and len(combination_names) == len(x_values):
            x_labels = combination_names
        else:
            x_labels = [f"Combo {i+1}" for i in x_values]
        
        traces_added = False
        
        # Add max values trace (line plot)
        if max_values_per_combo is not None and len(max_values_per_combo) > 0:
            max_vals = np.array(max_values_per_combo)
            
            # Create hover text
            hover_text = []
            for i, val in enumerate(max_vals):
                name = x_labels[i] if i < len(x_labels) else f"Combo {i+1}"
                hover_text.append(f"{name}<br>Max: {val:.4f} MPa")
            
            fig.add_trace(go.Scatter(
                x=x_values + 1,  # 1-based for display
                y=max_vals,
                mode='lines+markers',
                name=f'Max {config["short"]}',
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red'),
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        # Add min values trace if provided
        if min_values_per_combo is not None and len(min_values_per_combo) > 0:
            min_vals = np.array(min_values_per_combo)
            
            # Create hover text
            hover_text = []
            for i, val in enumerate(min_vals):
                name = x_labels[i] if i < len(x_labels) else f"Combo {i+1}"
                hover_text.append(f"{name}<br>Min: {val:.4f} MPa")
            
            fig.add_trace(go.Scatter(
                x=x_values + 1,  # 1-based for display
                y=min_vals,
                mode='lines+markers',
                name=f'Min {config["short"]}',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue'),
                hovertext=hover_text,
                hoverinfo='text'
            ))
            traces_added = True
        
        if not traces_added:
            fig.add_annotation(
                text="No envelope data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig.update_layout(
            title=config["title"],
            xaxis_title="Combination #",
            yaxis_title=config["label"],
            template="plotly_white",
            font=dict(size=10),
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(font=dict(size=9)),
            xaxis=dict(
                tickmode='array',
                tickvals=x_values + 1,
                ticktext=[f"{i+1}" for i in x_values],
                dtick=1
            )
        )
        
        # Display
        try:
            main_win = self.window()
            if hasattr(main_win, 'plotting_handler') and main_win.plotting_handler:
                main_win.plotting_handler.load_fig_to_webview(fig, self.web_view)
            else:
                self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        except Exception as e:
            print(f"Error rendering max over combinations plot: {e}")
            self.web_view.setHtml(fig.to_html(include_plotlyjs=True, full_html=True))
        
        # Populate table
        headers = ["Combo #", "Name"]
        if max_values_per_combo is not None:
            headers.append(f"Max {config['label']}")
        if min_values_per_combo is not None:
            headers.append(f"Min {config['label']}")
        
        self.model.setHorizontalHeaderLabels(headers)
        self.model.removeRows(0, self.model.rowCount())
        
        for i in range(len(combination_indices)):
            row_items = [
                QStandardItem(f"{combination_indices[i] + 1}"),  # 1-based
                QStandardItem(x_labels[i] if i < len(x_labels) else "")
            ]
            if max_values_per_combo is not None:
                row_items.append(QStandardItem(f"{max_values_per_combo[i]:.6f}"))
            if min_values_per_combo is not None:
                row_items.append(QStandardItem(f"{min_values_per_combo[i]:.6f}"))
            self.model.appendRow(row_items)
        
        self.table.resizeColumnsToContents()
        QTimer.singleShot(0, self.adjust_splitter_size)
    
    def clear_plot(self):
        """Clear the plot and table."""
        self.web_view.setHtml("")
        self.model.removeRows(0, self.model.rowCount())
        self.model.setHorizontalHeaderLabels(["Node ID", "Max Value", "Min Value"])
