"""
Main window for the MARS-SC: Solution Combination application.

Provides the main application window with menu bar, navigator, and tab widgets
for solver and display functionality. MARS-SC performs linear combination of 
static analysis results from two RST files and computes stress envelopes.
"""

from pathlib import Path

from PyQt5.QtCore import Qt, QDir, QObject, QEvent, QSettings, pyqtSlot
from PyQt5.QtGui import QPalette, QColor, QIcon
from PyQt5.QtWidgets import (
    QAction, QApplication, QDockWidget, QFileSystemModel,
    QMainWindow, QMenuBar, QMessageBox, QTabWidget, QTreeView, QToolTip
)

from ui.solver_tab import SolverTab
from ui.display_tab import DisplayTab
from ui.handlers.plotting_handler import PlottingHandler
from ui.handlers.navigator_handler import NavigatorHandler
from utils.tooltips import TOOLTIP_NAVIGATOR
from ui.styles.style_constants import (
    MENU_BAR_STYLE, NAVIGATOR_TITLE_STYLE, TREE_VIEW_STYLE, TAB_STYLE,
    TOOLTIP_STYLE
)


class TooltipEventFilter(QObject):
    """App-wide filter that can disable all tooltip popups."""

    def __init__(self, enabled: bool = True, parent=None):
        super().__init__(parent)
        self._enabled = enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable tooltip popups globally."""
        self._enabled = bool(enabled)

    def eventFilter(self, obj, event):
        """Block tooltip events when disabled."""
        if event.type() == QEvent.ToolTip and not self._enabled:
            QToolTip.hideText()
            return True
        return False


class ApplicationController(QMainWindow):
    """
    Main application controller for MARS-SC.

    Coordinates tabs, signal connections between components, and application-level
    state and navigation for solution combination analysis.
    """
    
    def __init__(self):
        """Initialize the application controller."""
        super().__init__()

        # Persistent app settings
        self.settings = QSettings("MARS-SC", "MARS-SC")
        self.tooltips_enabled = self.settings.value(
            "view/tooltips_enabled", True, type=bool
        )

        # Install global tooltip filter
        self._tooltip_filter = TooltipEventFilter(
            enabled=self.tooltips_enabled,
            parent=self
        )
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._tooltip_filter)

        # Set window background color (matching legacy)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(230, 230, 230))  # Light gray background
        self.setPalette(palette)
        
        # Apply global tooltip style (kept consistent with MARS_ project)
        self.setStyleSheet(TOOLTIP_STYLE)

        # Handlers
        self.plotting_handler = PlottingHandler()

        # Window configuration
        self.setWindowTitle('MARS-SC: Solution Combination - v1.0')
        self.setGeometry(40, 40, 600, 800)
        
        # Set application icon
        self._set_window_icon()
        
        # Create UI components (order matters - navigator before menu bar)
        self._create_tabs()
        self._create_navigator()
        self._create_menu_bar()
        self._connect_signals()
    
    def _set_window_icon(self):
        """Set the application window icon."""
        # Get path to icon file (relative to project root)
        icon_dir = Path(__file__).parent.parent.parent / "resources" / "icons"
        
        # Try .ico first (best for Windows), then fall back to PNG
        icon_paths = [
            icon_dir / "mars_icon.ico",
            icon_dir / "mars_128.png",
            icon_dir / "mars_64.png"
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                break
    
    def _create_menu_bar(self):
        """Create the menu bar with File and View menus."""
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        # Apply menu bar styles (white background, matching legacy)
        self.menu_bar.setStyleSheet(MENU_BAR_STYLE)
        
        # File menu
        file_menu = self.menu_bar.addMenu("File")
        select_dir_action = QAction("Select Project Directory", self)
        select_dir_action.triggered.connect(
            lambda: self.navigator_handler.select_project_directory(self)
        )
        file_menu.addAction(select_dir_action)
        
        # View menu
        view_menu = self.menu_bar.addMenu("View")
        toggle_navigator_action = self.navigator_dock.toggleViewAction()
        toggle_navigator_action.setText("Navigator")
        view_menu.addAction(toggle_navigator_action)

        self.toggle_tooltips_action = QAction("Enable Tooltips", self)
        self.toggle_tooltips_action.setCheckable(True)
        self.toggle_tooltips_action.setChecked(self.tooltips_enabled)
        self.toggle_tooltips_action.toggled.connect(self._on_tooltips_toggled)
        view_menu.addAction(self.toggle_tooltips_action)
    
    def _create_navigator(self):
        """Create file navigator dock widget."""
        self.navigator_dock = QDockWidget("Navigator", self)
        self.navigator_dock.setToolTip(TOOLTIP_NAVIGATOR)
        self.navigator_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.navigator_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
        )
        
        # File system model
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.file_model.setNameFilters(["*.csv", "*.rst", "*.txt"])
        self.file_model.setNameFilterDisables(False)
        
        # Tree view
        self.tree_view = QTreeView()
        self.tree_view.setToolTip(TOOLTIP_NAVIGATOR)
        self.tree_view.setModel(self.file_model)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setMinimumWidth(240)
        self.tree_view.setSortingEnabled(True)
        
        # Hide unwanted columns
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnWidth(0, 250)
        self.tree_view.header().setSectionResizeMode(
            0, self.tree_view.header().ResizeToContents
        )
        
        # Apply styles
        self._apply_navigator_styles()
        
        # Enable drag and drop
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setSelectionMode(QTreeView.SingleSelection)
        self.tree_view.setDragDropMode(QTreeView.DragDrop)
        
        # Set widget
        self.navigator_dock.setWidget(self.tree_view)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.navigator_dock)

        # Create handler and connect the signal to the handler
        self.navigator_handler = NavigatorHandler(self.file_model, self.tree_view, self.solver_tab)
        self.tree_view.doubleClicked.connect(self.navigator_handler.open_navigator_file)
    
    def _apply_navigator_styles(self):
        """Apply styles to navigator components (matching legacy approach)."""
        # Apply blue navigator title and tree view styles
        self.navigator_dock.setStyleSheet(NAVIGATOR_TITLE_STYLE)
        self.tree_view.setStyleSheet(TREE_VIEW_STYLE)
    
    def _create_tabs(self):
        """Create main tab widget with solver and display tabs."""
        self.tab_widget = QTabWidget()
        # Apply tab styles (matching legacy approach)
        self.tab_widget.setStyleSheet(TAB_STYLE)
        
        # Create tabs
        self.solver_tab = SolverTab()
        self.display_tab = DisplayTab()

        self.display_tab.set_plotting_handler(self.plotting_handler) #TODO: Add this plotting handler method inside Display tab
        
        self.tab_widget.addTab(self.solver_tab, "Main Window")
        self.tab_widget.addTab(self.display_tab, "Display")
        
        self.setCentralWidget(self.tab_widget)
    
    def _connect_signals(self):
        """Connect signals between tabs."""
        # Connect solver tab to display tab
        self.solver_tab.initial_data_loaded.connect(
            self.display_tab._setup_initial_view
        )
        
        # Connect solver display payload to display tab
        self.solver_tab.display_payload_ready.connect(
            self.display_tab.update_view_with_payload
        )
        
        self.display_tab.node_picked_signal.connect(
            self._on_node_picked
        )
        self.display_tab.node_picked_for_history_popup.connect(
            self._on_node_picked_for_history_popup
        )
    
    @pyqtSlot(int)
    def _on_node_picked(self, node_id: int):
        """Handle generic node picking from display tab (in-tab history output)."""
        self._trigger_node_history(node_id, popup=False)

    @pyqtSlot(int)
    def _on_node_picked_for_history_popup(self, node_id: int):
        """Handle context-menu node picking from display tab (popup history output)."""
        self._trigger_node_history(node_id, popup=True)

    def _trigger_node_history(self, node_id: int, popup: bool) -> None:
        """Trigger combination-history solve for a selected node."""
        # Update the node ID in solver tab for combination history mode
        self.solver_tab.node_line_edit.setText(str(node_id))
        self.solver_tab.console_textbox.append(f"Selected Node: {node_id}")
        
        # Enable combination history mode if not already
        if not self.solver_tab.combination_history_checkbox.isChecked():
            self.solver_tab.combination_history_checkbox.setChecked(True)
        
        # Trigger the combination history solve automatically
        # This mimics the behavior of original MARS plot_history_for_node
        self.solver_tab.plot_combination_history_for_node(
            node_id,
            open_popup=popup,
        )

    @pyqtSlot(bool)
    def _on_tooltips_toggled(self, checked: bool):
        """Enable/disable tooltips globally and persist the preference."""
        self.tooltips_enabled = bool(checked)
        self._tooltip_filter.set_enabled(self.tooltips_enabled)
        self.settings.setValue("view/tooltips_enabled", self.tooltips_enabled)
        if not self.tooltips_enabled:
            QToolTip.hideText()

    def closeEvent(self, event):
        """Clean up temporary files on application close."""
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._tooltip_filter)
        self.plotting_handler.cleanup_temp_files()
        event.accept()
