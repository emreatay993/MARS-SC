"""
Centralized style constants for MARS-SC GUI.

This module contains all inline stylesheet strings used throughout the application.
These styles are applied directly to widgets using setStyleSheet() to match
the legacy approach, but centralized here for maintainability.
"""

# Button styles
BUTTON_STYLE = """
    QPushButton {
        background-color: #e7f0fd;
        border: 1px solid #5b9bd5;
        padding: 10px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #cce4ff;
    }
"""

# Group box styles
GROUP_BOX_STYLE = """
    QGroupBox {
        border: 1px solid #5b9bd5;
        border-radius: 5px;
        margin-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
    }
"""

# Tab widget styles (legacy-matching)
TAB_STYLE = """
QTabWidget::pane {
    border: 1px solid #5b9bd5;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #d6e4f5;
    color: #666666;
    border: 1px solid #5b9bd5;
    padding: 3px 8px;            /* inner spacing (v,h) */
    margin: 2px;                 /* spacing between tabs */
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    font-family: Arial;
    font-size: 8pt;              /* legacy uses 8pt */
    min-width: 100px;            /* adjust to match screenshot width */
    min-height: 20px;            /* vertical size */
}

QTabBar::tab:selected {
    background-color: #e7f0fd;
    color: #000000;
    border: 2px solid #5b9bd5;  /* thicker active border */
    border-bottom: 1px solid #e7f0fd;
    margin-bottom: -1px;         /* merge with pane */
}

QTabBar::tab:!selected {
    margin-top: 3px;             /* give inactive tabs a lifted look */
}

QTabBar::tab:hover {
    background-color: #cce4ff;
}
"""

# Input field styles
READONLY_INPUT_STYLE = "background-color: #f0f0f0; color: grey; border: 1px solid #5b9bd5; padding: 5px;"

# Checkbox styles
CHECKBOX_STYLE = """
QCheckBox {
    spacing: 8px;
    font-family: Arial;
    font-size: 8pt;
    margin: 3px 0;
}

/* Note: Do not style the indicator to preserve native checkmark rendering */
QCheckBox:hover {
    color: #4a8bc2; /* subtle label hover only */
}
"""

# Console/TextEdit styles
# Use no inner border so the tab pane's 1px border defines the outline
CONSOLE_STYLE = "background-color: #ffffff; border: 0px;"

# Progress bar styles
PROGRESS_BAR_STYLE = "border: 1px solid #5b9bd5; padding: 10px; background-color: #ffffff;"

# Navigator/Dock widget styles (matching legacy)
NAVIGATOR_TITLE_STYLE = """
    QDockWidget::title {
        background-color: #e7f0fd;
        color: black;
        font-weight: bold;
        font-size: 9px;
        padding-top: 2px;
        padding-bottom: 2px;
        padding-left: 8px;
        padding-right: 8px;
        border-bottom: 2px solid #5b9bd5;
    }
"""

TREE_VIEW_STYLE = """
    QTreeView {
        font-size: 7.5pt;
        background-color: #ffffff;
        alternate-background-color: #f5f5f5;
        border: none;
    }
    QTreeView::item:hover {
        background-color: #d0e4ff;
    }
    QTreeView::item:selected {
        background-color: #5b9bd5;
        color: #ffffff;
    }
    QHeaderView::section {
        background-color: #e7f0fd;
        padding: 2px 3px;  /* vertical, horizontal - reduced vertical margin */
        border: none;
        border-right: 1px solid #5b9bd5;
        border-bottom: 1px solid #5b9bd5;
        font-weight: bold;
    }
"""

# Menu bar styles (matching legacy - white background, not blue)
MENU_BAR_STYLE = """
    QMenuBar {
        background-color: #ffffff;
        color: #000000;
        padding: 2px;
        font-family: Arial;
        font-size: 12px;
    }
    QMenuBar::item {
        background-color: #ffffff;
        color: #000000;
        padding: 2px 5px;
        margin: 0px;
    }
    QMenuBar::item:selected {
        background-color: #e0e0e0;
        border-radius: 2px;
    }
    QMenu {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #5b9bd5;
    }
    QMenu::item {
        padding: 4px 20px;
    }
    QMenu::item:selected {
        background-color: #5b9bd5;
        color: white;
    }
"""

# Dialog styles (from dialog.qss)
DIALOG_STYLE = """
QDialog {
    background-color: #ffffff;
    border: 1px solid #5b9bd5;
    border-radius: 5px;
}

QDialog QGroupBox {
    border: 1px solid #5b9bd5;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
    background-color: #ffffff;
}

QDialog QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #e7f0fd;
    color: #000000;
    font-weight: bold;
    font-size: 8pt;
}

QDialog QPushButton {
    background-color: #e7f0fd;
    border: 1px solid #5b9bd5;
    padding: 8px 15px;
    border-radius: 5px;
    font-family: Arial;
    font-size: 8pt;
    min-width: 60px;
}

QDialog QPushButton:hover {
    background-color: #cce4ff;
}

QDialog QPushButton:pressed {
    background-color: #a8d4ff;
}

QLabel {
    color: #000000;
    font-family: Arial;
    font-size: 8pt;
}

QLabel#titleLabel {
    font-weight: bold;
    color: #333333;
    text-decoration: underline;
    padding: 4px 0px 6px 7px;
    font-size: 9pt;
}

QLabel.titleLabel {
    font-weight: bold;
    color: #333333;
    text-decoration: underline;
    padding: 4px 0px 6px 7px;
    font-size: 8pt;
}

QLabel#currentSettingsLabel {
    font-family: Consolas, Monaco, monospace;
    font-size: 8pt;
    background-color: #f8f8f8;
    border: 1px solid #cccccc;
    border-radius: 3px;
    padding: 8px;
    margin: 5px 0;
}

QLabel.currentSettingsLabel {
    font-family: Consolas, Monaco, monospace;
    font-size: 9pt;
    background-color: #f0f0f0;
    border: 1px solid #dcdcdc;
    border-radius: 3px;
    padding: 8px;
    margin: 5px 0;
}

QMessageBox {
    background-color: #ffffff;
}

QMessageBox QLabel {
    color: #000000;
    font-family: Arial;
    font-size: 8pt;
}

QMessageBox QPushButton {
    background-color: #e7f0fd;
    border: 1px solid #5b9bd5;
    padding: 6px 12px;
    border-radius: 3px;
    font-family: Arial;
    font-size: 8pt;
    min-width: 50px;
}

QMessageBox QPushButton:hover {
    background-color: #cce4ff;
}

QFileDialog {
    background-color: #ffffff;
}

QFileDialog QTreeView {
    font-size: 7.5pt;
    background-color: #ffffff;
    alternate-background-color: #f5f5f5;
    border: 1px solid #5b9bd5;
    selection-background-color: #5b9bd5;
    selection-color: #ffffff;
}

QFileDialog QTreeView::item:hover {
    background-color: #d0e4ff;
}

QFileDialog QTreeView::item:selected {
    background-color: #5b9bd5;
    color: #ffffff;
}

QFileDialog QLineEdit {
    background-color: #ffffff;
    border: 1px solid #5b9bd5;
    padding: 3px;
    border-radius: 2px;
}

QFileDialog QComboBox {
    background-color: #ffffff;
    border: 1px solid #5b9bd5;
    padding: 3px;
    border-radius: 2px;
}

QDialog#AdvancedSettingsDialog {
    background-color: #ffffff;
    border: 2px solid #5b9bd5;
    border-radius: 8px;
}

QDialog#AdvancedSettingsDialog QGroupBox {
    border: 1px solid #5b9bd5;
    border-radius: 5px;
    margin: 10px;
    padding-top: 15px;
}

QDialog#AdvancedSettingsDialog QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    background-color: #e7f0fd;
    color: #000000;
    font-weight: bold;
    font-size: 9pt;
}

QDialog#AdvancedSettingsDialog QLabel {
    font-family: Arial;
    font-size: 8pt;
    padding: 2px 0;
}

QDialog#AdvancedSettingsDialog QLineEdit,
QDialog#AdvancedSettingsDialog QSpinBox,
QDialog#AdvancedSettingsDialog QDoubleSpinBox,
QDialog#AdvancedSettingsDialog QComboBox {
    background-color: #ffffff;
    border: 1px solid #5b9bd5;
    padding: 4px;
    border-radius: 3px;
    font-family: Arial;
    font-size: 8pt;
}

QDialog#AdvancedSettingsDialog QCheckBox {
    spacing: 8px;
    font-family: Arial;
    font-size: 8pt;
    margin: 3px 0;
}
"""

# Dialog group box styles (separate for when only group box needs styling)
DIALOG_GROUP_BOX_STYLE = """
QGroupBox {
    border: 1px solid #5b9bd5;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
    background-color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #e7f0fd;
    color: #000000;
    font-weight: bold;
    font-size: 8pt;
}
"""

# Context menu styles (from plot.qss)
CONTEXT_MENU_STYLE = """
QMenu {
    background-color: #ffffff;
    border: 1px solid #5b9bd5;
    border-radius: 5px;
    padding: 5px;
}

QMenu::item {
    padding: 5px 10px;
    border-radius: 3px;
    font-family: Arial;
    font-size: 8pt;
}

QMenu::item:selected {
    background-color: #e7f0fd;
    color: #000000;
}

QMenu::separator {
    height: 1px;
    background-color: #5b9bd5;
    margin: 3px 0;
}
"""

