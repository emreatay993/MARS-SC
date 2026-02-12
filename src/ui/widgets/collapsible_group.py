"""
Collapsible Group Box Widget for MARS-SC.

Provides a QGroupBox-like widget that can be collapsed/expanded by clicking
on the header. When collapsed, the content is hidden and the widget takes
minimal vertical space, allowing other widgets in the layout to expand.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QSizePolicy
)
from PyQt5.QtGui import QFont


class CollapsibleGroupBox(QWidget):
    """
    A collapsible group box widget.
    
    Features:
    - Clickable header to toggle collapse/expand
    - Arrow indicator showing state
    - Emits signal when toggled
    - Properly releases space when collapsed
    
    Usage:
        group = CollapsibleGroupBox("Section Title")
        group.setContentLayout(your_layout)
        # or
        group.addWidget(your_widget)
    """
    
    toggled = pyqtSignal(bool)  # Emitted when collapsed state changes (True = expanded)
    
    def __init__(self, title: str = "", parent=None, initially_expanded: bool = True):
        """
        Initialize the collapsible group box.
        
        Args:
            title: The title text for the header.
            parent: Parent widget.
            initially_expanded: Whether to start expanded (default True).
        """
        super().__init__(parent)
        
        self._is_expanded = initially_expanded
        self._title = title
        
        self._setup_ui()
        
        # Set initial state
        self._content_area.setVisible(self._is_expanded)
        self._update_arrow()
    
    def _setup_ui(self):
        """Setup the widget UI."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header frame (clickable)
        self._header_frame = QFrame()
        self._header_frame.setFrameShape(QFrame.StyledPanel)
        self._header_frame.setStyleSheet("""
            QFrame {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
            }
            QFrame:hover {
                background-color: #d0d0d0;
            }
        """)
        self._header_frame.setCursor(Qt.PointingHandCursor)
        self._header_frame.setFixedHeight(28)
        
        header_layout = QHBoxLayout(self._header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)
        
        # Arrow indicator
        self._arrow_label = QLabel()
        self._arrow_label.setFixedWidth(16)
        self._arrow_label.setAlignment(Qt.AlignCenter)
        self._arrow_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        
        # Title label
        self._title_label = QLabel(self._title)
        self._title_label.setFont(QFont('Arial', 9, QFont.Bold))
        self._title_label.setStyleSheet("background: transparent; border: none;")
        
        header_layout.addWidget(self._arrow_label)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        
        # Content area
        self._content_area = QWidget()
        self._content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._content_layout = QVBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(0, 4, 0, 0)
        self._content_layout.setSpacing(0)
        
        # Add to main layout
        main_layout.addWidget(self._header_frame)
        main_layout.addWidget(self._content_area)
        
        # Set size policy - this is key for proper space redistribution
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    
    def _update_arrow(self):
        """Update the arrow indicator based on expanded state."""
        if self._is_expanded:
            self._arrow_label.setText("▼")
        else:
            self._arrow_label.setText("▶")
    
    def mousePressEvent(self, event):
        """Handle mouse press on the widget."""
        # Check if click was on the header
        if self._header_frame.geometry().contains(event.pos()):
            self.toggle()
        super().mousePressEvent(event)
    
    def toggle(self):
        """Toggle the collapsed/expanded state."""
        self._is_expanded = not self._is_expanded
        self._content_area.setVisible(self._is_expanded)
        self._update_arrow()
        self.toggled.emit(self._is_expanded)
        
        # Force layout update
        self.updateGeometry()
        if self.parent():
            self.parent().updateGeometry()
    
    def expand(self):
        """Expand the group box."""
        if not self._is_expanded:
            self.toggle()
    
    def collapse(self):
        """Collapse the group box."""
        if self._is_expanded:
            self.toggle()
    
    def isExpanded(self) -> bool:
        """Return whether the group box is expanded."""
        return self._is_expanded
    
    def setTitle(self, title: str):
        """Set the title text."""
        self._title = title
        self._title_label.setText(title)
    
    def title(self) -> str:
        """Get the title text."""
        return self._title
    
    def setContentLayout(self, layout):
        """
        Set the layout for the content area.
        
        Args:
            layout: QLayout to use for the content.
        """
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        
        # Create a container widget with the new layout
        container = QWidget()
        container.setLayout(layout)
        self._content_layout.addWidget(container)
    
    def addWidget(self, widget):
        """
        Add a widget to the content area.
        
        Args:
            widget: QWidget to add.
        """
        self._content_layout.addWidget(widget)
    
    def contentLayout(self):
        """Get the content layout."""
        return self._content_layout
    
    def setHeaderStyle(self, stylesheet: str):
        """Set custom stylesheet for the header frame."""
        self._header_frame.setStyleSheet(stylesheet)
    
    def setContentMargins(self, left: int, top: int, right: int, bottom: int):
        """Set margins for the content area."""
        self._content_layout.setContentsMargins(left, top, right, bottom)


class CollapsibleGroupBoxStyled(CollapsibleGroupBox):
    """
    Collapsible group box with styling matching MARS-SC theme.
    
    This version includes styling that matches the existing GROUP_BOX_STYLE
    from the application.
    """
    
    def __init__(self, title: str = "", parent=None, initially_expanded: bool = True):
        super().__init__(title, parent, initially_expanded)
        
        # Apply MARS-SC styled header
        self._header_frame.setStyleSheet("""
            QFrame {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f0f0f0, stop: 1 #e0e0e0
                );
                border: 1px solid #b0b0b0;
                border-radius: 4px;
            }
            QFrame:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e8e8e8, stop: 1 #d8d8d8
                );
            }
        """)
        
        # Style the title
        self._title_label.setStyleSheet("""
            background: transparent;
            border: none;
            color: #333;
        """)
