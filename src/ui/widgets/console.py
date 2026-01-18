"""
Console logger widget for MARS-SC (Solution Combination).

Provides a Logger class that redirects stdout to a QTextEdit widget with buffered updates.
"""

import sys
from PyQt5.QtCore import QObject, QTimer, pyqtSlot
from PyQt5.QtGui import QTextCursor


class Logger(QObject):
    """
    Logger that redirects stdout to a QTextEdit widget with buffering.
    
    This class captures console output and displays it in a GUI text widget,
    using a buffer and timer to batch updates for better performance.
    """
    
    def __init__(self, text_edit, flush_interval=200):
        """
        Initialize the Logger.
        
        Args:
            text_edit: QTextEdit widget to display log messages.
            flush_interval: Interval in milliseconds to flush the buffer (default: 200).
        """
        super().__init__()
        self.text_edit = text_edit
        self.terminal = sys.stdout
        self.log_buffer = ""  # Buffer for messages
        self.flush_interval = flush_interval  # in milliseconds
        
        # Set up a QTimer to flush the buffer periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.flush_buffer)
        self.timer.start(self.flush_interval)
    
    def write(self, message):
        """
        Write a message to both terminal and buffer.
        
        Args:
            message: Text message to write.
        """
        # Write to the original terminal
        self.terminal.write(message)
        # Append the message to the buffer
        self.log_buffer += message
    
    @pyqtSlot()
    def flush_buffer(self):
        """Flush the buffered messages to the text edit widget."""
        if self.log_buffer:
            # Append the buffered messages to the text edit in one update
            self.text_edit.moveCursor(QTextCursor.End)
            self.text_edit.insertPlainText(self.log_buffer)
            self.text_edit.moveCursor(QTextCursor.End)
            self.text_edit.ensureCursorVisible()
            self.log_buffer = ""
    
    def flush(self):
        """Flush the buffer (called by sys.stdout.flush())."""
        self.flush_buffer()
