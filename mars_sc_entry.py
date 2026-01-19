#!/usr/bin/env python
"""
PyInstaller entry point for MARS-SC application.

This wrapper ensures proper module paths are set before importing the application.
"""

import sys
import os

# ============================================================================
# PATH SETUP - Must happen before ANY application imports
# ============================================================================

def _get_base_path():
    """Get the base path for module resolution."""
    if getattr(sys, 'frozen', False):
        # Running as frozen executable - use PyInstaller's temp directory
        return sys._MEIPASS
    else:
        # Running as script - use src directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, 'src')
        if os.path.exists(src_dir):
            return src_dir
        return script_dir

# Set up paths immediately
_BASE_PATH = _get_base_path()
if _BASE_PATH not in sys.path:
    sys.path.insert(0, _BASE_PATH)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main entry point for the MARS-SC application."""
    # Import PyQt5 first
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        print(f"ERROR: Failed to import PyQt5: {e}")
        print(f"sys.path: {sys.path[:5]}...")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Import application controller
    try:
        from ui.application_controller import ApplicationController
    except ImportError as e:
        print(f"ERROR: Failed to import application modules: {e}")
        print(f"Base path: {_BASE_PATH}")
        print(f"sys.path[0]: {sys.path[0]}")
        print(f"Frozen: {getattr(sys, 'frozen', False)}")
        if os.path.exists(_BASE_PATH):
            print(f"\nContents of {_BASE_PATH}:")
            try:
                items = sorted(os.listdir(_BASE_PATH))[:30]
                for item in items:
                    full_path = os.path.join(_BASE_PATH, item)
                    item_type = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
                    print(f"  {item_type} {item}")
            except Exception as list_err:
                print(f"  Could not list: {list_err}")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create and run application
    app = QApplication(sys.argv)
    main_window = ApplicationController()
    main_window.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
