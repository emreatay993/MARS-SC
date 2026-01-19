# PyInstaller runtime hook for MARS-SC
# This script runs BEFORE the main entry point
# It ensures sys.path is correctly configured

import sys
import os

def _configure_paths():
    """Configure Python path for frozen application."""
    if getattr(sys, 'frozen', False):
        # Get the directory where PyInstaller extracted files
        bundle_dir = sys._MEIPASS
        
        # Ensure bundle directory is first in path
        if bundle_dir in sys.path:
            sys.path.remove(bundle_dir)
        sys.path.insert(0, bundle_dir)
        
        # Set environment variable for any subprocesses
        os.environ['MARS_SC_BUNDLE_DIR'] = bundle_dir

# Execute immediately when this hook is loaded
_configure_paths()
