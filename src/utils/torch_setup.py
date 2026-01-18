"""
PyTorch initialization module for Windows CUDA compatibility.

This module MUST be imported before any other module that uses torch.
It fixes DLL loading issues on Windows when:
- Running from PyInstaller frozen executables
- Old CUDA DLLs exist in PATH (e.g., from NVIDIA PhysX)
- Running on different Windows PCs with varying configurations

Usage:
    In your entry point (main.py), add as the FIRST import:
        import utils.torch_setup  # noqa: F401
        
Diagnostics:
    Set environment variable MARS_TORCH_DIAG=1 for verbose output.
"""

from __future__ import annotations

import os
import sys
import warnings

# Globals to prevent garbage collection of DLL directory handles
_DLL_DIR_HANDLES: list[object] = []
_INITIALIZATION_COMPLETE = False


def _is_frozen() -> bool:
    """Check if running from PyInstaller frozen executable."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _get_base_path() -> str:
    """Get the base path for the application."""
    if _is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log_diag(message: str) -> None:
    """Log diagnostic message if MARS_TORCH_DIAG is enabled."""
    if os.environ.get("MARS_TORCH_DIAG") == "1":
        print(f"[MARS torch_setup] {message}")


def _add_dll_directory(path: str) -> bool:
    """
    Add a directory to the DLL search path.
    
    Returns True if successful, False otherwise.
    """
    if not path or not os.path.isdir(path):
        return False
    
    # Prepend to PATH environment variable
    current_path = os.environ.get("PATH", "")
    if path not in current_path:
        os.environ["PATH"] = path + os.pathsep + current_path
    
    # Use Python 3.8+ API for DLL directories
    if hasattr(os, "add_dll_directory"):
        try:
            handle = os.add_dll_directory(path)
            _DLL_DIR_HANDLES.append(handle)
            _log_diag(f"Added DLL directory: {path}")
            return True
        except (FileNotFoundError, OSError) as e:
            _log_diag(f"Failed to add DLL directory {path}: {e}")
            return False
    
    return True


def _setup_dll_directories() -> None:
    """Set up DLL search directories for Windows."""
    if sys.platform != "win32":
        return
        
    import ctypes
    from pathlib import Path
    
    _log_diag(f"Is frozen: {_is_frozen()}")
    _log_diag(f"Base path: {_get_base_path()}")
    
    # Directories to add to DLL search path
    dll_dirs: list[str] = []
    
    if _is_frozen():
        # PyInstaller frozen build
        # Note: sys._MEIPASS points to _internal folder (where files are extracted)
        meipass = sys._MEIPASS
        dll_dirs.append(meipass)
        
        # Look for torch in the frozen bundle (relative to _MEIPASS)
        for torch_location in [
            os.path.join(meipass, "torch", "lib"),
            os.path.join(meipass, "torch", "bin"),
        ]:
            if os.path.isdir(torch_location):
                dll_dirs.append(torch_location)
                _log_diag(f"Found torch DLL directory: {torch_location}")
        
        # Look for NVIDIA CUDA libraries
        nvidia_base = os.path.join(meipass, "nvidia")
        if os.path.isdir(nvidia_base):
            for subdir in os.listdir(nvidia_base):
                for lib_subdir in ["bin", "lib"]:
                    lib_path = os.path.join(nvidia_base, subdir, lib_subdir)
                    if os.path.isdir(lib_path):
                        dll_dirs.append(lib_path)
    else:
        # Running from source - find torch in site-packages
        import importlib.util
        
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_base = torch_spec.submodule_search_locations[0]
            torch_lib = os.path.join(torch_base, "lib")
            torch_bin = os.path.join(torch_base, "bin")
            
            if os.path.isdir(torch_lib):
                dll_dirs.append(torch_lib)
            if os.path.isdir(torch_bin):
                dll_dirs.append(torch_bin)
        
        # Also check for nvidia packages in site-packages
        for p in sys.path:
            if "site-packages" in p:
                nvidia_base = os.path.join(p, "nvidia")
                if os.path.isdir(nvidia_base):
                    for subdir in os.listdir(nvidia_base):
                        for lib_subdir in ["bin", "lib"]:
                            lib_path = os.path.join(nvidia_base, subdir, lib_subdir)
                            if os.path.isdir(lib_path):
                                dll_dirs.append(lib_path)
                break
    
    # Set Windows kernel32 flags FIRST before adding directories
    try:
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        
        # SetDefaultDllDirectories - LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
        if hasattr(kernel32, "SetDefaultDllDirectories"):
            kernel32.SetDefaultDllDirectories(0x00001000)
            _log_diag("Set default DLL directories flag")
        
        # Set the DLL directory to torch/lib for CUDA dependencies
        for dll_dir in dll_dirs:
            if "torch" in dll_dir.lower() and "lib" in dll_dir.lower() and os.path.isdir(dll_dir):
                if hasattr(kernel32, "SetDllDirectoryW"):
                    kernel32.SetDllDirectoryW(dll_dir)
                    _log_diag(f"Set DLL directory to: {dll_dir}")
                break
    except Exception as e:
        _log_diag(f"Failed to set kernel32 DLL flags: {e}")
    
    # Now add all directories to DLL search path
    for dll_dir in dll_dirs:
        _add_dll_directory(dll_dir)


def _test_dll_loading() -> None:
    """Test critical DLL loading and report issues."""
    if sys.platform != "win32":
        return
        
    import ctypes
    from pathlib import Path
    
    if os.environ.get("MARS_TORCH_DIAG") != "1":
        return
    
    # Test loading CRT
    try:
        ctypes.WinDLL("api-ms-win-crt-runtime-l1-1-0.dll")
        _log_diag("CRT runtime DLL: OK")
    except Exception as e:
        _log_diag(f"CRT runtime DLL: FAILED - {e}")
    
    # Find and test torch c10.dll
    if _is_frozen():
        meipass = sys._MEIPASS
        # _MEIPASS is the _internal folder, so torch/lib is directly under it
        c10_paths = [
            os.path.join(meipass, "torch", "lib", "c10.dll"),
        ]
    else:
        import importlib.util
        torch_spec = importlib.util.find_spec("torch")
        c10_paths = []
        if torch_spec and torch_spec.submodule_search_locations:
            c10_paths = [os.path.join(torch_spec.submodule_search_locations[0], "lib", "c10.dll")]
    
    for c10_path in c10_paths:
        if os.path.exists(c10_path):
            try:
                ctypes.WinDLL(c10_path)
                _log_diag(f"c10.dll: OK ({c10_path})")
                break
            except Exception as e:
                _log_diag(f"c10.dll: FAILED - {e}")


def _verify_cuda() -> None:
    """Verify CUDA availability after torch import."""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        _log_diag(f"PyTorch version: {torch.__version__}")
        _log_diag(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            _log_diag(f"CUDA device count: {device_count}")
            for i in range(device_count):
                _log_diag(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        _log_diag(f"CUDA verification failed: {e}")


def initialize() -> bool:
    """
    Initialize PyTorch with proper DLL handling.
    
    Call this at the start of your application before importing torch elsewhere.
    Returns True if torch was successfully imported.
    """
    global _INITIALIZATION_COMPLETE
    
    if _INITIALIZATION_COMPLETE:
        return True
    
    _log_diag("Starting PyTorch initialization...")
    
    if sys.platform == "win32":
        _setup_dll_directories()
        _test_dll_loading()
    
    try:
        import torch
        _INITIALIZATION_COMPLETE = True
        
        if sys.platform == "win32":
            _verify_cuda()
        
        _log_diag("PyTorch initialization complete")
        return True
        
    except ImportError as e:
        warnings.warn(
            f"Failed to import PyTorch: {e}\n"
            "Some GPU acceleration features may not be available.\n"
            "Set MARS_TORCH_DIAG=1 for detailed diagnostics."
        )
        return False
    except Exception as e:
        warnings.warn(f"Unexpected error during PyTorch initialization: {e}")
        return False


# =============================================================================
# Module initialization - runs when this module is imported
# =============================================================================

# ALWAYS set up DLL directories first, before any torch import
if sys.platform == "win32":
    _setup_dll_directories()
    _test_dll_loading()

# Now try to import torch
try:
    import torch
    _INITIALIZATION_COMPLETE = True
    
    if os.environ.get("MARS_TORCH_DIAG") == "1":
        _verify_cuda()
        
except ImportError as e:
    warnings.warn(
        f"PyTorch not available: {e}\n"
        "GPU acceleration will be disabled."
    )
    torch = None
except OSError as e:
    # This catches DLL loading errors
    warnings.warn(
        f"PyTorch DLL loading error: {e}\n"
        "Try running with MARS_TORCH_DIAG=1 for diagnostics.\n"
        "GPU acceleration will be disabled."
    )
    torch = None

__all__ = ["torch", "initialize"]
