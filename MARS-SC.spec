# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MARS-SC: Solution Combination application.

Build commands (from project root, with venv activated):
    pip install pyinstaller
    pyinstaller MARS-SC.spec --clean

The executable will be created in the dist/MARS-SC folder.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get the absolute path to the project directories
SPEC_ROOT = os.path.abspath(SPECPATH)
SRC_DIR = os.path.join(SPEC_ROOT, 'src')

# CRITICAL: Add src directory to sys.path BEFORE PyInstaller analyzes imports
# This allows PyInstaller to find our application modules
sys.path.insert(0, SRC_DIR)

# Now we can try to collect our application submodules
app_hidden_imports = []
for pkg in ['ui', 'core', 'solver', 'file_io', 'utils']:
    try:
        app_hidden_imports += collect_submodules(pkg)
    except Exception as e:
        print(f"Warning: Could not collect submodules for {pkg}: {e}")

# Collect all submodules for complex packages
hiddenimports = [
    # PyQt5 modules
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtWebEngine',
    'PyQt5.QtWebEngineWidgets',
    'PyQt5.sip',
    
    # Visualization
    'pyvista',
    'pyvistaqt',
    'vtk',
    'vtkmodules',
    'vtkmodules.all',
    'vtkmodules.util',
    'vtkmodules.util.numpy_support',
    'vtkmodules.numpy_interface',
    'vtkmodules.numpy_interface.dataset_adapter',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_qt5agg',
    'plotly',
    
    # Data processing
    'numpy',
    'pandas',
    'scipy',
    'scipy.spatial',
    'scipy.interpolate',
    'h5py',
    
    # ANSYS DPF
    'ansys.dpf.core',
    'ansys.dpf',
    
    # gRPC (required by ansys-dpf-core)
    'grpc',
    'grpcio',
    
    # Utilities
    'psutil',
    'imageio',
    'imageio_ffmpeg',
    'PIL',
    'pillow',
    'xlsxwriter',
    'tqdm',
    'lxml',
    'openpyxl',
    
    # Standard library modules that may be needed
    'unittest',
    'unittest.mock',
]

# Add collected application modules
hiddenimports += app_hidden_imports

# Manual fallback list in case collect_submodules didn't work
manual_app_imports = [
    'core',
    'core.data_models',
    'core.computation',
    'core.plasticity',
    'core.visualization',
    'file_io',
    'file_io.combination_parser',
    'file_io.dpf_reader',
    'file_io.exporters',
    'file_io.fea_utilities',
    'file_io.loaders',
    'file_io.validators',
    'solver',
    'solver.stress_engine',
    'solver.nodal_forces_engine',
    'solver.plasticity_engine',
    'ui',
    'ui.application_controller',
    'ui.display_tab',
    'ui.solver_tab',
    'ui.builders',
    'ui.builders.display_ui',
    'ui.builders.solver_ui',
    'ui.dialogs',
    'ui.dialogs.material_profile_dialog',
    'ui.handlers',
    'ui.handlers.file_handler',
    'ui.handlers.analysis_handler',
    'ui.handlers.display_file_handler',
    'ui.handlers.display_visualization_handler',
    'ui.handlers.display_interaction_handler',
    'ui.handlers.display_animation_handler',
    'ui.handlers.display_export_handler',
    'ui.handlers.display_results_handler',
    'ui.handlers.display_state',
    'ui.handlers.display_base_handler',
    'ui.handlers.plotting_handler',
    'ui.handlers.ui_state_handler',
    'ui.handlers.log_handler',
    'ui.handlers.navigator_handler',
    'ui.styles',
    'ui.styles.style_constants',
    'ui.widgets',
    'ui.widgets.collapsible_group',
    'ui.widgets.console',
    'ui.widgets.dialogs',
    'ui.widgets.editable_table',
    'ui.widgets.plotting',
    'utils',
    'utils.constants',
    'utils.file_utils',
    'utils.node_utils',
    'utils.tooltips',
]
hiddenimports += manual_app_imports

# Collect additional submodules dynamically
hiddenimports += collect_submodules('pyvista')
hiddenimports += collect_submodules('vtkmodules')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')

# Collect ANSYS DPF submodules
try:
    hiddenimports += collect_submodules('ansys')
    hiddenimports += collect_submodules('ansys.dpf')
    hiddenimports += collect_submodules('ansys.dpf.core')
    hiddenimports += collect_submodules('ansys.dpf.gate')
except Exception as e:
    print(f"Warning: Could not collect ansys submodules: {e}")

# Remove duplicates
hiddenimports = list(set(hiddenimports))

# Data files to include (non-Python resources)
datas = [
    # CSV data files
    (os.path.join(SRC_DIR, 'youngs_modulus.csv'), '.'),
]

# Add CSV files from ui/handlers
handlers_csv_dir = os.path.join(SRC_DIR, 'ui', 'handlers')
if os.path.exists(handlers_csv_dir):
    for f in os.listdir(handlers_csv_dir):
        if f.endswith('.csv'):
            datas.append((os.path.join(handlers_csv_dir, f), 'ui/handlers'))

# Collect data files from packages
datas += collect_data_files('pyvista')
datas += collect_data_files('vtkmodules')
datas += collect_data_files('matplotlib')

# CRITICAL: Collect ANSYS DPF data files and binaries
# DPF requires its gatebin directory with native binaries
try:
    import ansys.dpf.core as dpf_core
    dpf_path = os.path.dirname(dpf_core.__file__)
    
    # Collect all DPF data files
    datas += collect_data_files('ansys.dpf.core')
    datas += collect_data_files('ansys.dpf.gate')
    
    # Explicitly add gatebin directory if it exists
    gatebin_path = os.path.join(dpf_path, 'gatebin')
    if os.path.exists(gatebin_path):
        # Add entire gatebin directory
        datas.append((gatebin_path, 'ansys/dpf/core/gatebin'))
        print(f"Found DPF gatebin at: {gatebin_path}")
    
    # Also check in ansys.dpf.gate package
    try:
        import ansys.dpf.gate as dpf_gate
        gate_path = os.path.dirname(dpf_gate.__file__)
        gate_gatebin = os.path.join(gate_path, 'gatebin')
        if os.path.exists(gate_gatebin):
            datas.append((gate_gatebin, 'ansys/dpf/gate/gatebin'))
            print(f"Found DPF gate gatebin at: {gate_gatebin}")
    except ImportError:
        pass
    
    # Collect any .dll, .so, .dylib files from ansys packages
    ansys_base = os.path.dirname(os.path.dirname(dpf_path))  # ansys folder
    if os.path.exists(ansys_base):
        for root, dirs, files in os.walk(ansys_base):
            for f in files:
                if f.endswith(('.dll', '.so', '.dylib', '.pyd')):
                    src_file = os.path.join(root, f)
                    # Compute relative destination path
                    rel_path = os.path.relpath(root, os.path.dirname(ansys_base))
                    datas.append((src_file, rel_path))
                    
except ImportError as e:
    print(f"Warning: Could not import ansys.dpf.core: {e}")
except Exception as e:
    print(f"Warning: Error collecting DPF files: {e}")

# Collect gRPC binaries (needed by DPF)
try:
    datas += collect_data_files('grpc')
    datas += collect_data_files('grpcio')
except Exception:
    pass

# Collect binaries from DPF
binaries = []
try:
    import ansys.dpf.core as dpf_core
    dpf_path = os.path.dirname(dpf_core.__file__)
    
    # Look for binaries in gatebin
    gatebin_path = os.path.join(dpf_path, 'gatebin')
    if os.path.exists(gatebin_path):
        for f in os.listdir(gatebin_path):
            if f.endswith(('.dll', '.so', '.dylib')):
                src = os.path.join(gatebin_path, f)
                binaries.append((src, 'ansys/dpf/core/gatebin'))
    
    # Also check gate package
    try:
        import ansys.dpf.gate as dpf_gate
        gate_path = os.path.dirname(dpf_gate.__file__)
        gate_gatebin = os.path.join(gate_path, 'gatebin')
        if os.path.exists(gate_gatebin):
            for f in os.listdir(gate_gatebin):
                if f.endswith(('.dll', '.so', '.dylib')):
                    src = os.path.join(gate_gatebin, f)
                    binaries.append((src, 'ansys/dpf/gate/gatebin'))
    except ImportError:
        pass
except Exception as e:
    print(f"Warning: Could not collect DPF binaries: {e}")

a = Analysis(
    [os.path.join(SPEC_ROOT, 'mars_sc_entry.py')],
    pathex=[SRC_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPEC_ROOT, 'pyinstaller_runtime_hook.py')],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MARS-SC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep True for debugging, set False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='resources/icon.ico',  # Uncomment if you add an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MARS-SC',
)
