# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MARS-SC: Solution Combination application.

Build command:
    pyinstaller MARS-SC.spec

The executable will be created in the dist/MARS-SC folder.
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

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
    
    # Application modules
    'src',
    'src.core',
    'src.core.data_models',
    'src.core.computation',
    'src.core.plasticity',
    'src.core.visualization',
    'src.file_io',
    'src.file_io.combination_parser',
    'src.file_io.dpf_reader',
    'src.file_io.exporters',
    'src.file_io.fea_utilities',
    'src.file_io.loaders',
    'src.file_io.validators',
    'src.solver',
    'src.solver.combination_engine',
    'src.solver.nodal_forces_engine',
    'src.solver.plasticity_engine',
    'src.ui',
    'src.ui.application_controller',
    'src.ui.display_tab',
    'src.ui.solver_tab',
    'src.ui.builders',
    'src.ui.builders.display_ui',
    'src.ui.builders.solver_ui',
    'src.ui.dialogs',
    'src.ui.dialogs.material_profile_dialog',
    'src.ui.handlers',
    'src.ui.handlers.file_handler',
    'src.ui.handlers.analysis_handler',
    'src.ui.handlers.display_file_handler',
    'src.ui.handlers.display_visualization_handler',
    'src.ui.handlers.display_interaction_handler',
    'src.ui.handlers.display_animation_handler',
    'src.ui.handlers.display_export_handler',
    'src.ui.handlers.display_results_handler',
    'src.ui.handlers.display_state',
    'src.ui.handlers.display_base_handler',
    'src.ui.handlers.plotting_handler',
    'src.ui.handlers.ui_state_handler',
    'src.ui.handlers.log_handler',
    'src.ui.handlers.navigator_handler',
    'src.ui.styles',
    'src.ui.styles.style_constants',
    'src.ui.widgets',
    'src.ui.widgets.collapsible_group',
    'src.ui.widgets.console',
    'src.ui.widgets.dialogs',
    'src.ui.widgets.editable_table',
    'src.ui.widgets.plotting',
    'src.utils',
    'src.utils.constants',
    'src.utils.file_utils',
    'src.utils.node_utils',
    'src.utils.tooltips',
    'src.utils.torch_setup',
]

# Collect additional submodules dynamically
hiddenimports += collect_submodules('pyvista')
hiddenimports += collect_submodules('vtkmodules')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')

# Data files to include
datas = [
    # CSV data files in src
    ('src/youngs_modulus.csv', 'src'),
    # CSV files in handlers
    ('src/ui/handlers/*.csv', 'src/ui/handlers'),
]

# Collect data files from packages
datas += collect_data_files('pyvista')
datas += collect_data_files('vtkmodules')
datas += collect_data_files('matplotlib')

a = Analysis(
    ['src/main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'unittest',
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
    console=False,  # Set to True for debugging, False for release
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
