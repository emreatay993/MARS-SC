# FILE_INDEX

Generated: `2026-02-12 22:57:16`

## Scope

- Source root: `src/`
- Python modules indexed: `55`
- Total Python lines (physical): `21198`
- Line counts include blank lines and comments.
- Descriptions come from each module's top docstring (first sentence when possible).

## Package Totals

| Package | Modules | Lines |
| --- | ---: | ---: |
| `(root)` | 2 | 47 |
| `core` | 5 | 1157 |
| `file_io` | 7 | 4339 |
| `solver` | 5 | 3001 |
| `ui` | 30 | 11900 |
| `utils` | 6 | 754 |

## Module Index

| Path | Module | Lines | Classes | Functions | Description |
| --- | --- | ---: | ---: | ---: | --- |
| `src/__init__.py` | `src.__init__` | 7 | 0 | 0 | MARS-SC: Solution Combination Modernised successor to the legacy MSUP Smart Solver for transient structural analysis. |
| `src/core/__init__.py` | `src.core.__init__` | 2 | 0 | 0 | Core business logic and computation modules. |
| `src/core/computation.py` | `src.core.computation` | 37 | 1 | 0 | Legacy analysis engine module - NOT USED in MARS-SC. |
| `src/core/data_models.py` | `src.core.data_models` | 679 | 14 | 0 | Data models for MARS-SC (Solution Combination). |
| `src/core/plasticity.py` | `src.core.plasticity` | 238 | 1 | 7 | Utilities for preparing plasticity solver inputs. |
| `src/core/visualization.py` | `src.core.visualization` | 201 | 2 | 0 | Visualization management classes for MARS-SC (Solution Combination). |
| `src/file_io/__init__.py` | `src.file_io.__init__` | 43 | 0 | 0 | File I/O operations for loading and exporting data. |
| `src/file_io/combination_parser.py` | `src.file_io.combination_parser` | 392 | 2 | 0 | Combination Table Parser for MARS-SC (Solution Combination). |
| `src/file_io/dpf_reader.py` | `src.file_io.dpf_reader` | 1762 | 5 | 9 | DPF Analysis Reader for MARS-SC (Solution Combination). |
| `src/file_io/exporters.py` | `src.file_io.exporters` | 983 | 0 | 19 | Export helpers for MARS-SC (Solution Combination). |
| `src/file_io/fea_utilities.py` | `src.file_io.fea_utilities` | 41 | 0 | 1 | No module docstring. |
| `src/file_io/loaders.py` | `src.file_io.loaders` | 777 | 0 | 16 | File loading helpers for MARS-SC (Solution Combination). |
| `src/file_io/validators.py` | `src.file_io.validators` | 341 | 1 | 6 | File validation helpers for MARS-SC (Solution Combination). |
| `src/main.py` | `src.main` | 40 | 0 | 1 | Entry point for the MARS-SC: Solution Combination application. |
| `src/solver/__init__.py` | `src.solver.__init__` | 6 | 0 | 0 | Solver engines for MARS-SC solution combination analysis. |
| `src/solver/combination_engine.py` | `src.solver.combination_engine` | 1098 | 1 | 0 | Combination Engine for MARS-SC (Solution Combination). |
| `src/solver/deformation_engine.py` | `src.solver.deformation_engine` | 555 | 2 | 0 | Deformation (Displacement) Combination Engine for MARS-SC (Solution Combination). |
| `src/solver/nodal_forces_engine.py` | `src.solver.nodal_forces_engine` | 486 | 1 | 0 | Nodal Forces Combination Engine for MARS-SC (Solution Combination). |
| `src/solver/plasticity_engine.py` | `src.solver.plasticity_engine` | 856 | 1 | 27 | Plasticity correction solvers for Neuber, Glinka, and IBG methods. |
| `src/ui/__init__.py` | `src.ui.__init__` | 2 | 0 | 0 | GUI components and user interface modules. |
| `src/ui/application_controller.py` | `src.ui.application_controller` | 209 | 1 | 0 | Main window for the MARS-SC: Solution Combination application. |
| `src/ui/builders/__init__.py` | `src.ui.builders.__init__` | 2 | 0 | 0 | UI builders for constructing complex widget layouts. |
| `src/ui/builders/display_ui.py` | `src.ui.builders.display_ui` | 324 | 1 | 0 | Builds the Display tab UI: 3D view controls, result dropdowns, export buttons, etc. |
| `src/ui/builders/solver_ui.py` | `src.ui.builders.solver_ui` | 666 | 2 | 0 | Builds the Solver tab UI: RST inputs, combination table, output options, plasticity, console. |
| `src/ui/dialogs/__init__.py` | `src.ui.dialogs.__init__` | 5 | 0 | 0 | Dialog components for the UI package. |
| `src/ui/dialogs/material_profile_dialog.py` | `src.ui.dialogs.material_profile_dialog` | 470 | 1 | 0 | Material profile dialog providing editors for temperature-dependent properties. |
| `src/ui/display_tab.py` | `src.ui.display_tab` | 1326 | 1 | 0 | Display tab: 3D visualization of FEA results (PyVista). |
| `src/ui/handlers/analysis_handler.py` | `src.ui.handlers.analysis_handler` | 1306 | 1 | 0 | Runs combination analysis from the Solver tab: validates inputs, builds SolverConfig, calls CombinationEngine, and shows results. |
| `src/ui/handlers/display_base_handler.py` | `src.ui.handlers.display_base_handler` | 26 | 1 | 0 | Base utilities for Display tab handler classes. |
| `src/ui/handlers/display_export_handler.py` | `src.ui.handlers.display_export_handler` | 755 | 1 | 0 | Export-related functionality for the Display tab. |
| `src/ui/handlers/display_file_handler.py` | `src.ui.handlers.display_file_handler` | 261 | 1 | 0 | File loading logic for the Display tab. |
| `src/ui/handlers/display_interaction_handler.py` | `src.ui.handlers.display_interaction_handler` | 712 | 1 | 0 | Node interaction, picking, and hotspot analysis for the Display tab. |
| `src/ui/handlers/display_results_handler.py` | `src.ui.handlers.display_results_handler` | 111 | 1 | 0 | Helpers for applying solver output datasets to the Display tab. |
| `src/ui/handlers/display_state.py` | `src.ui.handlers.display_state` | 43 | 1 | 0 | Shared state container for the Display tab. |
| `src/ui/handlers/display_visualization_handler.py` | `src.ui.handlers.display_visualization_handler` | 548 | 1 | 0 | Visualization updates and rendering helpers for the Display tab. |
| `src/ui/handlers/file_handler.py` | `src.ui.handlers.file_handler` | 368 | 2 | 0 | Solver tab file I/O: file dialogs, loading RST via DPF, import/export of combination tables, and updating the tab state/UI. |
| `src/ui/handlers/log_handler.py` | `src.ui.handlers.log_handler` | 113 | 1 | 0 | Formats and writes log lines to the Solver tab console. |
| `src/ui/handlers/navigator_handler.py` | `src.ui.handlers.navigator_handler` | 55 | 1 | 0 | Handles user interactions with the File Navigator dock, such as selecting project directories and opening files. |
| `src/ui/handlers/plotting_handler.py` | `src.ui.handlers.plotting_handler` | 63 | 1 | 0 | Handles plotting operations, such as loading Plotly figures into WebViews and managing temporary files. |
| `src/ui/handlers/ui_state_handler.py` | `src.ui.handlers.ui_state_handler` | 399 | 1 | 0 | Tracks Solver tab UI state: whats enabled/disabled, visible/hidden, based on user input. |
| `src/ui/solver_tab.py` | `src.ui.solver_tab` | 1443 | 3 | 0 | Solver tab: combine stress (and optionally forces/deformation) from two static RST files via linear combination coefficients. |
| `src/ui/styles/__init__.py` | `src.ui.styles.__init__` | 1 | 0 | 0 | Styling constants for the app; applied via setStyleSheet(). |
| `src/ui/styles/style_constants.py` | `src.ui.styles.style_constants` | 366 | 0 | 0 | Qt stylesheet strings for MARS-SC; applied via setStyleSheet(). |
| `src/ui/widgets/__init__.py` | `src.ui.widgets.__init__` | 6 | 0 | 0 | Reusable UI widgets. |
| `src/ui/widgets/collapsible_group.py` | `src.ui.widgets.collapsible_group` | 235 | 2 | 0 | Collapsible Group Box Widget for MARS-SC. |
| `src/ui/widgets/console.py` | `src.ui.widgets.console` | 64 | 1 | 0 | Console logger widget for MARS-SC (Solution Combination). |
| `src/ui/widgets/dialogs.py` | `src.ui.widgets.dialogs` | 97 | 1 | 0 | Dialog widgets for MARS-SC (Solution Combination). |
| `src/ui/widgets/editable_table.py` | `src.ui.widgets.editable_table` | 269 | 1 | 0 | Generic editable table widget with copy/paste helpers and trailing blank row. |
| `src/ui/widgets/plotting.py` | `src.ui.widgets.plotting` | 1655 | 3 | 0 | Plotting widgets for MARS-SC (Solution Combination). |
| `src/utils/__init__.py` | `src.utils.__init__` | 2 | 0 | 0 | Utility functions and helpers. |
| `src/utils/constants.py` | `src.utils.constants` | 43 | 0 | 0 | Global constants, configuration settings, and UI styles for MARS-SC (Solution Combination). |
| `src/utils/file_utils.py` | `src.utils.file_utils` | 235 | 0 | 2 | File utility helpers for MARS-SC (Solution Combination). |
| `src/utils/node_utils.py` | `src.utils.node_utils` | 26 | 0 | 1 | Node-related utility helpers for MARS-SC (Solution Combination). |
| `src/utils/tooltips.py` | `src.utils.tooltips` | 153 | 0 | 0 | Tooltip text constants for MARS-SC (Solution Combination). |
| `src/utils/torch_setup.py` | `src.utils.torch_setup` | 295 | 0 | 8 | PyTorch initialization module for Windows CUDA compatibility. |
