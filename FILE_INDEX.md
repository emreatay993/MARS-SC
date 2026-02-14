# FILE_INDEX

Generated: `2026-02-14 23:56:49`

## Scope

- Source root: `src/`
- Python modules indexed: `64`
- Total Python lines (physical): `20423`
- Line counts include blank lines and comments.
- Descriptions come from each module's top docstring (first sentence when possible).

## Package Totals

| Package | Modules | Lines |
| --- | ---: | ---: |
| `(root)` | 2 | 47 |
| `core` | 4 | 1001 |
| `file_io` | 5 | 3512 |
| `solver` | 5 | 3001 |
| `ui` | 45 | 12309 |
| `utils` | 3 | 553 |

## Module Index

| Path | Module | Lines | Classes | Functions | Description |
| --- | --- | ---: | ---: | ---: | --- |
| `src/__init__.py` | `src.__init__` | 7 | 0 | 0 | MARS-SC: Solution Combination Modernised successor to the legacy MSUP Smart Solver for transient structural analysis. |
| `src/core/__init__.py` | `src.core.__init__` | 2 | 0 | 0 | Core business logic and computation modules. |
| `src/core/data_models.py` | `src.core.data_models` | 560 | 10 | 0 | Data models for MARS-SC (Solution Combination). |
| `src/core/plasticity.py` | `src.core.plasticity` | 238 | 1 | 7 | Utilities for preparing plasticity solver inputs. |
| `src/core/visualization.py` | `src.core.visualization` | 201 | 2 | 0 | Visualization management classes for MARS-SC (Solution Combination). |
| `src/file_io/__init__.py` | `src.file_io.__init__` | 43 | 0 | 0 | File I/O operations for loading and exporting data. |
| `src/file_io/combination_parser.py` | `src.file_io.combination_parser` | 392 | 2 | 0 | Combination Table Parser for MARS-SC (Solution Combination). |
| `src/file_io/dpf_reader.py` | `src.file_io.dpf_reader` | 1807 | 5 | 9 | DPF Analysis Reader for MARS-SC (Solution Combination). |
| `src/file_io/exporters.py` | `src.file_io.exporters` | 1071 | 0 | 19 | Export helpers for MARS-SC (Solution Combination). |
| `src/file_io/loaders.py` | `src.file_io.loaders` | 199 | 0 | 5 | Runtime file loading helpers for MARS-SC. |
| `src/main.py` | `src.main` | 40 | 0 | 1 | Entry point for the MARS-SC: Solution Combination application. |
| `src/solver/__init__.py` | `src.solver.__init__` | 6 | 0 | 0 | Solver engines for MARS-SC solution combination analysis. |
| `src/solver/deformation_engine.py` | `src.solver.deformation_engine` | 555 | 2 | 0 | Deformation (Displacement) Combination Engine for MARS-SC (Solution Combination). |
| `src/solver/nodal_forces_engine.py` | `src.solver.nodal_forces_engine` | 486 | 1 | 0 | Nodal Forces Combination Engine for MARS-SC (Solution Combination). |
| `src/solver/plasticity_engine.py` | `src.solver.plasticity_engine` | 856 | 1 | 27 | Plasticity correction solvers for Neuber, Glinka, and IBG methods. |
| `src/solver/stress_engine.py` | `src.solver.stress_engine` | 1098 | 1 | 0 | Stress Combination Engine for MARS-SC (Solution Combination). |
| `src/ui/__init__.py` | `src.ui.__init__` | 2 | 0 | 0 | GUI components and user interface modules. |
| `src/ui/application_controller.py` | `src.ui.application_controller` | 278 | 2 | 0 | Main window for the MARS-SC: Solution Combination application. |
| `src/ui/builders/__init__.py` | `src.ui.builders.__init__` | 2 | 0 | 0 | UI builders for constructing complex widget layouts. |
| `src/ui/builders/display_ui.py` | `src.ui.builders.display_ui` | 343 | 1 | 0 | Builds the Display tab UI: 3D view controls, result dropdowns, export buttons, etc. |
| `src/ui/builders/solver_ui.py` | `src.ui.builders.solver_ui` | 662 | 2 | 0 | Builds the Solver tab UI: RST inputs, combination table, output options, plasticity, console. |
| `src/ui/dialogs/__init__.py` | `src.ui.dialogs.__init__` | 5 | 0 | 0 | Dialog components for the UI package. |
| `src/ui/dialogs/material_profile_dialog.py` | `src.ui.dialogs.material_profile_dialog` | 518 | 1 | 0 | Material profile dialog providing editors for temperature-dependent properties. |
| `src/ui/display_payload.py` | `src.ui.display_payload` | 34 | 2 | 0 | Typed payloads for solver-to-display communication. |
| `src/ui/display_tab.py` | `src.ui.display_tab` | 900 | 1 | 0 | Display tab: 3D visualization of FEA results (PyVista). |
| `src/ui/handlers/display_base_handler.py` | `src.ui.handlers.display_base_handler` | 26 | 1 | 0 | Base utilities for Display tab handler classes. |
| `src/ui/handlers/display_contour_context.py` | `src.ui.handlers.display_contour_context` | 144 | 0 | 0 | No module docstring. |
| `src/ui/handlers/display_contour_policy.py` | `src.ui.handlers.display_contour_policy` | 77 | 0 | 0 | No module docstring. |
| `src/ui/handlers/display_contour_sync_handler.py` | `src.ui.handlers.display_contour_sync_handler` | 474 | 0 | 0 | No module docstring. |
| `src/ui/handlers/display_contour_types.py` | `src.ui.handlers.display_contour_types` | 9 | 0 | 0 | No module docstring. |
| `src/ui/handlers/display_export_handler.py` | `src.ui.handlers.display_export_handler` | 789 | 1 | 0 | Export-related functionality for the Display tab. |
| `src/ui/handlers/display_file_handler.py` | `src.ui.handlers.display_file_handler` | 272 | 1 | 0 | File loading logic for the Display tab. |
| `src/ui/handlers/display_interaction_handler.py` | `src.ui.handlers.display_interaction_handler` | 712 | 1 | 0 | Node interaction, picking, and hotspot analysis for the Display tab. |
| `src/ui/handlers/display_mesh_arrays.py` | `src.ui.handlers.display_mesh_arrays` | 167 | 0 | 0 | No module docstring. |
| `src/ui/handlers/display_results_handler.py` | `src.ui.handlers.display_results_handler` | 115 | 1 | 0 | Helpers for applying solver output datasets to the Display tab. |
| `src/ui/handlers/display_state.py` | `src.ui.handlers.display_state` | 44 | 1 | 0 | Shared state container for the Display tab. |
| `src/ui/handlers/display_visualization_handler.py` | `src.ui.handlers.display_visualization_handler` | 628 | 1 | 0 | Visualization updates and rendering helpers for the Display tab. |
| `src/ui/handlers/file_handler.py` | `src.ui.handlers.file_handler` | 368 | 2 | 0 | Solver tab file I/O: file dialogs, loading RST via DPF, import/export of combination tables, and updating the tab state/UI. |
| `src/ui/handlers/log_handler.py` | `src.ui.handlers.log_handler` | 113 | 1 | 0 | Formats and writes log lines to the Solver tab console. |
| `src/ui/handlers/navigator_handler.py` | `src.ui.handlers.navigator_handler` | 55 | 1 | 0 | Handles user interactions with the File Navigator dock, such as selecting project directories and opening files. |
| `src/ui/handlers/plotting_handler.py` | `src.ui.handlers.plotting_handler` | 63 | 1 | 0 | Handles plotting operations, such as loading Plotly figures into WebViews and managing temporary files. |
| `src/ui/handlers/solve_run_controller.py` | `src.ui.handlers.solve_run_controller` | 124 | 1 | 0 | Orchestration for solver-tab analysis runs. |
| `src/ui/handlers/solver_combination_table_handler.py` | `src.ui.handlers.solver_combination_table_handler` | 224 | 3 | 0 | Combination-table handling for the solver tab. |
| `src/ui/handlers/solver_engine_factory.py` | `src.ui.handlers.solver_engine_factory` | 76 | 1 | 0 | Engine creation helpers for solver analysis. |
| `src/ui/handlers/solver_input_validator.py` | `src.ui.handlers.solver_input_validator` | 190 | 1 | 0 | Input validation helpers for Solver analysis runs. |
| `src/ui/handlers/solver_named_selection_handler.py` | `src.ui.handlers.solver_named_selection_handler` | 129 | 1 | 0 | Named-selection and scoping handling for the solver tab. |
| `src/ui/handlers/solver_output_availability.py` | `src.ui.handlers.solver_output_availability` | 52 | 1 | 1 | Shared output-availability policy for solver-tab analyses. |
| `src/ui/handlers/solver_output_state_handler.py` | `src.ui.handlers.solver_output_state_handler` | 138 | 1 | 0 | Output-selection and UI-state handling for the solver tab. |
| `src/ui/handlers/solver_result_payload_handler.py` | `src.ui.handlers.solver_result_payload_handler` | 251 | 1 | 0 | Display/presentation handling for solver outputs. |
| `src/ui/handlers/solver_result_summary_handler.py` | `src.ui.handlers.solver_result_summary_handler` | 494 | 1 | 0 | UI result handling for analysis runs. |
| `src/ui/handlers/solver_run_execution_handler.py` | `src.ui.handlers.solver_run_execution_handler` | 294 | 2 | 0 | Execution concern for solver analysis runs. |
| `src/ui/handlers/solver_run_lifecycle_handler.py` | `src.ui.handlers.solver_run_lifecycle_handler` | 215 | 1 | 0 | Lifecycle/UI concern for solver analysis runs. |
| `src/ui/solver_tab.py` | `src.ui.solver_tab` | 700 | 1 | 0 | Solver tab: combine stress (and optionally forces/deformation) from two static RST files via linear combination coefficients. |
| `src/ui/styles/__init__.py` | `src.ui.styles.__init__` | 1 | 0 | 0 | Styling constants for the app; applied via setStyleSheet(). |
| `src/ui/styles/style_constants.py` | `src.ui.styles.style_constants` | 380 | 0 | 0 | Qt stylesheet strings for MARS-SC; applied via setStyleSheet(). |
| `src/ui/widgets/__init__.py` | `src.ui.widgets.__init__` | 6 | 0 | 0 | Reusable UI widgets. |
| `src/ui/widgets/collapsible_group.py` | `src.ui.widgets.collapsible_group` | 235 | 2 | 0 | Collapsible Group Box Widget for MARS-SC. |
| `src/ui/widgets/console.py` | `src.ui.widgets.console` | 64 | 1 | 0 | Console logger widget for MARS-SC (Solution Combination). |
| `src/ui/widgets/dialogs.py` | `src.ui.widgets.dialogs` | 97 | 1 | 0 | Dialog widgets for MARS-SC (Solution Combination). |
| `src/ui/widgets/editable_table.py` | `src.ui.widgets.editable_table` | 269 | 1 | 0 | Generic editable table widget with copy/paste helpers and trailing blank row. |
| `src/ui/widgets/plotting.py` | `src.ui.widgets.plotting` | 1570 | 2 | 0 | Plotting widgets for MARS-SC (Solution Combination). |
| `src/utils/__init__.py` | `src.utils.__init__` | 2 | 0 | 0 | Utility functions and helpers. |
| `src/utils/constants.py` | `src.utils.constants` | 43 | 0 | 0 | Global constants, configuration settings, and UI styles for MARS-SC (Solution Combination). |
| `src/utils/tooltips.py` | `src.utils.tooltips` | 508 | 0 | 0 | Tooltip text constants for MARS-SC (Solution Combination). |
