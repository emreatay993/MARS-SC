"""
Analysis Handler for the SolverTab (MARS-SC Solution Combination).

This class encapsulates all logic related to:
1. Validating UI inputs for combination analysis.
2. Building the SolverConfig.
3. Executing the combination analysis via the CombinationEngine.
4. Handling and displaying the results.

Note: DPF operations are run in the main thread to avoid gRPC threading issues.
QApplication.processEvents() is used to keep the GUI responsive during computation.
"""

import os
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QMessageBox, QApplication

from solver.combination_engine import CombinationEngine
from solver.nodal_forces_engine import NodalForcesCombinationEngine
from solver.deformation_engine import DeformationCombinationEngine, CylindricalCSNotFoundError
from file_io.dpf_reader import (
    DPFAnalysisReader, BeamElementNotSupportedError, 
    NodalForcesNotAvailableError, DisplacementNotAvailableError
)
from core.data_models import (
    SolverConfig, CombinationResult, CombinationTableData, PlasticityConfig, 
    NodalForcesResult, DeformationResult
)


class SolverAnalysisHandler:
    """Manages the combination analysis execution and result handling for the SolverTab."""

    # Memory threshold in GB to switch to chunked processing
    MEMORY_THRESHOLD_GB = 2.0

    def __init__(self, tab):
        """
        Initialize the analysis handler.

        Args:
            tab (SolverTab): The parent SolverTab instance.
        """
        self.tab = tab
        self._solve_start_time: Optional[datetime] = None
        self._combination_engine: Optional[CombinationEngine] = None
        self._nodal_forces_engine: Optional[NodalForcesCombinationEngine] = None
        self._deformation_engine: Optional[DeformationCombinationEngine] = None

    def solve(self, config: Optional[SolverConfig] = None):
        """
        Run combination analysis in the main thread with GUI updates.
        
        Note: DPF operations must run in the main thread due to gRPC threading
        constraints. We use QApplication.processEvents() to keep the GUI responsive.
        
        Args:
            config: Optional SolverConfig. If None, builds from UI.
        """
        # Use provided config or build from UI
        if config is None:
            config = self.tab._build_solver_config()
        
        # Validate inputs
        if not self._validate_inputs(config):
            return
        
        # Determine stress type to compute (can be None if only nodal forces are requested)
        stress_type = self._get_stress_type(config)
        
        # Check if at least one output is selected
        if stress_type is None and not config.calculate_nodal_forces and not config.calculate_deformation:
            QMessageBox.warning(
                self.tab, "No Output Selected",
                "Please select at least one output type (stress, nodal forces, or deformation)."
            )
            return
        
        # Log solve start
        self._log_solve_start(config)
        
        # Show progress bar
        self.tab.progress_bar.setVisible(True)
        self.tab.progress_bar.setValue(0)
        self.tab.console_textbox.append("Running combination analysis...\n")
        QApplication.processEvents()  # Update GUI
        
        stress_result = None
        forces_result = None
        deformation_result = None
        
        try:
            # Run stress analysis if any stress type is selected
            if stress_type is not None:
                # Create combination engine for stress
                engine = self._create_combination_engine(config)
                if engine is None:
                    self.tab.progress_bar.setVisible(False)
                    return
                
                self._combination_engine = engine
                
                # Check if we should force chunked processing based on memory
                use_chunked = self._should_use_chunked_processing(engine)
                
                # Run stress analysis
                stress_result = self._run_analysis(engine, config, stress_type, use_chunked)
            
            # Run nodal forces analysis if requested
            if config.calculate_nodal_forces:
                self.tab.console_textbox.append("\nRunning nodal forces combination...\n")
                QApplication.processEvents()
                
                forces_result = self._run_nodal_forces_analysis(config)
            
            # Run deformation analysis if requested
            if config.calculate_deformation:
                self.tab.console_textbox.append("\nRunning deformation combination...\n")
                QApplication.processEvents()
                
                deformation_result = self._run_deformation_analysis(config)
            
            # Handle results
            if stress_result is not None or forces_result is not None or deformation_result is not None:
                self._on_solve_complete(stress_result, config, forces_result, deformation_result)
            
        except NodalForcesNotAvailableError as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = (
                f"Nodal Forces Not Available\n\n"
                f"{str(e)}\n\n"
                f"Ensure 'Write element nodal forces' is enabled in ANSYS Output Controls."
            )
            QMessageBox.critical(self.tab, "Nodal Forces Error", error_msg)
            
        except DisplacementNotAvailableError as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = (
                f"Displacement Results Not Available\n\n"
                f"{str(e)}\n\n"
                f"Ensure displacement output is enabled in ANSYS Output Controls."
            )
            QMessageBox.critical(self.tab, "Displacement Error", error_msg)
        
        except CylindricalCSNotFoundError as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = (
                f"Cylindrical Coordinate System Error\n\n"
                f"{str(e)}"
            )
            QMessageBox.critical(self.tab, "Coordinate System Error", error_msg)
            
        except MemoryError as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = (
                f"Out of Memory Error\n\n"
                f"The analysis requires more RAM than available.\n\n"
                f"Suggestions:\n"
                f"• Use a smaller Named Selection to reduce nodes\n"
                f"• Close other applications to free memory\n\n"
                f"Details: {str(e)}"
            )
            QMessageBox.critical(self.tab, "Memory Error", error_msg)
            
        except Exception as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self._on_solve_error(error_msg)
    
    def _run_analysis(
        self, 
        engine: CombinationEngine, 
        config: SolverConfig, 
        stress_type: str,
        use_chunked: bool
    ) -> Optional[CombinationResult]:
        """
        Run the actual analysis computation in main thread.
        
        Args:
            engine: Configured CombinationEngine.
            config: SolverConfig with analysis settings.
            stress_type: Type of stress to compute.
            use_chunked: Whether to use chunked processing.
            
        Returns:
            CombinationResult or None if failed.
        """
        # Progress callback that updates GUI
        def on_progress(current, total, message):
            if total > 0:
                percent = int((current / total) * 100)
                self.tab.progress_bar.setValue(percent)
                self.tab.progress_bar.setFormat(f"{message} ({percent}%)")
            else:
                self.tab.progress_bar.setFormat(message)
            QApplication.processEvents()  # Keep GUI responsive
        
        # Check memory availability
        on_progress(0, 100, "Estimating memory requirements...")
        try:
            is_sufficient, estimates = engine.check_memory_available(raise_on_insufficient=False)
            if not is_sufficient:
                self.tab.console_textbox.append(
                    f"\n⚠ Memory Warning: Limited RAM detected.\n"
                    f"  Available: {estimates['available_bytes'] / 1e9:.2f} GB\n"
                    f"  Using memory-efficient processing.\n"
                )
                QApplication.processEvents()
        except Exception as mem_err:
            on_progress(0, 100, f"Memory check skipped: {mem_err}")
        
        # Compute results based on mode
        if config.combination_history_mode and config.selected_node_id:
            # Single node history mode - use optimized single-node method
            # This only loads stress data for the single node, avoiding full preload
            on_progress(0, 100, f"Computing history for node {config.selected_node_id}...")
            
            combo_indices, stress_values = engine.compute_single_node_history_fast(
                config.selected_node_id,
                stress_type,
                progress_callback=on_progress
            )
            
            # Create result object for history mode
            result = CombinationResult(
                node_ids=np.array([config.selected_node_id]),
                node_coords=np.array([[0, 0, 0]]),  # Placeholder
                result_type=stress_type,
                all_combo_results=stress_values.reshape(1, -1).T,
            )
            result.metadata = {
                'mode': 'history',
                'node_id': config.selected_node_id,
                'combination_indices': combo_indices,
                'stress_values': stress_values,
            }
        else:
            # Full envelope analysis
            on_progress(5, 100, "Starting envelope analysis...")
            
            if use_chunked:
                result = engine.compute_full_analysis_chunked(
                    stress_type=stress_type,
                    progress_callback=on_progress
                )
            else:
                result = engine.compute_full_analysis_auto(
                    stress_type=stress_type,
                    progress_callback=on_progress,
                    memory_threshold_gb=self.MEMORY_THRESHOLD_GB
                )
        
        on_progress(100, 100, "Complete")
        return result
    
    def _run_nodal_forces_analysis(self, config: SolverConfig) -> Optional[NodalForcesResult]:
        """
        Run nodal forces combination analysis.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            NodalForcesResult or None if failed.
        """
        # Progress callback that updates GUI
        def on_progress(current, total, message):
            if total > 0:
                percent = int((current / total) * 100)
                self.tab.progress_bar.setValue(percent)
                self.tab.progress_bar.setFormat(f"{message} ({percent}%)")
            else:
                self.tab.progress_bar.setFormat(message)
            QApplication.processEvents()  # Keep GUI responsive
        
        try:
            # Create nodal forces engine
            engine = self._create_nodal_forces_engine(config)
            if engine is None:
                return None
            
            self._nodal_forces_engine = engine
            
            # Validate nodal forces availability
            on_progress(0, 100, "Validating nodal forces availability...")
            is_valid, error_msg = engine.validate_nodal_forces_availability()
            if not is_valid:
                raise NodalForcesNotAvailableError(error_msg)
            
            # Preload force data
            on_progress(5, 100, "Loading nodal forces from RST files...")
            engine.preload_force_data(progress_callback=on_progress)
            
            # Compute results based on mode
            if config.combination_history_mode and config.selected_node_id:
                # Single node history mode
                on_progress(50, 100, f"Computing force history for node {config.selected_node_id}...")
                combo_indices, fx, fy, fz, magnitude = engine.compute_single_node_history(
                    config.selected_node_id
                )
                
                # Create result object for history mode
                result = NodalForcesResult(
                    node_ids=np.array([config.selected_node_id]),
                    node_coords=np.array([[0, 0, 0]]),  # Placeholder
                    all_combo_fx=fx.reshape(-1, 1),
                    all_combo_fy=fy.reshape(-1, 1),
                    all_combo_fz=fz.reshape(-1, 1),
                    force_unit=engine.force_unit,
                )
                result.metadata = {
                    'mode': 'history',
                    'node_id': config.selected_node_id,
                    'combination_indices': combo_indices,
                    'fx': fx,
                    'fy': fy,
                    'fz': fz,
                    'magnitude': magnitude,
                }
            else:
                # Full envelope analysis
                on_progress(10, 100, "Computing nodal forces for all combinations...")
                result = engine.compute_full_analysis(progress_callback=on_progress)
            
            on_progress(100, 100, "Nodal forces complete")
            return result
            
        except NodalForcesNotAvailableError:
            raise
        except Exception as e:
            self.tab.console_textbox.append(f"\nError in nodal forces analysis: {e}\n")
            raise
    
    def _create_nodal_forces_engine(self, config: SolverConfig) -> Optional[NodalForcesCombinationEngine]:
        """
        Create and configure the NodalForcesCombinationEngine.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            Configured NodalForcesCombinationEngine or None if creation fails.
        """
        try:
            # Get readers from file handler
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            
            if reader1 is None or reader2 is None:
                raise ValueError("DPF readers not available. Please reload RST files.")
            
            # Get nodal scoping from named selection
            ns_name = self.tab.get_selected_named_selection()
            nodal_scoping = reader1.get_nodal_scoping_from_named_selection(ns_name)
            
            # Get combination table
            combo_table = self.tab.get_combination_table_data()
            
            # Create engine with coordinate system option
            engine = NodalForcesCombinationEngine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combination_table=combo_table,
                rotate_to_global=config.nodal_forces_rotate_to_global
            )
            
            return engine
            
        except Exception as e:
            QMessageBox.critical(
                self.tab, "Engine Creation Error",
                f"Failed to create nodal forces engine:\n\n{e}"
            )
            return None
    
    def _run_deformation_analysis(self, config: SolverConfig) -> Optional[DeformationResult]:
        """
        Run deformation (displacement) combination analysis.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            DeformationResult or None if failed.
        """
        # Progress callback that updates GUI
        def on_progress(current, total, message):
            if total > 0:
                percent = int((current / total) * 100)
                self.tab.progress_bar.setValue(percent)
                self.tab.progress_bar.setFormat(f"{message} ({percent}%)")
            else:
                self.tab.progress_bar.setFormat(message)
            QApplication.processEvents()  # Keep GUI responsive
        
        try:
            # Create deformation engine
            engine = self._create_deformation_engine(config)
            if engine is None:
                return None
            
            self._deformation_engine = engine
            
            # Validate displacement availability
            on_progress(0, 100, "Validating displacement availability...")
            is_valid, error_msg = engine.validate_displacement_availability()
            if not is_valid:
                raise DisplacementNotAvailableError(error_msg)
            
            # Validate cylindrical coordinate system if specified
            if engine.uses_cylindrical_cs:
                on_progress(2, 100, f"Validating coordinate system {config.deformation_cylindrical_cs_id}...")
                is_valid, error_msg = engine.validate_cylindrical_cs()
                if not is_valid:
                    raise CylindricalCSNotFoundError(error_msg)
                self.tab.console_textbox.append(
                    f"  Cylindrical CS {config.deformation_cylindrical_cs_id} validated - "
                    f"results will be in cylindrical coordinates (R, Theta, Z)\n"
                )
            
            # Preload displacement data
            on_progress(5, 100, "Loading displacement from RST files...")
            engine.preload_displacement_data(progress_callback=on_progress)
            
            # Compute results based on mode
            if config.combination_history_mode and config.selected_node_id:
                # Single node history mode
                on_progress(50, 100, f"Computing displacement history for node {config.selected_node_id}...")
                combo_indices, ux, uy, uz, magnitude = engine.compute_single_node_history(
                    config.selected_node_id
                )
                
                # Create result object for history mode
                result = DeformationResult(
                    node_ids=np.array([config.selected_node_id]),
                    node_coords=np.array([[0, 0, 0]]),  # Placeholder
                    all_combo_ux=ux.reshape(-1, 1),
                    all_combo_uy=uy.reshape(-1, 1),
                    all_combo_uz=uz.reshape(-1, 1),
                    displacement_unit=engine.displacement_unit,
                )
                result.metadata = {
                    'mode': 'history',
                    'node_id': config.selected_node_id,
                    'combination_indices': combo_indices,
                    'ux': ux,
                    'uy': uy,
                    'uz': uz,
                    'magnitude': magnitude,
                }
            else:
                # Full envelope analysis
                on_progress(10, 100, "Computing deformation for all combinations...")
                result = engine.compute_full_analysis(progress_callback=on_progress)
            
            on_progress(100, 100, "Deformation complete")
            return result
            
        except (DisplacementNotAvailableError, CylindricalCSNotFoundError):
            raise
        except Exception as e:
            self.tab.console_textbox.append(f"\nError in deformation analysis: {e}\n")
            raise
    
    def _create_deformation_engine(self, config: SolverConfig) -> Optional[DeformationCombinationEngine]:
        """
        Create and configure the DeformationCombinationEngine.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            Configured DeformationCombinationEngine or None if creation fails.
        """
        try:
            # Get readers from file handler
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            
            if reader1 is None or reader2 is None:
                raise ValueError("DPF readers not available. Please reload RST files.")
            
            # Get nodal scoping from named selection
            ns_name = self.tab.get_selected_named_selection()
            nodal_scoping = reader1.get_nodal_scoping_from_named_selection(ns_name)
            
            # Get combination table
            combo_table = self.tab.get_combination_table_data()
            
            # Create engine with optional cylindrical CS
            engine = DeformationCombinationEngine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combination_table=combo_table,
                cylindrical_cs_id=config.deformation_cylindrical_cs_id,
            )
            
            return engine
            
        except Exception as e:
            QMessageBox.critical(
                self.tab, "Engine Creation Error",
                f"Failed to create deformation engine:\n\n{e}"
            )
            return None
    
    def _validate_inputs(self, config: SolverConfig) -> bool:
        """Validate inputs before running analysis."""
        # Check RST files loaded
        if not self.tab.base_rst_loaded or not self.tab.combine_rst_loaded:
            QMessageBox.warning(
                self.tab, "Missing Files",
                "Please load both RST files before running analysis."
            )
            return False
        
        # Check combination table
        combo_table = self.tab.get_combination_table_data()
        if combo_table is None or combo_table.num_combinations == 0:
            QMessageBox.warning(
                self.tab, "Missing Combinations",
                "Please define at least one combination in the table."
            )
            return False
        
        # Validate combination table (check for all-zero coefficients)
        is_valid, error_msg = combo_table.validate()
        if not is_valid:
            QMessageBox.warning(
                self.tab, "Invalid Combination Coefficients",
                error_msg
            )
            return False
        
        # Check named selection
        ns_name = self.tab.get_selected_named_selection()
        if ns_name is None:
            QMessageBox.warning(
                self.tab, "Missing Named Selection",
                "Please select a valid named selection."
            )
            return False
        
        # Check node ID for history mode
        if config.combination_history_mode:
            if config.selected_node_id is None:
                QMessageBox.warning(
                    self.tab, "Missing Node ID",
                    "Please enter a Node ID for combination history mode."
                )
                return False
        
        return True
    
    def _get_stress_type(self, config: SolverConfig) -> Optional[str]:
        """
        Determine which stress type to compute based on config.
        
        Returns None if only nodal forces are requested (no stress calculation).
        """
        if config.calculate_von_mises:
            return "von_mises"
        elif config.calculate_max_principal_stress:
            return "max_principal"
        elif config.calculate_min_principal_stress:
            return "min_principal"
        return None
    
    def _create_combination_engine(self, config: SolverConfig) -> Optional[CombinationEngine]:
        """
        Create and configure the CombinationEngine.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            Configured CombinationEngine or None if creation fails.
        """
        try:
            # Get readers from file handler
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            
            if reader1 is None or reader2 is None:
                raise ValueError("DPF readers not available. Please reload RST files.")
            
            # Get nodal scoping from named selection
            ns_name = self.tab.get_selected_named_selection()
            nodal_scoping = reader1.get_nodal_scoping_from_named_selection(ns_name)
            
            # Get combination table
            combo_table = self.tab.get_combination_table_data()
            
            # Create engine
            engine = CombinationEngine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combination_table=combo_table
            )
            
            return engine
            
        except Exception as e:
            QMessageBox.critical(
                self.tab, "Engine Creation Error",
                f"Failed to create combination engine:\n\n{e}"
            )
            return None
    
    def _should_use_chunked_processing(self, engine: CombinationEngine) -> bool:
        """
        Determine if chunked processing should be used based on memory estimates.
        
        Args:
            engine: The combination engine.
            
        Returns:
            True if chunked processing is recommended.
        """
        try:
            num_nodes = len(engine.scoping.ids)
            estimates = engine.estimate_memory_requirements(num_nodes)
            available = engine._get_available_memory()
            
            # Use chunked if estimated memory exceeds 80% of available
            threshold = available * 0.8
            
            if estimates['total_bytes'] > threshold:
                self.tab.console_textbox.append(
                    f"Large model detected ({num_nodes:,} nodes, "
                    f"~{estimates['total_bytes'] / 1e9:.2f} GB estimated).\n"
                    f"Using memory-efficient chunked processing.\n"
                )
                return True
            
            return False
            
        except Exception:
            # Default to standard processing if estimation fails
            return False
    
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress updates from solver thread."""
        if total > 0:
            percent = int((current / total) * 100)
            self.tab.progress_bar.setValue(percent)
            self.tab.progress_bar.setFormat(f"{message} ({percent}%)")
    
    def _on_memory_warning(self, warning_msg: str):
        """Handle memory warning from solver thread."""
        self.tab.console_textbox.append(f"\n⚠ Memory Warning: {warning_msg}\n")
    
    def _on_solve_complete(
        self, 
        stress_result: Optional[CombinationResult], 
        config: SolverConfig,
        forces_result: Optional[NodalForcesResult] = None,
        deformation_result: Optional[DeformationResult] = None
    ):
        """Handle successful solve completion."""
        # Re-enable UI
        self.tab.setEnabled(True)
        self.tab.progress_bar.setVisible(False)
        
        # Store results
        if stress_result is not None:
            self.tab.combination_result = stress_result
        if forces_result is not None:
            self.tab.nodal_forces_result = forces_result
        if deformation_result is not None:
            self.tab.deformation_result = deformation_result
        
        # Handle stress results based on mode
        if stress_result is not None:
            if config.combination_history_mode:
                self._handle_history_result(stress_result, config)
            else:
                self._handle_envelope_result(stress_result, config)
        
        # Handle nodal forces results
        if forces_result is not None:
            if config.combination_history_mode:
                self._handle_forces_history_result(forces_result, config)
            else:
                self._handle_forces_envelope_result(forces_result, config)
        
        # Handle deformation results
        if deformation_result is not None:
            if config.combination_history_mode:
                self._handle_deformation_history_result(deformation_result, config)
            else:
                self._handle_deformation_envelope_result(deformation_result, config)
        
        # Notify tab of completion for visualization
        # IMPORTANT: Skip 3D mesh update in history mode to preserve existing contour
        # History mode only updates the time history plot, not the 3D display
        if not config.combination_history_mode:
            # Use stress result if available for 3D display, otherwise use forces
            if stress_result is not None:
                self.tab.on_analysis_complete(stress_result)
            elif forces_result is not None:
                # Only display forces if no stress was computed
                self.tab.on_forces_analysis_complete(forces_result)
            
            # Notify deformation result - if no stress/forces, deformation creates its own mesh
            if deformation_result is not None:
                is_standalone = (stress_result is None and forces_result is None)
                self.tab.on_deformation_analysis_complete(deformation_result, is_standalone=is_standalone)
        
        # Log completion
        self._log_solve_complete()
    
    def _on_solve_error(self, error_msg: str):
        """Handle solve error."""
        self.tab.setEnabled(True)
        self.tab.progress_bar.setVisible(False)
        
        self.tab.console_textbox.append(f"\nSolver Error:\n{error_msg}\n")
        
        QMessageBox.critical(
            self.tab, "Solver Error",
            f"An error occurred during analysis:\n\n{error_msg}"
        )
    
    def _handle_history_result(self, result: CombinationResult, config: SolverConfig):
        """Handle results from combination history mode (single node)."""
        metadata = result.metadata or {}
        node_id = metadata.get('node_id', config.selected_node_id)
        combo_indices = metadata.get('combination_indices', np.arange(result.num_combinations))
        stress_values = metadata.get('stress_values', result.all_combo_results[:, 0])
        
        # Update combination history plot
        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        
        self.tab.plot_combo_history_tab.update_combination_history_plot(
            combo_indices,
            stress_values,
            node_id=node_id,
            stress_type=result.result_type,
            combination_names=combo_names
        )
        
        # Show the plot tab
        plot_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_combo_history_tab)
        if plot_idx >= 0:
            self.tab.show_output_tab_widget.setTabVisible(plot_idx, True)
            self.tab.show_output_tab_widget.setCurrentIndex(plot_idx)
        
        # Log completion
        self.tab.console_textbox.append(
            f"\nCombination history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max {result.result_type}: {np.max(stress_values):.4f}\n"
            f"  Min {result.result_type}: {np.min(stress_values):.4f}\n"
        )
    
    def _handle_envelope_result(self, result: CombinationResult, config: SolverConfig):
        """Handle results from envelope (batch) analysis."""
        # Determine if chunked processing was used
        used_chunked = result.all_combo_results is None
        
        # Determine which envelope to show based on result type:
        # - For min_principal stress: show MIN over combinations (most compressive = critical)
        # - For von_mises/max_principal: show MAX over combinations (highest stress = critical)
        is_min_principal = result.result_type == "min_principal"
        show_max = not is_min_principal
        show_min = is_min_principal
        
        # Log summary
        num_combos = self.tab.combination_table.num_combinations if self.tab.combination_table else 0
        self.tab.console_textbox.append(
            f"\nEnvelope analysis complete\n"
            f"  Nodes: {result.num_nodes}\n"
            f"  Combinations: {num_combos}\n"
            f"  Result type: {result.result_type}\n"
            f"  Processing mode: {'Chunked (memory-efficient)' if used_chunked else 'Standard'}\n"
        )
        
        # Log max results (only for von_mises and max_principal)
        if show_max and result.max_over_combo is not None:
            max_val = np.max(result.max_over_combo)
            max_node_idx = np.argmax(result.max_over_combo)
            max_node_id = result.node_ids[max_node_idx]
            max_combo_idx = result.combo_of_max[max_node_idx] if result.combo_of_max is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= max_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[max_combo_idx]})"
            
            # Display 1-based combination number for user-friendliness
            self.tab.console_textbox.append(
                f"  Maximum {result.result_type}: {max_val:.4f} at Node {max_node_id} "
                f"(Combination {max_combo_idx + 1}{combo_name})\n"
            )
        
        # Log min results (only for min_principal)
        if show_min and result.min_over_combo is not None:
            min_val = np.min(result.min_over_combo)
            min_node_idx = np.argmin(result.min_over_combo)
            min_node_id = result.node_ids[min_node_idx]
            min_combo_idx = result.combo_of_min[min_node_idx] if result.combo_of_min is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"
            
            # Display 1-based combination number for user-friendliness
            self.tab.console_textbox.append(
                f"  Minimum {result.result_type}: {min_val:.4f} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )
        
        # Get combination names
        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        
        # Compute max/min per combination (across all nodes) for the value vs combination plots
        max_per_combo = None
        min_per_combo = None
        
        if result.all_combo_results is not None:
            # all_combo_results has shape (num_combinations, num_nodes)
            # Compute max across all nodes for each combination
            if show_max:
                max_per_combo = np.max(result.all_combo_results, axis=1)
            if show_min:
                min_per_combo = np.min(result.all_combo_results, axis=1)
            combination_indices = np.arange(result.all_combo_results.shape[0])
        elif num_combos > 0:
            # Chunked mode - we only have per-node envelope values, not per-combination data
            combination_indices = np.arange(num_combos)
            # We don't have per-combo data in chunked mode, so we can't show this plot accurately
            max_per_combo = None
            min_per_combo = None
        else:
            combination_indices = np.arange(1)
        
        # Update max over combination plot (only for von_mises and max_principal)
        max_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_max_combo_tab)
        if max_idx >= 0:
            if show_max:
                if max_per_combo is not None:
                    # Use the new value-vs-combination plot (like original MARS)
                    self.tab.plot_max_combo_tab.update_max_over_combinations_plot(
                        combination_indices=combination_indices,
                        max_values_per_combo=max_per_combo,
                        min_values_per_combo=None,
                        combination_names=combo_names,
                        stress_type=result.result_type
                    )
                else:
                    # Fallback to bar chart for chunked mode (limited data available)
                    self.tab.plot_max_combo_tab.update_envelope_plot(
                        node_ids=result.node_ids,
                        max_values=result.max_over_combo,
                        min_values=None,
                        combo_of_max=result.combo_of_max,
                        combo_of_min=None,
                        stress_type=result.result_type,
                        combination_names=combo_names,
                        show_top_n=50
                    )
                self.tab.show_output_tab_widget.setTabVisible(max_idx, True)
                self.tab.show_output_tab_widget.setTabText(max_idx, "Maximum Over Combination")
            else:
                # Hide max tab for min_principal stress
                self.tab.show_output_tab_widget.setTabVisible(max_idx, False)
        
        # Update min over combination plot (only for min_principal)
        min_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_min_combo_tab)
        if min_idx >= 0:
            if show_min:
                if min_per_combo is not None:
                    # Use the new value-vs-combination plot (like original MARS)
                    self.tab.plot_min_combo_tab.update_max_over_combinations_plot(
                        combination_indices=combination_indices,
                        max_values_per_combo=None,
                        min_values_per_combo=min_per_combo,
                        combination_names=combo_names,
                        stress_type=result.result_type
                    )
                else:
                    # Fallback to bar chart for chunked mode (limited data available)
                    self.tab.plot_min_combo_tab.update_envelope_plot(
                        node_ids=result.node_ids,
                        max_values=None,
                        min_values=result.min_over_combo,
                        combo_of_max=None,
                        combo_of_min=result.combo_of_min,
                        stress_type=result.result_type,
                        combination_names=combo_names,
                        show_top_n=50
                    )
                self.tab.show_output_tab_widget.setTabVisible(min_idx, True)
                self.tab.show_output_tab_widget.setTabText(min_idx, "Minimum Over Combination")
            else:
                # Hide min tab for von_mises and max_principal stress
                self.tab.show_output_tab_widget.setTabVisible(min_idx, False)
        
        if used_chunked:
            # Log that per-combination plots are not available in chunked mode
            self.tab.console_textbox.append(
                f"  Note: Value-vs-combination plots not available in chunked mode.\n"
                f"  Showing top nodes by envelope value instead.\n"
            )
        
        # Auto-export envelope results to CSV
        output_dir = self._get_output_directory()
        if output_dir:
            self._export_envelope_csv(result, config, output_dir)
    
    def _handle_forces_history_result(self, result: NodalForcesResult, config: SolverConfig):
        """Handle results from nodal forces combination history mode (single node)."""
        metadata = getattr(result, 'metadata', {}) or {}
        node_id = metadata.get('node_id', config.selected_node_id)
        combo_indices = metadata.get('combination_indices', np.arange(result.num_combinations))
        fx = metadata.get('fx', result.all_combo_fx[:, 0] if result.all_combo_fx is not None else np.array([]))
        fy = metadata.get('fy', result.all_combo_fy[:, 0] if result.all_combo_fy is not None else np.array([]))
        fz = metadata.get('fz', result.all_combo_fz[:, 0] if result.all_combo_fz is not None else np.array([]))
        magnitude = metadata.get('magnitude', np.sqrt(fx**2 + fy**2 + fz**2))
        
        # Log results
        self.tab.console_textbox.append(
            f"\nNodal forces history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max Force Magnitude: {np.max(magnitude):.4f} {result.force_unit}\n"
            f"  Min Force Magnitude: {np.min(magnitude):.4f} {result.force_unit}\n"
        )
    
    def _handle_forces_envelope_result(self, result: NodalForcesResult, config: SolverConfig):
        """Handle results from nodal forces envelope (batch) analysis."""
        # Log summary
        self.tab.console_textbox.append(
            f"\nNodal forces envelope analysis complete\n"
            f"  Nodes: {result.num_nodes}\n"
            f"  Combinations: {self.tab.combination_table.num_combinations if self.tab.combination_table else 'N/A'}\n"
            f"  Force Unit: {result.force_unit}\n"
        )
        
        if result.max_magnitude_over_combo is not None:
            max_val = np.max(result.max_magnitude_over_combo)
            max_node_idx = np.argmax(result.max_magnitude_over_combo)
            max_node_id = result.node_ids[max_node_idx]
            max_combo_idx = result.combo_of_max[max_node_idx] if result.combo_of_max is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= max_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[max_combo_idx]})"
            
            self.tab.console_textbox.append(
                f"  Maximum Force Magnitude: {max_val:.4f} {result.force_unit} at Node {max_node_id} "
                f"(Combination {max_combo_idx + 1}{combo_name})\n"
            )
        
        if result.min_magnitude_over_combo is not None:
            min_val = np.min(result.min_magnitude_over_combo)
            min_node_idx = np.argmin(result.min_magnitude_over_combo)
            min_node_id = result.node_ids[min_node_idx]
            min_combo_idx = result.combo_of_min[min_node_idx] if result.combo_of_min is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"
            
            self.tab.console_textbox.append(
                f"  Minimum Force Magnitude: {min_val:.4f} {result.force_unit} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )
        
        # Update the plot tabs with nodal forces envelope
        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        
        # Use the max combo tab for nodal forces (shows nodes with highest force magnitude)
        max_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_max_combo_tab)
        if max_idx >= 0 and result.max_magnitude_over_combo is not None:
            self.tab.plot_max_combo_tab.update_forces_envelope_plot(
                node_ids=result.node_ids,
                max_magnitude=result.max_magnitude_over_combo,
                min_magnitude=None,
                combo_of_max=result.combo_of_max,
                combo_of_min=None,
                combination_names=combo_names,
                force_unit=result.force_unit,
                show_top_n=50
            )
            self.tab.show_output_tab_widget.setTabVisible(max_idx, True)
            # Rename tab to indicate it's showing forces
            self.tab.show_output_tab_widget.setTabText(max_idx, "Max Forces Over Combination")
        
        # Use the min combo tab for min nodal forces
        min_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_min_combo_tab)
        if min_idx >= 0 and result.min_magnitude_over_combo is not None:
            self.tab.plot_min_combo_tab.update_forces_envelope_plot(
                node_ids=result.node_ids,
                max_magnitude=None,
                min_magnitude=result.min_magnitude_over_combo,
                combo_of_max=None,
                combo_of_min=result.combo_of_min,
                combination_names=combo_names,
                force_unit=result.force_unit,
                show_top_n=50
            )
            self.tab.show_output_tab_widget.setTabVisible(min_idx, True)
            # Rename tab to indicate it's showing forces
            self.tab.show_output_tab_widget.setTabText(min_idx, "Min Forces Over Combination")
        
        # Auto-export nodal forces envelope results to CSV
        output_dir = self._get_output_directory()
        if output_dir:
            self._export_forces_envelope_csv(result, output_dir)
    
    def _handle_deformation_history_result(self, result: DeformationResult, config: SolverConfig):
        """Handle results from deformation combination history mode (single node)."""
        metadata = getattr(result, 'metadata', {}) or {}
        node_id = metadata.get('node_id', config.selected_node_id)
        combo_indices = metadata.get('combination_indices', np.arange(result.num_combinations))
        ux = metadata.get('ux', result.all_combo_ux[:, 0] if result.all_combo_ux is not None else np.array([]))
        uy = metadata.get('uy', result.all_combo_uy[:, 0] if result.all_combo_uy is not None else np.array([]))
        uz = metadata.get('uz', result.all_combo_uz[:, 0] if result.all_combo_uz is not None else np.array([]))
        magnitude = metadata.get('magnitude', np.sqrt(ux**2 + uy**2 + uz**2))

        # Update combination history plot for deformation
        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        deformation_data = {
            'ux': ux,
            'uy': uy,
            'uz': uz,
            'u_mag': magnitude
        }
        self.tab.plot_combo_history_tab.update_combination_history_plot(
            combo_indices,
            stress_values=None,
            node_id=node_id,
            combination_names=combo_names,
            deformation_data=deformation_data,
            displacement_unit=result.displacement_unit
        )

        # Show the plot tab
        plot_idx = self.tab.show_output_tab_widget.indexOf(self.tab.plot_combo_history_tab)
        if plot_idx >= 0:
            self.tab.show_output_tab_widget.setTabVisible(plot_idx, True)
            self.tab.show_output_tab_widget.setCurrentIndex(plot_idx)
        
        # Log results
        self.tab.console_textbox.append(
            f"\nDeformation history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max Displacement Magnitude: {np.max(magnitude):.6f} {result.displacement_unit}\n"
            f"  Min Displacement Magnitude: {np.min(magnitude):.6f} {result.displacement_unit}\n"
        )
    
    def _handle_deformation_envelope_result(self, result: DeformationResult, config: SolverConfig):
        """Handle results from deformation envelope (batch) analysis."""
        # Log summary
        self.tab.console_textbox.append(
            f"\nDeformation envelope analysis complete\n"
            f"  Nodes: {result.num_nodes}\n"
            f"  Combinations: {self.tab.combination_table.num_combinations if self.tab.combination_table else 'N/A'}\n"
            f"  Displacement Unit: {result.displacement_unit}\n"
        )
        
        if result.max_magnitude_over_combo is not None:
            max_val = np.max(result.max_magnitude_over_combo)
            max_node_idx = np.argmax(result.max_magnitude_over_combo)
            max_node_id = result.node_ids[max_node_idx]
            max_combo_idx = result.combo_of_max[max_node_idx] if result.combo_of_max is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= max_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[max_combo_idx]})"
            
            self.tab.console_textbox.append(
                f"  Maximum Displacement: {max_val:.6f} {result.displacement_unit} at Node {max_node_id} "
                f"(Combination {max_combo_idx + 1}{combo_name})\n"
            )
        
        if result.min_magnitude_over_combo is not None:
            min_val = np.min(result.min_magnitude_over_combo)
            min_node_idx = np.argmin(result.min_magnitude_over_combo)
            min_node_id = result.node_ids[min_node_idx]
            min_combo_idx = result.combo_of_min[min_node_idx] if result.combo_of_min is not None else -1
            
            # Get combination name if available
            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"
            
            self.tab.console_textbox.append(
                f"  Minimum Displacement: {min_val:.6f} {result.displacement_unit} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )
        
        # Auto-export deformation envelope results to CSV
        output_dir = self._get_output_directory()
        if output_dir:
            self._export_deformation_envelope_csv(result, output_dir)
    
    def _export_deformation_envelope_csv(
        self, 
        result: DeformationResult, 
        output_dir: str
    ):
        """
        Export deformation envelope results to CSV files.
        
        Args:
            result: DeformationResult containing envelope data.
            output_dir: Directory to write the CSV file.
        """
        from file_io.exporters import export_deformation_envelope
        
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
            
            # Build filename
            filename = os.path.join(output_dir, "envelope_deformation.csv")
            
            export_deformation_envelope(
                filename=filename,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                max_magnitude=result.max_magnitude_over_combo,
                min_magnitude=result.min_magnitude_over_combo,
                combo_of_max=result.combo_of_max,
                combo_of_min=result.combo_of_min,
                combination_names=combo_names,
                displacement_unit=result.displacement_unit
            )
            
            self.tab.console_textbox.append(f"  Exported deformation envelope to: {filename}\n")
            
        except Exception as e:
            self.tab.console_textbox.append(f"  Warning: Failed to export deformation CSV: {e}\n")
    
    def _log_solve_start(self, config: SolverConfig):
        """Log solve start information."""
        current_time = datetime.now()
        self._solve_start_time = current_time
        
        output_types = []
        if config.calculate_von_mises:
            output_types.append("Von Mises")
        if config.calculate_max_principal_stress:
            output_types.append("Max Principal")
        if config.calculate_min_principal_stress:
            output_types.append("Min Principal")
        if config.calculate_nodal_forces:
            csys = "Global" if config.nodal_forces_rotate_to_global else "Local (Element)"
            output_types.append(f"Nodal Forces ({csys})")
        if config.calculate_deformation:
            output_types.append("Deformation")
        
        mode = "Combination History" if config.combination_history_mode else "Envelope"
        
        self.tab.console_textbox.append(
            f"\n{'='*60}\n"
            f"BEGIN COMBINATION ANALYSIS\n"
            f"{'='*60}\n"
            f"Datetime: {current_time}\n"
            f"Mode: {mode}\n"
            f"Outputs: {', '.join(output_types)}\n"
        )
        
        if config.combination_history_mode and config.selected_node_id:
            self.tab.console_textbox.append(f"Node ID: {config.selected_node_id}\n")
    
    def _log_solve_complete(self):
        """Log solve completion."""
        end_time = datetime.now()
        elapsed = None
        if self._solve_start_time:
            elapsed = (end_time - self._solve_start_time).total_seconds()
        
        msg = f"\n{'='*60}\nANALYSIS COMPLETE\n{'='*60}\n"
        if elapsed is not None:
            msg += f"Elapsed time: {elapsed:.3f} seconds\n"
        
        self.tab.console_textbox.append(msg)
        self._solve_start_time = None
    
    def get_combination_engine(self) -> Optional[CombinationEngine]:
        """Get the current combination engine (if available)."""
        return self._combination_engine
    
    def _get_output_directory(self) -> Optional[str]:
        """
        Get output directory for CSV export.
        
        Priority:
        1. Project directory if set
        2. Directory of the base RST file
        
        Returns:
            Output directory path or None if not available.
        """
        # First try project directory
        if self.tab.project_directory:
            return self.tab.project_directory
        
        # Fallback to RST file directory
        if self.tab.file_handler.base_reader:
            rst_path = getattr(self.tab.file_handler.base_reader, 'rst_path', None)
            if rst_path:
                return os.path.dirname(rst_path)
        
        return None
    
    def _export_envelope_csv(
        self, 
        result: CombinationResult, 
        config: SolverConfig,
        output_dir: str
    ):
        """
        Export envelope results to CSV files.
        
        Creates a CSV file containing max OR min stress values over all combinations
        for each node, along with which combination produced the extreme values.
        
        For von_mises and max_principal: exports MAX values (highest stress = critical)
        For min_principal: exports MIN values (most compressive = critical)
        
        Args:
            result: CombinationResult containing envelope data.
            config: SolverConfig with analysis settings.
            output_dir: Directory to write the CSV file.
        """
        from file_io.exporters import export_envelope_results
        
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
            result_type = result.result_type  # e.g., "von_mises"
            
            # Determine which envelope to export based on result type
            is_min_principal = result_type == "min_principal"
            
            # Build filename based on result type
            filename = os.path.join(output_dir, f"envelope_{result_type}.csv")
            
            # For min_principal: export only min values
            # For von_mises/max_principal: export only max values
            export_envelope_results(
                filename=filename,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                max_values=None if is_min_principal else result.max_over_combo,
                min_values=result.min_over_combo if is_min_principal else None,
                combo_of_max=None if is_min_principal else result.combo_of_max,
                combo_of_min=result.combo_of_min if is_min_principal else None,
                result_type=result_type,
                combination_names=combo_names
            )
            
            self.tab.console_textbox.append(f"  Exported envelope results to: {filename}\n")
            
        except Exception as e:
            self.tab.console_textbox.append(f"  Warning: Failed to export envelope CSV: {e}\n")
    
    def _export_forces_envelope_csv(
        self, 
        result: NodalForcesResult, 
        output_dir: str
    ):
        """
        Export nodal forces envelope results to CSV files.
        
        Creates a CSV file containing max/min force magnitudes over all combinations
        for each node, along with which combination produced the extreme values.
        
        Args:
            result: NodalForcesResult containing envelope data.
            output_dir: Directory to write the CSV file.
        """
        from file_io.exporters import export_nodal_forces_envelope
        
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
            
            # Build filename
            filename = os.path.join(output_dir, "envelope_nodal_forces.csv")
            
            export_nodal_forces_envelope(
                filename=filename,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                max_magnitude=result.max_magnitude_over_combo,
                min_magnitude=result.min_magnitude_over_combo,
                combo_of_max=result.combo_of_max,
                combo_of_min=result.combo_of_min,
                combination_names=combo_names,
                force_unit=result.force_unit
            )
            
            self.tab.console_textbox.append(f"  Exported nodal forces envelope to: {filename}\n")
            
        except Exception as e:
            self.tab.console_textbox.append(f"  Warning: Failed to export nodal forces CSV: {e}\n")
