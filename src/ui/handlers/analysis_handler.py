"""
Runs combination analysis from the Solver tab: validates inputs, builds SolverConfig,
calls StressCombinationEngine, and shows results. DPF runs in the main thread (gRPC doesn't
play nice with threads); processEvents() keeps the GUI responsive.
"""

import traceback
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QMessageBox, QApplication

from solver.stress_engine import StressCombinationEngine
from solver.nodal_forces_engine import NodalForcesCombinationEngine
from solver.deformation_engine import DeformationCombinationEngine, CylindricalCSNotFoundError
from file_io.dpf_reader import (
    NodalForcesNotAvailableError, DisplacementNotAvailableError
)
from core.data_models import (
    SolverConfig, CombinationResult,
    NodalForcesResult, DeformationResult
)
from ui.handlers.analysis_results_handler import AnalysisResultsHandler
from ui.handlers.solver_engine_factory import SolverEngineFactory
from ui.handlers.solver_input_validator import SolverInputValidator
from utils.constants import MSG_NODAL_FORCES_ANSYS


class SolverAnalysisHandler:
    """Manages the combination analysis execution and result handling for the SolverTab."""

    # Memory threshold in GB to switch to chunked processing
    MEMORY_THRESHOLD_GB = 2.0

    def __init__(self, tab):
        """tab: the SolverTab we're attached to."""
        self.tab = tab
        self.input_validator = SolverInputValidator(tab)
        self.engine_factory = SolverEngineFactory()
        self.results_handler = AnalysisResultsHandler(tab)
        self._solve_start_time: Optional[datetime] = None
        self._combination_engine: Optional[StressCombinationEngine] = None
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
                f"{MSG_NODAL_FORCES_ANSYS}"
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
                f"- Use a smaller Named Selection to reduce nodes\n"
                f"- Close other applications to free memory\n\n"
                f"Details: {str(e)}"
            )
            QMessageBox.critical(self.tab, "Memory Error", error_msg)
            
        except Exception as e:
            self.tab.progress_bar.setVisible(False)
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self._on_solve_error(error_msg)
    
    def _run_analysis(
        self, 
        engine: StressCombinationEngine, 
        config: SolverConfig, 
        stress_type: str,
        use_chunked: bool
    ) -> Optional[CombinationResult]:
        """
        Run the actual analysis computation in main thread.
        
        Args:
            engine: Configured StressCombinationEngine.
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
                    f"\n[Warning] Memory Warning: Limited RAM detected.\n"
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
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            nodal_scoping = self.tab.get_nodal_scoping_for_selected_named_selection()
            combo_table = self.tab.get_combination_table_data()
            return self.engine_factory.create_nodal_forces_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
                rotate_to_global=config.nodal_forces_rotate_to_global,
            )
            
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
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            nodal_scoping = self.tab.get_nodal_scoping_for_selected_named_selection()
            combo_table = self.tab.get_combination_table_data()
            return self.engine_factory.create_deformation_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
                cylindrical_cs_id=config.deformation_cylindrical_cs_id,
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.tab, "Engine Creation Error",
                f"Failed to create deformation engine:\n\n{e}"
            )
            return None
    
    def _validate_inputs(self, config: SolverConfig) -> bool:
        """Validate inputs before running analysis."""
        return self.input_validator.validate_inputs(config)
    
    def _get_stress_type(self, config: SolverConfig) -> Optional[str]:
        """
        Determine which stress type to compute based on config.

        Returns None if only nodal forces are requested (no stress calculation).
        """
        return self.input_validator.get_selected_stress_type(config)
    
    def _create_combination_engine(self, config: SolverConfig) -> Optional[StressCombinationEngine]:
        """
        Create and configure the StressCombinationEngine.
        
        Args:
            config: SolverConfig with analysis settings.
            
        Returns:
            Configured StressCombinationEngine or None if creation fails.
        """
        try:
            reader1 = self.tab.file_handler.base_reader
            reader2 = self.tab.file_handler.combine_reader
            nodal_scoping = self.tab.get_nodal_scoping_for_selected_named_selection()
            combo_table = self.tab.get_combination_table_data()
            return self.engine_factory.create_stress_engine(
                reader1=reader1,
                reader2=reader2,
                nodal_scoping=nodal_scoping,
                combo_table=combo_table,
            )
            
        except Exception as e:
            QMessageBox.critical(
                self.tab, "Engine Creation Error",
                f"Failed to create combination engine:\n\n{e}"
            )
            return None
    
    def _should_use_chunked_processing(self, engine: StressCombinationEngine) -> bool:
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
        self.tab.console_textbox.append(f"\n[Warning] Memory Warning: {warning_msg}\n")
    
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
        
        # Store current-run results (clear stale outputs when not computed)
        self.tab.combination_result = stress_result
        self.tab.nodal_forces_result = forces_result
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
        self.results_handler.handle_stress_history_result(result, config)

    def _handle_envelope_result(self, result: CombinationResult, config: SolverConfig):
        """Handle results from envelope (batch) analysis."""
        self.results_handler.handle_stress_envelope_result(result, config)

    def _handle_forces_history_result(self, result: NodalForcesResult, config: SolverConfig):
        """Handle results from nodal forces combination history mode (single node)."""
        self.results_handler.handle_forces_history_result(result, config)

    def _handle_forces_envelope_result(self, result: NodalForcesResult, config: SolverConfig):
        """Handle results from nodal forces envelope (batch) analysis."""
        self.results_handler.handle_forces_envelope_result(result, config)

    def _handle_deformation_history_result(self, result: DeformationResult, config: SolverConfig):
        """Handle results from deformation combination history mode (single node)."""
        self.results_handler.handle_deformation_history_result(result, config)

    def _handle_deformation_envelope_result(self, result: DeformationResult, config: SolverConfig):
        """Handle results from deformation envelope (batch) analysis."""
        self.results_handler.handle_deformation_envelope_result(result, config)

    def _export_deformation_envelope_csv(self, result: DeformationResult, output_dir: str):
        """Export deformation envelope results to CSV files."""
        self.results_handler.export_deformation_envelope_csv(result, output_dir)
    
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
    
    def get_combination_engine(self) -> Optional[StressCombinationEngine]:
        """Get the current combination engine (if available)."""
        return self._combination_engine
    
    def _get_output_directory(self) -> Optional[str]:
        """Get output directory for CSV export."""
        return self.results_handler.get_output_directory()
    
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
        self.results_handler.export_stress_envelope_csv(result, config, output_dir)
    
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
        self.results_handler.export_forces_envelope_csv(result, output_dir)

