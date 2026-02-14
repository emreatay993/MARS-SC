"""
UI result handling for analysis runs.

This module owns post-solve behavior: plotting, summary logging, and CSV exports.
"""

import os
from typing import Optional

import numpy as np

from core.data_models import (
    CombinationResult,
    DeformationResult,
    NodalForcesResult,
    SolverConfig,
)
from file_io.exporters import (
    export_deformation_envelope,
    export_envelope_results,
    export_nodal_forces_envelope,
)


class SolverResultSummaryHandler:
    """Handle and present stress, force, and deformation results in the UI."""

    def __init__(self, tab):
        self.tab = tab

    def _show_result_tab(self, tab_page, tab_title: str, make_current: bool = False) -> int:
        """
        Ensure a result tab page exists and is visible.

        Tabs are added lazily. If a page already exists but is hidden by visibility APIs,
        this method re-adds it as a fallback.
        """
        tab_widget = self.tab.show_output_tab_widget
        tab_index = tab_widget.indexOf(tab_page)

        if tab_index < 0:
            tab_index = tab_widget.addTab(tab_page, tab_title)
        else:
            tab_widget.setTabText(tab_index, tab_title)

        # Compatibility fallback for runtimes that keep a hidden page selected.
        is_tab_visible = getattr(tab_widget, "isTabVisible", None)
        if callable(is_tab_visible):
            try:
                if not is_tab_visible(tab_index):
                    tab_widget.removeTab(tab_index)
                    tab_index = tab_widget.addTab(tab_page, tab_title)
            except Exception:
                pass

        if make_current:
            tab_widget.setCurrentIndex(tab_index)

        return tab_index

    def _hide_result_tab(self, tab_page) -> None:
        """Remove a result tab page when it should not be shown."""
        tab_widget = self.tab.show_output_tab_widget
        tab_index = tab_widget.indexOf(tab_page)
        if tab_index >= 0:
            tab_widget.removeTab(tab_index)

    def handle_stress_history_result(self, result: CombinationResult, config: SolverConfig) -> None:
        """Handle stress history mode result (single node)."""
        metadata = result.metadata or {}
        node_id = metadata.get("node_id", config.selected_node_id)
        combo_indices = metadata.get("combination_indices", np.arange(result.num_combinations))
        stress_values = metadata.get("stress_values", result.all_combo_results[:, 0])

        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        self.tab.plot_combo_history_tab.update_combination_history_plot(
            combo_indices,
            stress_values,
            node_id=node_id,
            stress_type=result.result_type,
            combination_names=combo_names,
        )

        self._show_result_tab(
            self.tab.plot_combo_history_tab,
            "Plot (Combo History)",
            make_current=True,
        )

        if getattr(self.tab, "_history_popup_requested", False) and hasattr(self.tab, "_show_history_popup"):
            self.tab._show_history_popup(
                combination_indices=combo_indices,
                stress_values=stress_values,
                node_id=node_id,
                combination_names=combo_names,
                stress_type=result.result_type,
            )

        self.tab.console_textbox.append(
            f"\nCombination history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max {result.result_type}: {np.max(stress_values):.4f}\n"
            f"  Min {result.result_type}: {np.min(stress_values):.4f}\n"
        )

    def handle_stress_envelope_result(self, result: CombinationResult, config: SolverConfig) -> None:
        """Handle stress envelope mode result (all nodes)."""
        used_chunked = result.all_combo_results is None
        is_min_principal = result.result_type == "min_principal"
        show_max = not is_min_principal
        show_min = is_min_principal

        num_combos = self.tab.combination_table.num_combinations if self.tab.combination_table else 0
        self.tab.console_textbox.append(
            f"\nEnvelope analysis complete\n"
            f"  Nodes: {result.num_nodes}\n"
            f"  Combinations: {num_combos}\n"
            f"  Result type: {result.result_type}\n"
            f"  Processing mode: {'Chunked (memory-efficient)' if used_chunked else 'Standard'}\n"
        )

        if show_max and result.max_over_combo is not None:
            max_val = np.max(result.max_over_combo)
            max_node_idx = np.argmax(result.max_over_combo)
            max_node_id = result.node_ids[max_node_idx]
            max_combo_idx = result.combo_of_max[max_node_idx] if result.combo_of_max is not None else -1

            combo_name = ""
            if self.tab.combination_table and 0 <= max_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[max_combo_idx]})"

            self.tab.console_textbox.append(
                f"  Maximum {result.result_type}: {max_val:.4f} at Node {max_node_id} "
                f"(Combination {max_combo_idx + 1}{combo_name})\n"
            )

        if show_min and result.min_over_combo is not None:
            min_val = np.min(result.min_over_combo)
            min_node_idx = np.argmin(result.min_over_combo)
            min_node_id = result.node_ids[min_node_idx]
            min_combo_idx = result.combo_of_min[min_node_idx] if result.combo_of_min is not None else -1

            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"

            self.tab.console_textbox.append(
                f"  Minimum {result.result_type}: {min_val:.4f} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )

        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None

        max_per_combo = None
        min_per_combo = None
        if result.all_combo_results is not None:
            if show_max:
                max_per_combo = np.max(result.all_combo_results, axis=1)
            if show_min:
                min_per_combo = np.min(result.all_combo_results, axis=1)
            combination_indices = np.arange(result.all_combo_results.shape[0])
        elif num_combos > 0:
            combination_indices = np.arange(num_combos)
        else:
            combination_indices = np.arange(1)

        if show_max:
            if max_per_combo is not None:
                self.tab.plot_max_combo_tab.update_max_over_combinations_plot(
                    combination_indices=combination_indices,
                    max_values_per_combo=max_per_combo,
                    min_values_per_combo=None,
                    combination_names=combo_names,
                    stress_type=result.result_type,
                )
            else:
                self.tab.plot_max_combo_tab.update_envelope_plot(
                    node_ids=result.node_ids,
                    max_values=result.max_over_combo,
                    min_values=None,
                    combo_of_max=result.combo_of_max,
                    combo_of_min=None,
                    stress_type=result.result_type,
                    combination_names=combo_names,
                    show_top_n=50,
                )
            self._show_result_tab(self.tab.plot_max_combo_tab, "Maximum Over Combination")
        else:
            self._hide_result_tab(self.tab.plot_max_combo_tab)

        if show_min:
            if min_per_combo is not None:
                self.tab.plot_min_combo_tab.update_max_over_combinations_plot(
                    combination_indices=combination_indices,
                    max_values_per_combo=None,
                    min_values_per_combo=min_per_combo,
                    combination_names=combo_names,
                    stress_type=result.result_type,
                )
            else:
                self.tab.plot_min_combo_tab.update_envelope_plot(
                    node_ids=result.node_ids,
                    max_values=None,
                    min_values=result.min_over_combo,
                    combo_of_max=None,
                    combo_of_min=result.combo_of_min,
                    stress_type=result.result_type,
                    combination_names=combo_names,
                    show_top_n=50,
                )
            self._show_result_tab(self.tab.plot_min_combo_tab, "Minimum Over Combination")
        else:
            self._hide_result_tab(self.tab.plot_min_combo_tab)

        if used_chunked:
            self.tab.console_textbox.append(
                "  Note: Value-vs-combination plots not available in chunked mode.\n"
                "  Showing top nodes by envelope value instead.\n"
            )

        output_dir = self.get_output_directory()
        if output_dir:
            self.export_stress_envelope_csv(result, config, output_dir)

    def handle_forces_history_result(self, result: NodalForcesResult, config: SolverConfig) -> None:
        """Handle nodal forces history mode result (single node)."""
        metadata = getattr(result, "metadata", {}) or {}
        node_id = metadata.get("node_id", config.selected_node_id)
        combo_indices = metadata.get("combination_indices", np.arange(result.num_combinations))
        fx = metadata.get("fx", result.all_combo_fx[:, 0] if result.all_combo_fx is not None else np.array([]))
        fy = metadata.get("fy", result.all_combo_fy[:, 0] if result.all_combo_fy is not None else np.array([]))
        fz = metadata.get("fz", result.all_combo_fz[:, 0] if result.all_combo_fz is not None else np.array([]))
        magnitude = metadata.get("magnitude", np.sqrt(fx**2 + fy**2 + fz**2))

        self.tab.console_textbox.append(
            f"\nNodal forces history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max Force Magnitude: {np.max(magnitude):.4f} {result.force_unit}\n"
            f"  Min Force Magnitude: {np.min(magnitude):.4f} {result.force_unit}\n"
        )

    def handle_forces_envelope_result(self, result: NodalForcesResult, _config: SolverConfig) -> None:
        """Handle nodal forces envelope mode result (all nodes)."""
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

            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"

            self.tab.console_textbox.append(
                f"  Minimum Force Magnitude: {min_val:.4f} {result.force_unit} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )

        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None

        if result.max_magnitude_over_combo is not None:
            self.tab.plot_max_combo_tab.update_forces_envelope_plot(
                node_ids=result.node_ids,
                max_magnitude=result.max_magnitude_over_combo,
                min_magnitude=None,
                combo_of_max=result.combo_of_max,
                combo_of_min=None,
                combination_names=combo_names,
                force_unit=result.force_unit,
                show_top_n=50,
            )
            self._show_result_tab(self.tab.plot_max_combo_tab, "Max Forces Over Combination")
        else:
            self._hide_result_tab(self.tab.plot_max_combo_tab)

        if result.min_magnitude_over_combo is not None:
            self.tab.plot_min_combo_tab.update_forces_envelope_plot(
                node_ids=result.node_ids,
                max_magnitude=None,
                min_magnitude=result.min_magnitude_over_combo,
                combo_of_max=None,
                combo_of_min=result.combo_of_min,
                combination_names=combo_names,
                force_unit=result.force_unit,
                show_top_n=50,
            )
            self._show_result_tab(self.tab.plot_min_combo_tab, "Min Forces Over Combination")
        else:
            self._hide_result_tab(self.tab.plot_min_combo_tab)

        output_dir = self.get_output_directory()
        if output_dir:
            self.export_forces_envelope_csv(result, output_dir)

    def handle_deformation_history_result(self, result: DeformationResult, config: SolverConfig) -> None:
        """Handle deformation history mode result (single node)."""
        metadata = getattr(result, "metadata", {}) or {}
        node_id = metadata.get("node_id", config.selected_node_id)
        combo_indices = metadata.get("combination_indices", np.arange(result.num_combinations))
        ux = metadata.get("ux", result.all_combo_ux[:, 0] if result.all_combo_ux is not None else np.array([]))
        uy = metadata.get("uy", result.all_combo_uy[:, 0] if result.all_combo_uy is not None else np.array([]))
        uz = metadata.get("uz", result.all_combo_uz[:, 0] if result.all_combo_uz is not None else np.array([]))
        magnitude = metadata.get("magnitude", np.sqrt(ux**2 + uy**2 + uz**2))

        combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
        deformation_data = {
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "u_mag": magnitude,
        }
        self.tab.plot_combo_history_tab.update_combination_history_plot(
            combo_indices,
            stress_values=None,
            node_id=node_id,
            combination_names=combo_names,
            deformation_data=deformation_data,
            displacement_unit=result.displacement_unit,
        )

        self._show_result_tab(
            self.tab.plot_combo_history_tab,
            "Plot (Combo History)",
            make_current=True,
        )

        if getattr(self.tab, "_history_popup_requested", False) and hasattr(self.tab, "_show_history_popup"):
            self.tab._show_history_popup(
                combination_indices=combo_indices,
                stress_values=None,
                node_id=node_id,
                combination_names=combo_names,
                deformation_data=deformation_data,
                displacement_unit=result.displacement_unit,
            )

        self.tab.console_textbox.append(
            f"\nDeformation history computed for Node {node_id}\n"
            f"  Combinations: {len(combo_indices)}\n"
            f"  Max Displacement Magnitude: {np.max(magnitude):.6f} {result.displacement_unit}\n"
            f"  Min Displacement Magnitude: {np.min(magnitude):.6f} {result.displacement_unit}\n"
        )

    def handle_deformation_envelope_result(self, result: DeformationResult, _config: SolverConfig) -> None:
        """Handle deformation envelope mode result (all nodes)."""
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

            combo_name = ""
            if self.tab.combination_table and 0 <= min_combo_idx < len(self.tab.combination_table.combination_names):
                combo_name = f" ({self.tab.combination_table.combination_names[min_combo_idx]})"

            self.tab.console_textbox.append(
                f"  Minimum Displacement: {min_val:.6f} {result.displacement_unit} at Node {min_node_id} "
                f"(Combination {min_combo_idx + 1}{combo_name})\n"
            )

        output_dir = self.get_output_directory()
        if output_dir:
            self.export_deformation_envelope_csv(result, output_dir)

    def get_output_directory(self) -> Optional[str]:
        """Return output directory for result exports."""
        if self.tab.project_directory:
            return self.tab.project_directory
        if self.tab.file_handler.base_reader:
            rst_path = getattr(self.tab.file_handler.base_reader, "rst_path", None)
            if rst_path:
                return os.path.dirname(rst_path)
        return None

    def export_stress_envelope_csv(
        self,
        result: CombinationResult,
        _config: SolverConfig,
        output_dir: str,
    ) -> None:
        """Export stress envelope results to CSV."""
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
            result_type = result.result_type
            is_min_principal = result_type == "min_principal"
            filename = os.path.join(output_dir, f"envelope_{result_type}.csv")

            export_envelope_results(
                filename=filename,
                node_ids=result.node_ids,
                node_coords=result.node_coords,
                max_values=None if is_min_principal else result.max_over_combo,
                min_values=result.min_over_combo if is_min_principal else None,
                combo_of_max=None if is_min_principal else result.combo_of_max,
                combo_of_min=result.combo_of_min if is_min_principal else None,
                result_type=result_type,
                combination_names=combo_names,
            )

            self.tab.console_textbox.append(f"  Exported envelope results to: {filename}\n")
        except Exception as error:
            self.tab.console_textbox.append(f"  Warning: Failed to export envelope CSV: {error}\n")

    def export_forces_envelope_csv(self, result: NodalForcesResult, output_dir: str) -> None:
        """Export nodal forces envelope results to CSV."""
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
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
                force_unit=result.force_unit,
                all_combo_fx=result.all_combo_fx,
                all_combo_fy=result.all_combo_fy,
                all_combo_fz=result.all_combo_fz,
                include_shear_variants=True,
                include_component_envelopes=True,
                include_component_combo_indices=True,
            )

            self.tab.console_textbox.append(f"  Exported nodal forces envelope to: {filename}\n")
        except Exception as error:
            self.tab.console_textbox.append(f"  Warning: Failed to export nodal forces CSV: {error}\n")

    def export_deformation_envelope_csv(self, result: DeformationResult, output_dir: str) -> None:
        """Export deformation envelope results to CSV."""
        try:
            combo_names = self.tab.combination_table.combination_names if self.tab.combination_table else None
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
                displacement_unit=result.displacement_unit,
            )

            self.tab.console_textbox.append(f"  Exported deformation envelope to: {filename}\n")
        except Exception as error:
            self.tab.console_textbox.append(f"  Warning: Failed to export deformation CSV: {error}\n")
