"""
Display/presentation handling for solver outputs.

This module converts solver results into visualization-ready payloads
and emits them to the display tab.
"""

import numpy as np

from core.data_models import CombinationResult, DeformationResult, NodalForcesResult
from ui.display_payload import DisplayResultPayload, SolverOutputFlags


class SolverResultPayloadHandler:
    """Build display payloads and meshes from solver output results."""

    def __init__(self, tab):
        self.tab = tab

    def on_analysis_complete(self, result: CombinationResult) -> None:
        """Handle stress analysis completion."""
        self.tab.combination_result = result

        mesh = self._create_mesh_from_result(result)

        scalar_bar_titles = {
            "von_mises": "Von Mises Stress [MPa]",
            "max_principal": "Max Principal Stress (S1) [MPa]",
            "min_principal": "Min Principal Stress (S3) [MPa]",
        }
        scalar_bar_title = scalar_bar_titles.get(result.result_type, "Stress [MPa]")

        data_min = float(np.min(result.max_over_combo)) if result.max_over_combo is not None else 0.0
        data_max = float(np.max(result.max_over_combo)) if result.max_over_combo is not None else 0.0

        self._emit_display_payload(mesh, scalar_bar_title, data_min, data_max)

    def on_forces_analysis_complete(self, result: NodalForcesResult) -> None:
        """Handle nodal forces analysis completion."""
        self.tab.nodal_forces_result = result

        mesh = self._create_mesh_from_forces_result(result)
        scalar_bar_title = f"Force Magnitude [{result.force_unit}]"

        data_min = (
            float(np.min(result.max_magnitude_over_combo))
            if result.max_magnitude_over_combo is not None
            else 0.0
        )
        data_max = (
            float(np.max(result.max_magnitude_over_combo))
            if result.max_magnitude_over_combo is not None
            else 0.0
        )

        self._emit_display_payload(mesh, scalar_bar_title, data_min, data_max)

    def on_deformation_analysis_complete(
        self,
        result: DeformationResult,
        is_standalone: bool = False,
    ) -> None:
        """Handle deformation analysis completion."""
        self.tab.deformation_result = result

        if is_standalone:
            mesh = self._create_mesh_from_deformation_result(result)
            scalar_bar_title = f"Displacement Magnitude [{result.displacement_unit}]"
            data_min = (
                float(np.min(result.max_magnitude_over_combo))
                if result.max_magnitude_over_combo is not None
                else 0.0
            )
            data_max = (
                float(np.max(result.max_magnitude_over_combo))
                if result.max_magnitude_over_combo is not None
                else 0.0
            )
            self._emit_display_payload(mesh, scalar_bar_title, data_min, data_max)

    def _build_display_output_flags(self) -> SolverOutputFlags:
        stress_type = (
            self.tab.combination_result.result_type
            if self.tab.combination_result is not None
            else None
        )
        return SolverOutputFlags(
            compute_von_mises=stress_type == "von_mises",
            compute_max_principal=stress_type == "max_principal",
            compute_min_principal=stress_type == "min_principal",
            compute_nodal_forces=self.tab.nodal_forces_result is not None,
            compute_deformation=self.tab.deformation_result is not None,
        )

    def _emit_display_payload(
        self,
        mesh,
        scalar_bar_title: str,
        data_min: float,
        data_max: float,
    ) -> None:
        combo_names = (
            self.tab.combination_table.combination_names
            if self.tab.combination_table is not None
            else []
        )
        payload = DisplayResultPayload(
            mesh=mesh,
            scalar_bar_title=scalar_bar_title,
            data_min=data_min,
            data_max=data_max,
            combination_names=list(combo_names),
            stress_result=self.tab.combination_result,
            forces_result=self.tab.nodal_forces_result,
            deformation_result=self.tab.deformation_result,
            output_flags=self._build_display_output_flags(),
        )
        self.tab.display_payload_ready.emit(payload)

    def _create_mesh_from_forces_result(self, result: NodalForcesResult):
        """Create a PyVista mesh from nodal forces results for visualization."""
        import pyvista as pv

        mesh = pv.PolyData(result.node_coords)
        mesh["NodeID"] = result.node_ids

        if result.max_magnitude_over_combo is not None:
            mesh["Max_Force_Magnitude"] = result.max_magnitude_over_combo
            mesh["Force_Magnitude"] = result.max_magnitude_over_combo
            mesh.set_active_scalars("Max_Force_Magnitude")

        if result.min_magnitude_over_combo is not None:
            mesh["Min_Force_Magnitude"] = result.min_magnitude_over_combo

        if result.combo_of_max is not None:
            mesh["Combo_of_Max"] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh["Combo_of_Min"] = result.combo_of_min

        if result.all_combo_fx is not None:
            num_nodes = result.num_nodes
            fx_at_max = np.zeros(num_nodes)
            fy_at_max = np.zeros(num_nodes)
            fz_at_max = np.zeros(num_nodes)

            if result.combo_of_max is not None:
                for node_idx in range(num_nodes):
                    combo_idx = int(result.combo_of_max[node_idx])
                    fx_at_max[node_idx] = result.all_combo_fx[combo_idx, node_idx]
                    fy_at_max[node_idx] = result.all_combo_fy[combo_idx, node_idx]
                    fz_at_max[node_idx] = result.all_combo_fz[combo_idx, node_idx]

            mesh["FX"] = fx_at_max
            mesh["FY"] = fy_at_max
            mesh["FZ"] = fz_at_max
            mesh["Shear_XY"] = np.sqrt(fx_at_max ** 2 + fy_at_max ** 2)
            mesh["Shear_XZ"] = np.sqrt(fx_at_max ** 2 + fz_at_max ** 2)
            mesh["Shear_YZ"] = np.sqrt(fy_at_max ** 2 + fz_at_max ** 2)
            mesh["Shear_Force"] = mesh["Shear_YZ"]

            mesh["Max_FX"] = np.max(result.all_combo_fx, axis=0)
            mesh["Min_FX"] = np.min(result.all_combo_fx, axis=0)
            mesh["Max_FY"] = np.max(result.all_combo_fy, axis=0)
            mesh["Min_FY"] = np.min(result.all_combo_fy, axis=0)
            mesh["Max_FZ"] = np.max(result.all_combo_fz, axis=0)
            mesh["Min_FZ"] = np.min(result.all_combo_fz, axis=0)
            mesh["Combo_of_Max_FX"] = np.argmax(result.all_combo_fx, axis=0)
            mesh["Combo_of_Min_FX"] = np.argmin(result.all_combo_fx, axis=0)
            mesh["Combo_of_Max_FY"] = np.argmax(result.all_combo_fy, axis=0)
            mesh["Combo_of_Min_FY"] = np.argmin(result.all_combo_fy, axis=0)
            mesh["Combo_of_Max_FZ"] = np.argmax(result.all_combo_fz, axis=0)
            mesh["Combo_of_Min_FZ"] = np.argmin(result.all_combo_fz, axis=0)

            shear_xy_all = np.sqrt(result.all_combo_fx ** 2 + result.all_combo_fy ** 2)
            shear_xz_all = np.sqrt(result.all_combo_fx ** 2 + result.all_combo_fz ** 2)
            shear_yz_all = np.sqrt(result.all_combo_fy ** 2 + result.all_combo_fz ** 2)
            mesh["Max_Shear_XY"] = np.max(shear_xy_all, axis=0)
            mesh["Min_Shear_XY"] = np.min(shear_xy_all, axis=0)
            mesh["Max_Shear_XZ"] = np.max(shear_xz_all, axis=0)
            mesh["Min_Shear_XZ"] = np.min(shear_xz_all, axis=0)
            mesh["Max_Shear_YZ"] = np.max(shear_yz_all, axis=0)
            mesh["Min_Shear_YZ"] = np.min(shear_yz_all, axis=0)
            mesh["Combo_of_Max_Shear_XY"] = np.argmax(shear_xy_all, axis=0)
            mesh["Combo_of_Min_Shear_XY"] = np.argmin(shear_xy_all, axis=0)
            mesh["Combo_of_Max_Shear_XZ"] = np.argmax(shear_xz_all, axis=0)
            mesh["Combo_of_Min_Shear_XZ"] = np.argmin(shear_xz_all, axis=0)
            mesh["Combo_of_Max_Shear_YZ"] = np.argmax(shear_yz_all, axis=0)
            mesh["Combo_of_Min_Shear_YZ"] = np.argmin(shear_yz_all, axis=0)

        return mesh

    def _create_mesh_from_deformation_result(self, result: DeformationResult):
        """Create a PyVista mesh from deformation results for visualization."""
        import pyvista as pv

        mesh = pv.PolyData(result.node_coords)
        mesh["NodeID"] = result.node_ids

        if result.max_magnitude_over_combo is not None:
            mesh["Max_U_mag"] = result.max_magnitude_over_combo
            mesh["U_mag"] = result.max_magnitude_over_combo
            mesh.set_active_scalars("Max_U_mag")

        if result.min_magnitude_over_combo is not None:
            mesh["Min_U_mag"] = result.min_magnitude_over_combo

        if result.combo_of_max is not None:
            mesh["Combo_of_Max"] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh["Combo_of_Min"] = result.combo_of_min

        if result.all_combo_ux is not None:
            num_nodes = result.num_nodes
            ux_at_max = np.zeros(num_nodes)
            uy_at_max = np.zeros(num_nodes)
            uz_at_max = np.zeros(num_nodes)

            if result.combo_of_max is not None:
                for node_idx in range(num_nodes):
                    combo_idx = int(result.combo_of_max[node_idx])
                    ux_at_max[node_idx] = result.all_combo_ux[combo_idx, node_idx]
                    uy_at_max[node_idx] = result.all_combo_uy[combo_idx, node_idx]
                    uz_at_max[node_idx] = result.all_combo_uz[combo_idx, node_idx]

            mesh["UX"] = ux_at_max
            mesh["UY"] = uy_at_max
            mesh["UZ"] = uz_at_max

        return mesh

    def _create_mesh_from_result(self, result: CombinationResult):
        """Create a PyVista mesh from stress results for visualization."""
        import pyvista as pv

        mesh = pv.PolyData(result.node_coords)
        mesh["NodeID"] = result.node_ids

        if result.max_over_combo is not None:
            mesh["Max_Stress"] = result.max_over_combo
            mesh.set_active_scalars("Max_Stress")

        if result.min_over_combo is not None:
            mesh["Min_Stress"] = result.min_over_combo

        if result.combo_of_max is not None:
            mesh["Combo_of_Max"] = result.combo_of_max
        if result.combo_of_min is not None:
            mesh["Combo_of_Min"] = result.combo_of_min

        mesh.field_data["result_type"] = [result.result_type]
        return mesh
