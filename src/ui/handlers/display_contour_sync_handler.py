from __future__ import annotations

from typing import Optional

import numpy as np

from ui.handlers.display_base_handler import DisplayBaseHandler
from ui.handlers.display_contour_context import build_contour_context
from ui.handlers.display_contour_policy import (
    get_scalar_display_items,
    resolve_contour_type,
    should_show_contour_type_selector,
)
from ui.handlers.display_contour_types import ContourType
from ui.handlers.display_mesh_arrays import (
    attach_deformation_envelope_arrays,
    attach_deformation_specific_arrays,
    attach_force_component_arrays,
)


class DisplayContourSyncHandler(DisplayBaseHandler):
    """Synchronize contour family controls and active scalar selection."""

    FORCE_COMPONENT_MAP = {
        0: "Force_Magnitude",
        1: "FX",
        2: "FY",
        3: "FZ",
        4: "Shear_XY",
        5: "Shear_XZ",
        6: "Shear_YZ",
    }

    DISPLACEMENT_COMPONENT_MAP = {
        0: "U_mag",
        1: "UX",
        2: "UY",
        3: "UZ",
    }

    def sync_from_current_state(self) -> None:
        """Recompute contour options and apply the active family/view selection."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            self._set_current_contour_type(None)
            self._update_contour_selector([], None)
            self._set_scalar_display_visibility(False)
            self.tab._show_force_component_controls(False)
            self.tab._show_displacement_component_controls(False)
            return

        self._ensure_view_combination_options()
        context = build_contour_context(self.tab, mesh)
        active_type = self._resolve_active_type(context)

        self._update_contour_selector(context.available_types, active_type)
        self._sync_scalar_display_options(active_type, context)
        self._sync_family_control_visibility(active_type)

        self.apply_current_selection()

    def on_contour_type_changed(self, index: int) -> None:
        """Handle contour-family selection from combo box."""
        contour_type = self._combo_value_to_type(index)
        self._set_current_contour_type(contour_type)
        self.sync_from_current_state()

    def on_scalar_display_changed(self, _index: int) -> None:
        """Handle envelope scalar-display option changes."""
        self.apply_current_selection()

    def on_view_combination_changed(self, _index: int) -> None:
        """Handle envelope/specific-combination mode changes."""
        self.sync_from_current_state()

    def on_force_component_changed(self, _index: int) -> None:
        """Handle force-component option changes."""
        self.apply_current_selection()

    def on_displacement_component_changed(self, _index: int) -> None:
        """Handle displacement-component option changes."""
        self.apply_current_selection()

    def apply_current_selection(self) -> None:
        """Apply current family + display mode selection to mesh scalars."""
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None:
            return

        context = build_contour_context(self.tab, mesh)
        contour_type = self._resolve_active_type(context)
        if contour_type is None:
            return

        use_specific = (not context.is_envelope_view) and self._family_supports_specific(contour_type)

        if contour_type == ContourType.DEFORMATION and getattr(self.tab, "deformation_result", None) is not None:
            attach_deformation_envelope_arrays(mesh, self.tab.deformation_result)

        combo_idx = context.selected_combination_index
        if use_specific and combo_idx >= 0:
            if contour_type == ContourType.STRESS:
                self._attach_stress_specific_array(mesh, combo_idx)
            elif contour_type == ContourType.FORCES:
                attach_force_component_arrays(mesh, self.tab.nodal_forces_result, combo_idx)
            elif contour_type == ContourType.DEFORMATION:
                attach_deformation_specific_arrays(mesh, self.tab.deformation_result, combo_idx)

        if contour_type == ContourType.STRESS:
            array_name, legend_title = self._resolve_stress_array(mesh, context, use_specific)
        elif contour_type == ContourType.FORCES:
            array_name, legend_title = self._resolve_force_array(mesh, context, use_specific)
        else:
            array_name, legend_title = self._resolve_deformation_array(mesh, context, use_specific)

        if not array_name or array_name not in mesh.array_names:
            return

        mesh.set_active_scalars(array_name)
        self.state.data_column = legend_title
        self.tab.data_column = legend_title

        self._update_scalar_range(array_name)
        self.tab.update_visualization()

    def _resolve_active_type(self, context) -> Optional[ContourType]:
        inferred = context.inferred_active_type or self._infer_type_from_active_scalar(context.active_scalar_name)
        active_type = resolve_contour_type(
            current=getattr(self.state, "current_contour_type", None),
            available_types=context.available_types,
            inferred=inferred,
        )
        self._set_current_contour_type(active_type)
        return active_type

    def _set_current_contour_type(self, contour_type: Optional[ContourType]) -> None:
        value = contour_type.value if contour_type is not None else None
        self.state.current_contour_type = value
        self.tab.current_contour_type = value

    def _combo_value_to_type(self, index: int) -> Optional[ContourType]:
        combo = getattr(self.tab, "contour_type_combo", None)
        if combo is None:
            return None

        if index < 0 or index >= combo.count():
            return None

        value = combo.itemText(index)
        for contour_type in ContourType:
            if value == contour_type.value:
                return contour_type
        return None

    def _update_contour_selector(
        self,
        available_types: list[ContourType],
        active_type: Optional[ContourType],
    ) -> None:
        combo = getattr(self.tab, "contour_type_combo", None)
        label = getattr(self.tab, "contour_type_label", None)
        if combo is None or label is None:
            return

        combo.blockSignals(True)
        try:
            combo.clear()
            for contour_type in available_types:
                combo.addItem(contour_type.value)

            show_selector = should_show_contour_type_selector(available_types)
            label.setVisible(show_selector)
            combo.setVisible(show_selector)

            if active_type is not None:
                idx = combo.findText(active_type.value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def _set_scalar_display_visibility(self, visible: bool) -> None:
        self.tab.scalar_display_label.setVisible(visible)
        self.tab.scalar_display_combo.setVisible(visible)

    def _sync_scalar_display_options(self, contour_type: Optional[ContourType], context) -> None:
        if contour_type is None:
            self._set_scalar_display_visibility(False)
            return

        has_min = self._has_min_for_family(contour_type, context)
        items = get_scalar_display_items(has_min)

        combo = self.tab.scalar_display_combo
        current_text = combo.currentText()

        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItems(items)
            if current_text in items:
                combo.setCurrentText(current_text)
            else:
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(False)

        self._set_scalar_display_visibility(True)

        supports_specific = self._family_supports_specific(contour_type)
        enable_scalar = context.is_envelope_view or not supports_specific
        self.tab.scalar_display_combo.setEnabled(enable_scalar)
        self.tab.scalar_display_label.setEnabled(enable_scalar)

    def _sync_family_control_visibility(self, contour_type: Optional[ContourType]) -> None:
        if contour_type == ContourType.FORCES and self.tab.nodal_forces_result is not None:
            previous_index = self.tab.force_component_combo.currentIndex()
            self.tab._show_force_component_controls(
                True,
                has_beam_nodes=self.tab.nodal_forces_result.has_beam_nodes,
            )
            if 0 <= previous_index < self.tab.force_component_combo.count():
                self.tab.force_component_combo.blockSignals(True)
                try:
                    self.tab.force_component_combo.setCurrentIndex(previous_index)
                finally:
                    self.tab.force_component_combo.blockSignals(False)
        else:
            self.tab._show_force_component_controls(False)

        show_disp = contour_type == ContourType.DEFORMATION and self.tab.deformation_result is not None
        self.tab._show_displacement_component_controls(show_disp)

    def _ensure_view_combination_options(self) -> None:
        combo = getattr(self.tab, "view_combination_combo", None)
        label = getattr(self.tab, "view_combination_label", None)
        if combo is None or label is None:
            return

        names = list(getattr(self.tab, "combination_names", []) or [])
        supports_specific = any(
            [
                getattr(self.tab, "all_combo_results", None) is not None,
                getattr(getattr(self.tab, "nodal_forces_result", None), "all_combo_fx", None) is not None,
                getattr(getattr(self.tab, "deformation_result", None), "all_combo_ux", None) is not None,
            ]
        )

        show_controls = bool(names) and supports_specific

        if not show_controls:
            label.setVisible(False)
            combo.setVisible(False)
            return

        previous_index = combo.currentIndex()
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("(Envelope View)")
            for i, name in enumerate(names):
                combo.addItem(f"{i + 1}: {name}")

            if previous_index < 0 or previous_index >= combo.count():
                previous_index = 0
            combo.setCurrentIndex(previous_index)
        finally:
            combo.blockSignals(False)

        label.setVisible(True)
        combo.setVisible(True)

    def _family_supports_specific(self, contour_type: ContourType) -> bool:
        if contour_type == ContourType.STRESS:
            all_combo = getattr(self.tab, "all_combo_results", None)
            return all_combo is not None
        if contour_type == ContourType.FORCES:
            result = getattr(self.tab, "nodal_forces_result", None)
            return result is not None and getattr(result, "all_combo_fx", None) is not None
        if contour_type == ContourType.DEFORMATION:
            result = getattr(self.tab, "deformation_result", None)
            return result is not None and getattr(result, "all_combo_ux", None) is not None
        return False

    def _has_min_for_family(self, contour_type: ContourType, context) -> bool:
        if contour_type == ContourType.STRESS:
            return context.has_min_stress
        if contour_type == ContourType.FORCES:
            return context.has_min_forces
        if contour_type == ContourType.DEFORMATION:
            return context.has_min_deformation
        return False

    @staticmethod
    def _infer_type_from_active_scalar(active_scalar_name: Optional[str]) -> Optional[ContourType]:
        if not active_scalar_name:
            return None

        if active_scalar_name.startswith("Def_"):
            return ContourType.DEFORMATION
        if active_scalar_name in {"U_mag", "UX", "UY", "UZ", "Max_U_mag", "Min_U_mag"}:
            return ContourType.DEFORMATION

        if active_scalar_name in {
            "Force_Magnitude",
            "Max_Force_Magnitude",
            "Min_Force_Magnitude",
            "FX",
            "FY",
            "FZ",
            "Shear_XY",
            "Shear_XZ",
            "Shear_YZ",
            "Shear_Force",
        }:
            return ContourType.FORCES

        if active_scalar_name.startswith("Max_F") or active_scalar_name.startswith("Min_F"):
            return ContourType.FORCES
        if active_scalar_name.startswith("Max_Shear_") or active_scalar_name.startswith("Min_Shear_"):
            return ContourType.FORCES
        if active_scalar_name.startswith("Combo_of_Max_F") or active_scalar_name.startswith("Combo_of_Min_F"):
            return ContourType.FORCES
        if active_scalar_name.startswith("Combo_of_Max_Shear_") or active_scalar_name.startswith("Combo_of_Min_Shear_"):
            return ContourType.FORCES

        if active_scalar_name in {"Max_Stress", "Min_Stress"}:
            return ContourType.STRESS
        if active_scalar_name.endswith("_Stress") and active_scalar_name.startswith("Combo_"):
            return ContourType.STRESS

        return None

    def _attach_stress_specific_array(self, mesh, combo_idx: int) -> None:
        all_combo = getattr(self.tab, "all_combo_results", None)
        if all_combo is None:
            return
        if combo_idx < 0 or combo_idx >= all_combo.shape[0]:
            return

        values = np.asarray(all_combo[combo_idx, :]).reshape(-1)
        if values.shape[0] != mesh.n_points:
            return

        mesh[f"Combo_{combo_idx}_Stress"] = values

    def _resolve_stress_array(self, mesh, context, use_specific: bool) -> tuple[str, str]:
        if use_specific and context.selected_combination_index >= 0:
            combo_idx = context.selected_combination_index
            array_name = f"Combo_{combo_idx}_Stress"
            if array_name in mesh.array_names:
                stress_label = self.tab._get_stress_type_label()
                return array_name, f"Combo_{combo_idx + 1}_{stress_label} [MPa]"

        selected = context.scalar_display_text
        wants_combo = "Combo # of" in selected
        wants_min = "Min" in selected

        if wants_combo:
            array_name = "Combo_of_Min" if wants_min else "Combo_of_Max"
            if array_name in mesh.array_names:
                return array_name, array_name

        array_name = "Min_Stress" if wants_min else "Max_Stress"
        if array_name not in mesh.array_names:
            array_name = "Max_Stress"

        stress_label = self.tab._get_stress_type_label()
        prefix = "Min" if array_name.startswith("Min") else "Max"
        return array_name, f"{prefix}_{stress_label} [MPa]"

    def _resolve_force_array(self, mesh, context, use_specific: bool) -> tuple[str, str]:
        force_unit = "N"
        if self.tab.nodal_forces_result is not None:
            force_unit = self.tab.nodal_forces_result.force_unit

        base_name = self.FORCE_COMPONENT_MAP.get(context.force_component_index, "Force_Magnitude")

        if use_specific:
            array_name = base_name
            if array_name == "Shear_YZ" and array_name not in mesh.array_names:
                array_name = "Shear_Force"
            if array_name not in mesh.array_names:
                array_name = "Force_Magnitude"
            return array_name, f"{array_name} [{force_unit}]"

        selected = context.scalar_display_text
        wants_combo = "Combo # of" in selected
        wants_min = "Min" in selected

        if wants_combo:
            if base_name == "Force_Magnitude":
                array_name = "Combo_of_Min" if wants_min else "Combo_of_Max"
            else:
                prefix = "Min" if wants_min else "Max"
                array_name = f"Combo_of_{prefix}_{base_name}"
            if array_name in mesh.array_names:
                return array_name, array_name

        if base_name == "Force_Magnitude":
            array_name = "Min_Force_Magnitude" if wants_min else "Max_Force_Magnitude"
            if array_name not in mesh.array_names:
                array_name = "Max_Force_Magnitude"
            return array_name, f"Force_Magnitude [{force_unit}]"

        prefix = "Min" if wants_min else "Max"
        array_name = f"{prefix}_{base_name}"
        if array_name not in mesh.array_names:
            array_name = f"Max_{base_name}" if f"Max_{base_name}" in mesh.array_names else base_name
        return array_name, f"{array_name} [{force_unit}]"

    def _resolve_deformation_array(self, mesh, context, use_specific: bool) -> tuple[str, str]:
        disp_unit = "mm"
        if self.tab.deformation_result is not None:
            disp_unit = self.tab.deformation_result.displacement_unit

        base_name = self.DISPLACEMENT_COMPONENT_MAP.get(context.displacement_component_index, "U_mag")

        if use_specific:
            array_name = base_name
            if array_name not in mesh.array_names:
                array_name = "U_mag"
            return array_name, f"{array_name} [{disp_unit}]"

        selected = context.scalar_display_text
        wants_combo = "Combo # of" in selected
        wants_min = "Min" in selected

        if wants_combo:
            prefix = "Min" if wants_min else "Max"
            array_name = f"Def_Combo_of_{prefix}_{base_name}"
            if array_name in mesh.array_names:
                return array_name, array_name

        prefix = "Min" if wants_min else "Max"
        array_name = f"Def_{prefix}_{base_name}"
        if array_name not in mesh.array_names:
            fallback = "Min_U_mag" if wants_min else "Max_U_mag"
            if fallback in mesh.array_names:
                array_name = fallback
            else:
                array_name = "U_mag"
        return array_name, f"{base_name} [{disp_unit}]"

    def _update_scalar_range(self, array_name: str) -> None:
        mesh = self.state.current_mesh or self.tab.current_mesh
        if mesh is None or array_name not in mesh.array_names:
            return

        data = np.asarray(mesh[array_name]).reshape(-1)
        if data.size == 0:
            return

        data_min = float(np.min(data))
        data_max = float(np.max(data))

        if data_min == data_max:
            eps = abs(data_min) * 0.01 or 0.01
            data_min -= eps
            data_max += eps

        self.tab.scalar_min_spin.blockSignals(True)
        self.tab.scalar_max_spin.blockSignals(True)
        try:
            self.tab.scalar_min_spin.setRange(data_min, data_max)
            self.tab.scalar_max_spin.setRange(data_min, 1e30)
            self.tab.scalar_min_spin.setValue(data_min)
            self.tab.scalar_max_spin.setValue(data_max)
        finally:
            self.tab.scalar_min_spin.blockSignals(False)
            self.tab.scalar_max_spin.blockSignals(False)
