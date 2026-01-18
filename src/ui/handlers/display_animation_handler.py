"""
Animation lifecycle management for the Display tab.
"""

import gc
import time
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from utils.constants import NP_DTYPE
from ui.handlers.display_base_handler import DisplayBaseHandler
from core.visualization import AnimationManager


class DisplayAnimationHandler(DisplayBaseHandler):
    """Encapsulates animation control, timers, and frame updates."""

    def __init__(self, tab, state, anim_manager: AnimationManager):
        super().__init__(tab, state)
        self.anim_manager = anim_manager

    # ------------------------------------------------------------------
    # Public API called from DisplayTab slots
    # ------------------------------------------------------------------
    def start_animation(self) -> None:
        """Start animation playback or resume if paused."""
        tab = self.tab

        if tab.current_mesh is None:
            QMessageBox.warning(
                tab, "No Data",
                "Please load or initialize the mesh before animating."
            )
            return

        tracked_node_index = tab.target_node_index
        tracked_node_id = tab.target_node_id
        is_frozen = tab.freeze_tracked_node

        if tracked_node_index is not None and is_frozen:
            if tab.target_node_marker_actor:
                tab.target_node_marker_actor.SetVisibility(False)
            if tab.target_node_label_actor:
                tab.target_node_label_actor.SetVisibility(False)
            tab.plotter.render()

        if self.state.animation_paused:
            if self.anim_manager.precomputed_scalars is None:
                QMessageBox.warning(
                    tab, "Resume Error",
                    "Cannot resume. Precomputed data is missing. Please stop and start again."
                )
                self.stop_animation()
                return

            print("Resuming animation...")
            self.set_state_attr("animation_paused", False)
            tab.pause_button.setEnabled(True)
            tab.stop_button.setEnabled(True)
            tab.play_button.setEnabled(False)
            tab.deformation_scale_edit.setEnabled(False)

            timer = self.state.anim_timer
            if timer:
                timer.start(tab.anim_interval_spin.value())
            else:
                timer = QTimer(tab)
                timer.timeout.connect(self._animate_frame)
                timer.start(tab.anim_interval_spin.value())
            self.set_state_attr("anim_timer", timer)
            return

        # Start fresh
        self.stop_animation()
        tab.play_button.setEnabled(False)

        if tracked_node_index is not None:
            tab.target_node_index = tracked_node_index
            tab.target_node_id = tracked_node_id
            tab.freeze_tracked_node = is_frozen

            if tab.freeze_tracked_node and tab.original_node_coords is not None:
                if tab.current_mesh is not None and tab.current_mesh.n_points > tracked_node_index:
                    tab.freeze_baseline = tab.current_mesh.points[tab.target_node_index].copy()
                else:
                    print("Warning: Cannot set freeze_baseline")
                    tab.freeze_tracked_node = False

        anim_times, anim_indices, error_msg = self._get_animation_time_steps()
        if error_msg:
            QMessageBox.warning(tab, "Animation Setup Error", error_msg)
            tab.play_button.setEnabled(True)
            return
        if anim_times is None or len(anim_times) == 0:
            QMessageBox.warning(tab, "Animation Setup Error", "No time steps generated.")
            tab.play_button.setEnabled(True)
            return

        main_tab = tab.window().solver_tab
        selected_outputs = [
            main_tab.von_mises_checkbox.isChecked(),
            main_tab.max_principal_stress_checkbox.isChecked(),
            main_tab.min_principal_stress_checkbox.isChecked(),
            main_tab.deformation_checkbox.isChecked(),
            main_tab.velocity_checkbox.isChecked(),
            main_tab.acceleration_checkbox.isChecked()
        ]

        num_selected = sum(selected_outputs)
        if num_selected == 0:
            QMessageBox.warning(tab, "No Selection", "No valid output is selected for animation.")
            tab.play_button.setEnabled(True)
            return
        if num_selected > 1:
            QMessageBox.warning(
                tab, "Error - Multiple Selections",
                "Please select only one output type for animation playback.\n\n"
                "Animation currently supports displaying one output at a time for clarity.\n"
                "You can switch between different outputs using the checkboxes and run separate animations."
            )
            tab.play_button.setEnabled(True)
            return

        params = {
            "compute_von_mises": main_tab.von_mises_checkbox.isChecked(),
            "compute_max_principal": main_tab.max_principal_stress_checkbox.isChecked(),
            "compute_min_principal": main_tab.min_principal_stress_checkbox.isChecked(),
            "compute_deformation_anim": main_tab.deformations_checkbox.isChecked(),
            "compute_deformation_contour": main_tab.deformation_checkbox.isChecked(),
            "compute_velocity": main_tab.velocity_checkbox.isChecked(),
            "compute_acceleration": main_tab.acceleration_checkbox.isChecked(),
            "include_steady": main_tab.steady_state_checkbox.isChecked(),
            "skip_n_modes": int(main_tab.skip_modes_combo.currentText())
            if main_tab.skip_modes_combo.currentText() else 0,
            "scale_factor": float(tab.deformation_scale_edit.text()),
            "anim_indices": anim_indices,
            "show_absolute_deformation": tab.absolute_deformation_checkbox.isChecked(),
        }

        QApplication.setOverrideCursor(Qt.WaitCursor)
        print("DisplayTab: Delegating animation precomputation...")
        tab.animation_precomputation_requested.emit(params)

    def pause_animation(self) -> None:
        """Pause animation playback."""
        timer = self.state.anim_timer
        tab = self.tab

        if timer is not None and timer.isActive():
            timer.stop()
            self.set_state_attr("animation_paused", True)
            tab.play_button.setEnabled(True)
            tab.pause_button.setEnabled(False)

            if tab.target_node_index is not None and tab.freeze_tracked_node:
                if tab.target_node_marker_actor:
                    tab.target_node_marker_actor.SetVisibility(True)
                if tab.target_node_label_actor:
                    tab.target_node_label_actor.SetVisibility(True)
                tab.plotter.render()

            print("\nAnimation paused.")
        else:
            print("\nPause command ignored: Animation timer not active.")

    def stop_animation(self) -> None:
        """Stop animation, release precomputed data, and reset state."""
        tab = self.tab
        timer = self.state.anim_timer
        has_data = timer is not None or self.anim_manager.precomputed_scalars is not None

        tab.interaction_handler.clear_goto_node_markers()

        if has_data:
            print("\nStopping animation and releasing resources...")

        if timer is not None:
            timer.stop()
            try:
                timer.timeout.disconnect(self._animate_frame)
            except TypeError:
                pass
            self.set_state_attr("anim_timer", None)

        if self.anim_manager.precomputed_scalars is not None:
            print("Released precomputed scalars.")
        self.anim_manager.reset()
        gc.collect()

        self.set_state_attr("current_anim_frame_index", 0)
        self.set_state_attr("animation_paused", False)
        self.set_state_attr("is_deformation_included_in_anim", False)

        tab.deformation_scale_edit.setEnabled(True)
        tab.play_button.setEnabled(True)
        tab.pause_button.setEnabled(False)
        tab.stop_button.setEnabled(False)
        tab.save_anim_button.setEnabled(False)

        if self.state.time_text_actor is not None:
            try:
                tab.plotter.remove_actor(self.state.time_text_actor)
            except Exception:
                pass
            self.set_state_attr("time_text_actor", None)

        if tab.current_mesh is not None and tab.original_node_coords is not None:
            print("Resetting mesh to original coordinates.")
            try:
                if tab.current_mesh.n_points == tab.original_node_coords.shape[0]:
                    tab.current_mesh.points = tab.original_node_coords.copy()
                    try:
                        tab.current_mesh.points_modified()
                    except AttributeError:
                        tab.current_mesh.GetPoints().Modified()
                    tab.plotter.render()
            except Exception as exc:
                print(f"Error resetting mesh points: {exc}")

        if has_data:
            print("\nAnimation stopped.")

    def save_animation(self) -> None:
        """Save animation to file (MP4 or GIF)."""
        tab = self.tab
        if self.anim_manager.precomputed_scalars is None:
            QMessageBox.warning(
                tab, "No Animation",
                "No animation has been precomputed. Please play the animation first."
            )
            return

        file_path, file_format = self.get_save_path_and_format()
        if not file_path:
            return

        try:
            success = self.write_animation_to_file(file_path, file_format)
            if success:
                QMessageBox.information(
                    tab, "Save Successful",
                    f"Animation saved successfully to:\n{file_path}"
                )
        except Exception as exc:
            QMessageBox.critical(
                tab, "Save Error",
                f"Failed to save animation:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_animation_time_steps(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Determine animation time steps based on user settings."""
        time_values = self.state.time_values if self.state.time_values is not None else self.tab.time_values
        tab = self.tab

        if time_values is None or len(time_values) == 0:
            return None, None, "Time values not loaded."

        start_time = tab.anim_start_spin.value()
        end_time = tab.anim_end_spin.value()

        if start_time >= end_time:
            return None, None, "Animation start time must be less than end time."

        anim_times_list = []
        anim_indices_list = []

        if tab.time_step_mode_combo.currentText() == "Custom Time Step":
            step = tab.custom_step_spin.value()
            if step <= 0:
                return None, None, "Custom time step must be positive."

            current_t = start_time
            last_added_idx = -1

            while current_t <= end_time:
                idx = np.argmin(np.abs(time_values - current_t))

                if (time_values[idx] >= start_time and
                        time_values[idx] <= end_time and
                        idx != last_added_idx):
                    anim_indices_list.append(idx)
                    anim_times_list.append(time_values[idx])
                    last_added_idx = idx

                if current_t + step <= current_t:
                    print("Warning: Custom time step too small, breaking loop.")
                    break
                current_t += step

            end_idx = np.argmin(np.abs(time_values - end_time))
            if (time_values[end_idx] >= start_time and
                    time_values[end_idx] <= end_time):
                if not anim_indices_list or end_idx != anim_indices_list[-1]:
                    anim_indices_list.append(end_idx)
                    anim_times_list.append(time_values[end_idx])

        else:
            nth = tab.actual_interval_spin.value()
            if nth <= 0:
                return None, None, "Actual data step interval must be positive."

            valid_indices = np.where(
                (time_values >= start_time) & (time_values <= end_time)
            )[0]

            if len(valid_indices) == 0:
                return None, None, "No data time steps found in specified range."

            selected_indices = valid_indices[::nth].tolist()

            first_idx = valid_indices[0]
            if first_idx not in selected_indices:
                selected_indices.insert(0, first_idx)

            last_idx = valid_indices[-1]
            if not selected_indices or last_idx != selected_indices[-1]:
                selected_indices.append(last_idx)

            anim_indices_list = selected_indices
            anim_times_list = time_values[anim_indices_list].tolist()

        if not anim_times_list:
            return None, None, "No animation frames generated."

        unique_indices, order_indices = np.unique(anim_indices_list, return_index=True)
        final_indices = unique_indices[np.argsort(order_indices)]
        final_times = time_values[final_indices]

        return np.array(final_times), np.array(final_indices, dtype=int), None

    def estimate_animation_ram(self, num_nodes: int, num_anim_steps: int, include_deformation: bool) -> float:
        """Estimate peak RAM needed for animation precomputation in GB."""
        element_size = np.dtype(NP_DTYPE).itemsize

        normal_stress_ram = num_nodes * 6 * num_anim_steps * element_size
        scalar_ram = num_nodes * num_anim_steps * element_size

        intermediate_displacement_ram = 0
        final_coordinate_ram = 0
        if include_deformation:
            intermediate_displacement_ram = num_nodes * 3 * num_anim_steps * element_size
            final_coordinate_ram = num_nodes * 3 * num_anim_steps * element_size

        total_ram_bytes = (
            normal_stress_ram + scalar_ram +
            intermediate_displacement_ram + final_coordinate_ram
        )
        total_ram_bytes *= 1.20  # 20% safety buffer

        return total_ram_bytes / (1024 ** 3)

    def animate_frame(self, update_index: bool = True) -> None:
        """Public wrapper to advance the animation by one frame."""
        self._animate_frame(update_index=update_index)

    def get_save_path_and_format(self) -> Tuple[Optional[str], Optional[str]]:
        """Public wrapper for save-path selection."""
        return self._get_save_path_and_format()

    def write_animation_to_file(self, file_path: str, file_format: str) -> bool:
        """Public wrapper for writing animation frames."""
        return self._write_animation_to_file(file_path, file_format)

    def _animate_frame(self, update_index: bool = True) -> None:
        """Update display using next precomputed animation frame."""
        if (self.anim_manager.precomputed_scalars is None or
                self.anim_manager.precomputed_anim_times is None):
            print("Animation frame skipped: Precomputed data not available.")
            self.stop_animation()
            return

        if not self._update_mesh_for_frame(self.state.current_anim_frame_index):
            print("Animation frame skipped: Failed to update mesh.")
            self.stop_animation()
            return

        self.tab.plotter.render()

        if update_index:
            num_frames = self.anim_manager.get_num_frames()
            if num_frames > 0:
                next_index = self.state.current_anim_frame_index + 1
                if next_index >= num_frames:
                    next_index = 0
                self.set_state_attr("current_anim_frame_index", next_index)
            else:
                self.stop_animation()

    def _update_mesh_for_frame(self, frame_index: int) -> bool:
        """Update mesh data for given animation frame."""
        tab = self.tab

        if (self.anim_manager.precomputed_scalars is None or
                self.anim_manager.precomputed_anim_times is None or
                tab.current_mesh is None):
            return False

        num_frames = self.anim_manager.get_num_frames()
        if frame_index < 0 or frame_index >= num_frames:
            return False

        try:
            scalars, coords, time_val = self.anim_manager.get_frame_data(frame_index)

            if coords is not None:
                current_coords = coords.copy()

                if tab.freeze_tracked_node and tab.freeze_baseline is not None:
                    tracked_now = current_coords[tab.target_node_index]
                    shift = tracked_now - tab.freeze_baseline
                    if np.any(shift):
                        current_coords -= shift
                        if tab.marker_poly is not None:
                            tab.marker_poly.points[:] -= shift
                            tab.marker_poly.Modified()

                tab.current_mesh.points = current_coords
                try:
                    tab.current_mesh.points_modified()
                except AttributeError:
                    tab.current_mesh.GetPoints().Modified()

            data_column = self.anim_manager.data_column_name
            tab.current_mesh[data_column] = scalars
            if tab.current_mesh.active_scalars_name != data_column:
                tab.current_mesh.set_active_scalars(data_column)

            time_text = f"Time: {time_val:.5f} s"
            if self.state.time_text_actor is not None:
                try:
                    self.state.time_text_actor.SetInput(time_text)
                except Exception:
                    try:
                        tab.plotter.remove_actor(self.state.time_text_actor, render=False)
                    except Exception:
                        pass
                    actor = tab.plotter.add_text(
                        time_text, position=(0.8, 0.9), viewport=False, font_size=10
                    )
                    self.set_state_attr("time_text_actor", actor)
            else:
                actor = tab.plotter.add_text(
                    time_text, position=(0.8, 0.9), viewport=False, font_size=10
                )
                self.set_state_attr("time_text_actor", actor)

            if tab.target_node_index is not None:
                new_coords = tab.current_mesh.points[tab.target_node_index]
                if (not tab.freeze_tracked_node and tab.target_node_label_actor
                        and tab.label_point_data):
                    tab.label_point_data.points[0, :] = new_coords
                    tab.label_point_data.Modified()
                if tab.target_node_marker_actor and tab.marker_poly:
                    tab.marker_poly.points[0] = new_coords
                    tab.marker_poly.Modified()

            return True

        except Exception as exc:
            print(f"Error updating mesh for frame {frame_index}: {exc}")
            return False

    def _get_save_path_and_format(self) -> Tuple[Optional[str], Optional[str]]:
        """Open a save dialog and return (file_path, format)."""
        tab = self.tab
        default_name = "animation.mp4"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            tab,
            "Save Animation",
            default_name,
            "MP4 Video (*.mp4);;GIF Animation (*.gif)"
        )

        if not file_path:
            return None, None

        if selected_filter.endswith("*.mp4") and not file_path.lower().endswith(".mp4"):
            file_path += ".mp4"
            file_format = "mp4"
        elif selected_filter.endswith("*.gif") and not file_path.lower().endswith(".gif"):
            file_path += ".gif"
            file_format = "gif"
        else:
            file_format = "mp4" if file_path.lower().endswith(".mp4") else "gif"

        return file_path, file_format

    def _write_animation_to_file(self, file_path: str, file_format: str) -> bool:
        """Render precomputed frames and write them to disk."""
        tab = self.tab

        if file_format == "mp4":
            try:
                pv.start_xvfb()
            except Exception as exc:
                print(f"Warning: Failed to start xvfb: {exc}")

        original_position = tab.plotter.camera_position
        temp_plotter = pv.Plotter(off_screen=True)
        temp_plotter.background_color = tab.plotter.background_color

        output_frames = []
        num_frames = self.anim_manager.get_num_frames()

        print(f"Saving animation to {file_path} ...")
        for frame_index in range(num_frames):
            if not self._update_mesh_for_frame(frame_index):
                print(f"Skipping frame {frame_index} due to update failure.")
                continue

            temp_plotter.clear()
            temp_plotter.add_mesh(
                tab.current_mesh.copy(deep=True),
                scalars=self.anim_manager.data_column_name,
                cmap="jet",
                clim=tab.current_actor.mapper.scalar_range if tab.current_actor else None,
                render_points_as_spheres=True
            )
            temp_plotter.camera_position = original_position
            img = temp_plotter.screenshot(return_img=True, window_size=(1280, 720))
            output_frames.append(img)

        if not output_frames:
            QMessageBox.warning(
                tab, "Save Error",
                "No frames were generated for the animation."
            )
            return False

        if file_format == "mp4":
            try:
                import imageio.v2 as imageio

                imageio.mimsave(file_path, output_frames, fps=max(1, 1000 // tab.anim_interval_spin.value()))
            except Exception as exc:
                raise RuntimeError(f"Failed to write MP4:\n{exc}")
        else:
            try:
                import imageio.v2 as imageio

                imageio.mimsave(file_path, output_frames, duration=tab.anim_interval_spin.value() / 1000.0)
            except Exception as exc:
                raise RuntimeError(f"Failed to write GIF:\n{exc}")

        time.sleep(0.01)
        print(f"Animation saved to {file_path}")

        return True
