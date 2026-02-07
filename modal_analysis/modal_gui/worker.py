"""Background worker for modal extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from modal_gui import csv_writer, dpf_modal_extractor


@dataclass
class ExtractionJob:
    rst_path: str
    output_dir: str
    named_selection: str
    mode_count: int
    do_stress: bool
    do_strain: bool
    do_displacement: bool
    do_element_nodal: bool = False
    chunk_size: Optional[int] = None
    specific_mode: Optional[int] = None  # If set, extract only this mode


class ModalExtractionWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    canceled = pyqtSignal()

    def __init__(self, job: ExtractionJob):
        super().__init__()
        self._job = job
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _should_cancel(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            self._run_extraction()
            if self._cancel_requested:
                self.canceled.emit()
            else:
                self.finished.emit()
        except dpf_modal_extractor.ModalExtractionCanceled:
            self.canceled.emit()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run_extraction(self) -> None:
        job = self._job
        mode_count = max(1, int(job.mode_count))
        
        # Determine if extracting specific mode or range
        mode_ids = None
        if job.specific_mode is not None:
            mode_ids = [job.specific_mode]
            self.log.emit(f"Extracting specific mode: {job.specific_mode}")
        
        outputs = []

        if job.do_stress:
            outputs.append((
                "stress",
                "modal_stress_tensor_w_coords.csv",
                dpf_modal_extractor.extract_modal_stress_csv,
                False,
            ))
        if job.do_strain:
            outputs.append((
                "strain",
                "modal_strain_tensor_w_coords.csv",
                dpf_modal_extractor.extract_modal_strain_csv,
                False,
            ))
        if job.do_displacement:
            outputs.append((
                "displacement",
                "modal_directional_deformation_w_coords.csv",
                dpf_modal_extractor.extract_modal_displacement_csv,
                False,
            ))
        if job.do_element_nodal:
            outputs.append((
                "element_nodal_force_moment",
                "modal_element_nodal_forces_moments_w_coords.csv",
                dpf_modal_extractor.extract_modal_element_nodal_forces_moments_csv,
                False,
            ))

        if not outputs:
            raise ValueError("Select at least one result type to extract.")

        for label, filename, func, optional_if_unavailable in outputs:
            if self._should_cancel():
                raise dpf_modal_extractor.ModalExtractionCanceled("Canceled by user.")

            output_path = os.path.join(job.output_dir, filename)
            self.log.emit(f"Starting {label} extraction -> {output_path}")

            def progress_cb(current: int, total: int) -> None:
                self.progress.emit(current, total)

            # Use mode_ids if specific mode, otherwise use mode_count
            try:
                if mode_ids is not None:
                    func(
                        rst_path=job.rst_path,
                        output_csv_path=output_path,
                        named_selection=job.named_selection,
                        mode_ids=mode_ids,
                        chunk_size=job.chunk_size,
                        log_cb=self.log.emit,
                        progress_cb=progress_cb,
                        should_cancel=self._should_cancel,
                    )
                else:
                    func(
                        rst_path=job.rst_path,
                        output_csv_path=output_path,
                        named_selection=job.named_selection,
                        mode_count=mode_count,
                        chunk_size=job.chunk_size,
                        log_cb=self.log.emit,
                        progress_cb=progress_cb,
                        should_cancel=self._should_cancel,
                    )
            except dpf_modal_extractor.ModalExtractionError as exc:
                if optional_if_unavailable and "not available" in str(exc).lower():
                    self.log.emit(f"Skipping {label} extraction: {exc}")
                    continue
                raise

            self.log.emit(f"Finished {label} extraction.")
            
            # Validate output for zero value issues
            self.log.emit(f"Validating {label} output...")
            validation = csv_writer.validate_csv_for_zeros(
                output_path, 
                log_cb=self.log.emit
            )
            if not validation['valid']:
                if validation.get('all_zeros'):
                    self.log.emit(
                        f"CRITICAL WARNING: {label} output contains ALL ZEROS! "
                        "This indicates a node ID mismatch between the mesh and results."
                    )
                else:
                    self.log.emit(
                        f"WARNING: {label} output has {len(validation.get('high_zero_cols', []))} "
                        "columns with >50% zeros. Please verify results."
                    )
