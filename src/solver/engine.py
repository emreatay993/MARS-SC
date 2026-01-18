# ---- Standard Library Imports ----
import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

# ---- Third-Party Imports ----
import psutil
from numba import njit, prange
import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

# Import torch via setup module (handles Windows CUDA DLL compatibility)
from utils.torch_setup import torch

# ---- Local Imports ----
import utils.constants as constants
from solver.plasticity_engine import (
    MaterialDB,
    apply_glinka_correction,
    apply_ibg_correction,
    apply_neuber_correction,
    von_mises_from_voigt,
)


@dataclass
class PlasticityRuntimeContext:
    """Runtime parameters for applying plasticity corrections."""

    method: str
    material_db: MaterialDB
    temperatures: Optional[np.ndarray] = None
    max_iterations: int = 60
    tolerance: float = 1e-10
    poisson_ratio: float = 0.3
    default_temperature: float = 22.0
    use_plateau: bool = False

class MSUPSmartSolverTransient(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, modal_sx, modal_sy, modal_sz, modal_sxy, modal_syz, modal_sxz, modal_coord, time_values,
                 steady_sx=None, steady_sy=None, steady_sz=None, steady_sxy=None, steady_syz=None, steady_sxz=None,
                 steady_node_ids=None, modal_node_ids=None, output_directory=None, modal_deformations=None):
        super().__init__()

        # Initializing class attributes used
        self.total_memory = None
        self.available_memory = None
        self.allocated_memory = None

        self.max_over_time_s1 = None
        self.min_over_time_s3 = None
        self.max_over_time_svm = None
        self.max_over_time_def = None
        self.max_over_time_vel = None
        self.max_over_time_acc = None

        self.max_over_time_svm_corrected = None
        self.plasticity_context: Optional[PlasticityRuntimeContext] = None

        self.fatigue_A = None
        self.fatigue_m = None


        # Use selected output directory or fallback to script location
        self.output_directory = output_directory if output_directory else os.path.dirname(os.path.abspath(__file__))

        # Global settings (accessed directly from constants module)
        self.device = torch.device("cuda" if constants.IS_GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else "cpu")

        self.modal_coord = torch.tensor(modal_coord, dtype=constants.TORCH_DTYPE).to(self.device)

        if modal_deformations is not None:
            self.modal_deformations_ux = torch.tensor(modal_deformations[0], dtype=constants.TORCH_DTYPE).to(self.device)
            self.modal_deformations_uy = torch.tensor(modal_deformations[1], dtype=constants.TORCH_DTYPE).to(self.device)
            self.modal_deformations_uz = torch.tensor(modal_deformations[2], dtype=constants.TORCH_DTYPE).to(self.device)
        else:
            self.modal_deformations_ux = None
            self.modal_deformations_uy = None
            self.modal_deformations_uz = None

        # Initialize modal inputs
        self.modal_sx = torch.tensor(modal_sx, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_sy = torch.tensor(modal_sy, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_sz = torch.tensor(modal_sz, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_sxy = torch.tensor(modal_sxy, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_syz = torch.tensor(modal_syz, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_sxz = torch.tensor(modal_sxz, dtype=constants.TORCH_DTYPE).to(self.device)
        self.modal_coord = torch.tensor(modal_coord, dtype=constants.TORCH_DTYPE).to(self.device)

        # Store modal node IDs
        self.modal_node_ids = modal_node_ids

        # If steady-state stress data is provided, process it
        if steady_sx is not None and steady_node_ids is not None:
            self.is_steady_state_included = True
            # Map steady-state stresses to modal nodes
            self.steady_sx = self.map_steady_state_stresses(steady_sx, steady_node_ids, modal_node_ids)
            self.steady_sy = self.map_steady_state_stresses(steady_sy, steady_node_ids, modal_node_ids)
            self.steady_sz = self.map_steady_state_stresses(steady_sz, steady_node_ids, modal_node_ids)
            self.steady_sxy = self.map_steady_state_stresses(steady_sxy, steady_node_ids, modal_node_ids)
            self.steady_syz = self.map_steady_state_stresses(steady_syz, steady_node_ids, modal_node_ids)
            self.steady_sxz = self.map_steady_state_stresses(steady_sxz, steady_node_ids, modal_node_ids)

            # Convert to torch tensors and move to device
            self.steady_sx = torch.tensor(self.steady_sx, dtype=constants.TORCH_DTYPE).to(self.device)
            self.steady_sy = torch.tensor(self.steady_sy, dtype=constants.TORCH_DTYPE).to(self.device)
            self.steady_sz = torch.tensor(self.steady_sz, dtype=constants.TORCH_DTYPE).to(self.device)
            self.steady_sxy = torch.tensor(self.steady_sxy, dtype=constants.TORCH_DTYPE).to(self.device)
            self.steady_syz = torch.tensor(self.steady_syz, dtype=constants.TORCH_DTYPE).to(self.device)
            self.steady_sxz = torch.tensor(self.steady_sxz, dtype=constants.TORCH_DTYPE).to(self.device)
        else:
            self.is_steady_state_included = False

        # Store time axis once for gradient calcs
        self.time_values = time_values.astype(constants.NP_DTYPE)

    # region Memory Management
    def _is_gpu_mode(self) -> bool:
        """Check if solver is running in GPU mode."""
        return self.device.type == 'cuda'

    def _get_gpu_memory_info(self) -> dict:
        """
        Get GPU memory information from both driver and PyTorch levels.
        
        Returns:
            dict with keys:
                - total: Total VRAM on device
                - used: Actual VRAM used (driver-level, matches Task Manager)
                - free: Actual VRAM free (driver-level)
                - pytorch_allocated: Memory used by PyTorch tensors
                - pytorch_reserved: Memory reserved by PyTorch's caching allocator
                - available_for_chunks: Memory available for new chunk allocations
        """
        if not self._is_gpu_mode():
            return {
                'total': 0, 'used': 0, 'free': 0,
                'pytorch_allocated': 0, 'pytorch_reserved': 0,
                'available_for_chunks': 0
            }
        
        # Driver-level memory info (matches Task Manager)
        free_driver, total_driver = torch.cuda.mem_get_info(self.device)
        used_driver = total_driver - free_driver
        
        # PyTorch allocator-level info (for debugging)
        pytorch_allocated = torch.cuda.memory_allocated(self.device)
        pytorch_reserved = torch.cuda.memory_reserved(self.device)
        
        # Available for new chunks = driver-level free memory with safety margin
        available_for_chunks = int(free_driver * constants.GPU_MEMORY_PERCENT)
        
        return {
            'total': total_driver,
            'used': used_driver,
            'free': free_driver,
            'pytorch_allocated': pytorch_allocated,
            'pytorch_reserved': pytorch_reserved,
            'available_for_chunks': max(0, available_for_chunks)
        }

    def _get_available_memory(self) -> int:
        """
        Get available memory for computation based on device type.
        
        Returns:
            Available memory in bytes.
        """
        if self._is_gpu_mode():
            return self._get_gpu_memory_info()['available_for_chunks']
        else:
            return int(psutil.virtual_memory().available * constants.RAM_PERCENT)

    def _get_current_memory_usage_str(self) -> str:
        """Get a formatted string of current memory usage for logging."""
        if self._is_gpu_mode():
            info = self._get_gpu_memory_info()
            used_pct = (info['used'] / info['total']) * 100 if info['total'] > 0 else 0
            return (f"GPU VRAM - Used: {info['used'] / (1024**3):.2f} GB / "
                    f"{info['total'] / (1024**3):.2f} GB ({used_pct:.1f}%)")
        else:
            mem = psutil.virtual_memory()
            return f"RAM Available: {mem.available / (1024**3):.2f} GB"

    def _estimate_chunk_size(self, num_time_points, calculate_von_mises, calculate_max_principal_stress,
                             calculate_damage, calculate_deformation=False,
                             calculate_velocity=False, calculate_acceleration=False):
        """
        Calculate the optimal chunk size for processing based on available memory.
        
        For GPU mode: Uses GPU VRAM, only counting tensors that actually reside on GPU.
        For CPU mode: Uses system RAM for all arrays.
        """
        available_memory = self._get_available_memory()
        
        if self._is_gpu_mode():
            # GPU mode: Only the 6 stress matmul outputs + 3 deformation outputs are on GPU
            # Von Mises, principal stresses, etc. are computed on CPU with Numba
            memory_per_node = self._get_gpu_memory_per_node(
                num_time_points,
                calculate_deformation or calculate_velocity or calculate_acceleration
            )
        else:
            # CPU mode: All arrays are in RAM
            memory_per_node = self._get_memory_per_node(
                num_time_points,
                calculate_von_mises,
                calculate_max_principal_stress,
                calculate_damage,
                calculate_deformation,
                calculate_velocity,
                calculate_acceleration
            )
        
        max_nodes_per_iteration = available_memory // memory_per_node
        return max(1, int(max_nodes_per_iteration))
    
    def _get_gpu_memory_per_node(self, num_time_points, calculate_kinematics=False):
        """
        Calculate GPU VRAM required per node for GPU-resident tensors only.
        
        Only counts tensors that actually live on GPU during computation:
        - 6 stress component outputs from torch.matmul
        - 3 deformation outputs from torch.matmul (if kinematics enabled)
        
        Note: Von Mises, principal stresses, etc. are computed on CPU with Numba,
        so they don't consume GPU memory.
        """
        # Base: 6 stress component tensors from matmul (sx, sy, sz, sxy, syz, sxz)
        num_gpu_arrays = 6
        
        # Deformation tensors if kinematics are computed (ux, uy, uz)
        if calculate_kinematics:
            num_gpu_arrays += 3
        
        dtype_size = np.dtype(constants.NP_DTYPE).itemsize
        memory_per_node = num_gpu_arrays * num_time_points * dtype_size
        return memory_per_node

    def _estimate_memory_required_per_iteration(self, chunk_size, memory_per_node):
        """Estimate the total memory required per iteration to compute stresses."""
        total_memory = chunk_size * memory_per_node
        return total_memory / (1024 ** 3)  # Convert bytes to GB

    def _get_memory_per_node(self, num_time_points, calculate_von_mises, calculate_max_principal_stress,
                             calculate_damage, calculate_deformation=False,
                             calculate_velocity=False, calculate_acceleration=False):
        """
        Calculate memory required per node for the requested calculations.
        
        Uses the appropriate dtype size based on precision settings.
        """
        num_arrays = 6  # For actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz

        if calculate_von_mises:
            num_arrays += 1  # For sigma_vm
        if calculate_max_principal_stress:
            num_arrays += 3  # For s1, s2, s3
        if calculate_damage:
            num_arrays += 1  # For signed_von_mises

        # Base displacement components (ux, uy, uz) if any kinematics are needed
        if calculate_deformation or calculate_velocity or calculate_acceleration:
            num_arrays += 3

        # Deformation magnitude array
        if calculate_deformation:
            num_arrays += 1

        # Velocity and acceleration arrays (vel_x/y/z, acc_x/y/z, vel_mag, acc_mag)
        if calculate_velocity or calculate_acceleration:
            num_arrays += 8

        # Use appropriate dtype size
        dtype_size = np.dtype(constants.NP_DTYPE).itemsize
        memory_per_node = num_arrays * num_time_points * dtype_size
        return memory_per_node
    # endregion

    # region JIT Compiled Kernels (for heavy numerical operations)
    @staticmethod
    @njit(parallel=True)
    def compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """Compute von Mises stress."""
        sigma_vm = np.sqrt(
            0.5 * ((actual_sx - actual_sy) ** 2 + (actual_sy - actual_sz) ** 2 + (actual_sz - actual_sx) ** 2) +
            3 * (actual_sxy ** 2 + actual_syz ** 2 + actual_sxz ** 2)
        )
        return sigma_vm

    @staticmethod
    @njit(parallel=True)
    def _vel_acc_from_disp(ux, uy, uz, dt):
        """
        Compute velocity and acceleration using 4th-order central differences
        on a uniform grid dt. Endpoints use lower-order one-sided formulas.
        NumPy arrays are created with np.empty_like to avoid dtype issues.
        """
        n_nodes, n_times = ux.shape

        # Preallocate arrays without explicit dtype arguments
        vel_x = np.empty_like(ux)
        vel_y = np.empty_like(ux)
        vel_z = np.empty_like(ux)
        acc_x = np.empty_like(ux)
        acc_y = np.empty_like(ux)
        acc_z = np.empty_like(ux)

        # Uniform step size
        h = dt[1] - dt[0]

        for i in prange(n_nodes):
            # --- Velocity (first derivative) interior points ---
            for j in range(2, n_times - 2):
                # Coefficients as float literals for type inference
                vel_x[i, j] = (-ux[i, j + 2] + 8.0 * ux[i, j + 1]
                               - 8.0 * ux[i, j - 1] + ux[i, j - 2]) / (12.0 * h)
                vel_y[i, j] = (-uy[i, j + 2] + 8.0 * uy[i, j + 1]
                               - 8.0 * uy[i, j - 1] + uy[i, j - 2]) / (12.0 * h)
                vel_z[i, j] = (-uz[i, j + 2] + 8.0 * uz[i, j + 1]
                               - 8.0 * uz[i, j - 1] + uz[i, j - 2]) / (12.0 * h)

            # Fallback (second-order) at boundaries
            # j = 0, 1
            for j in (0, 1):
                vel_x[i, j] = (ux[i, j + 1] - ux[i, j]) / (dt[j + 1] - dt[j])
                vel_y[i, j] = (uy[i, j + 1] - uy[i, j]) / (dt[j + 1] - dt[j])
                vel_z[i, j] = (uz[i, j + 1] - uz[i, j]) / (dt[j + 1] - dt[j])
            # j = n_times-2, n_times-1
            for j in (n_times - 2, n_times - 1):
                vel_x[i, j] = (ux[i, j] - ux[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_y[i, j] = (uy[i, j] - uy[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_z[i, j] = (uz[i, j] - uz[i, j - 1]) / (dt[j] - dt[j - 1])

            # --- Acceleration (second derivative) interior points ---
            for j in range(2, n_times - 2):
                acc_x[i, j] = (-ux[i, j + 2] + 16.0 * ux[i, j + 1]
                               - 30.0 * ux[i, j] + 16.0 * ux[i, j - 1]
                               - ux[i, j - 2]) / (12.0 * h * h)
                acc_y[i, j] = (-uy[i, j + 2] + 16.0 * uy[i, j + 1]
                               - 30.0 * uy[i, j] + 16.0 * uy[i, j - 1]
                               - uy[i, j - 2]) / (12.0 * h * h)
                acc_z[i, j] = (-uz[i, j + 2] + 16.0 * uz[i, j + 1]
                               - 30.0 * uz[i, j] + 16.0 * uz[i, j - 1]
                               - uz[i, j - 2]) / (12.0 * h * h)

            # Fallback (second-order) at boundaries
            for j in (0, 1, n_times - 2, n_times - 1):
                if 0 < j < n_times - 1:
                    acc_x[i, j] = (ux[i, j + 1] - 2.0 * ux[i, j] + ux[i, j - 1]) / (h * h)
                    acc_y[i, j] = (uy[i, j + 1] - 2.0 * uy[i, j] + uy[i, j - 1]) / (h * h)
                    acc_z[i, j] = (uz[i, j + 1] - 2.0 * uz[i, j] + uz[i, j - 1]) / (h * h)
                else:
                    # One-sided second-order
                    k0 = 0 if j == 0 else n_times - 3
                    acc_x[i, j] = (ux[i, k0] - 2.0 * ux[i, k0 + 1] + ux[i, k0 + 2]) / (h * h)
                    acc_y[i, j] = (uy[i, k0] - 2.0 * uy[i, k0 + 1] + uy[i, k0 + 2]) / (h * h)
                    acc_z[i, j] = (uz[i, k0] - 2.0 * uz[i, k0 + 1] + uz[i, k0 + 2]) / (h * h)

        # Compute magnitudes with supported ufuncs
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        return vel_mag, acc_mag

    @staticmethod
    @njit(parallel=True)
    def _vel_acc_from_disp(ux, uy, uz, dt):
        """
        Compute velocity and acceleration using 6th-order central differences
        on a uniform grid dt. Endpoints use lower-order one-sided formulas.
        """
        n_nodes, n_times = ux.shape
        # Preallocate output arrays
        vel_x = np.empty_like(ux)
        vel_y = np.empty_like(ux)
        vel_z = np.empty_like(ux)
        acc_x = np.empty_like(ux)
        acc_y = np.empty_like(ux)
        acc_z = np.empty_like(ux)

        # Uniform step size
        h = dt[1] - dt[0]

        for i in prange(n_nodes):
            # --- Velocity (first derivative) interior points (6th-order) ---
            for j in range(3, n_times - 3):
                vel_x[i, j] = (-ux[i, j + 3]
                               + 9.0 * ux[i, j + 2]
                               - 45.0 * ux[i, j + 1]
                               + 45.0 * ux[i, j - 1]
                               - 9.0 * ux[i, j - 2]
                               + ux[i, j - 3]
                               ) / (60.0 * h)
                vel_y[i, j] = (-uy[i, j + 3]
                               + 9.0 * uy[i, j + 2]
                               - 45.0 * uy[i, j + 1]
                               + 45.0 * uy[i, j - 1]
                               - 9.0 * uy[i, j - 2]
                               + uy[i, j - 3]
                               ) / (60.0 * h)
                vel_z[i, j] = (-uz[i, j + 3]
                               + 9.0 * uz[i, j + 2]
                               - 45.0 * uz[i, j + 1]
                               + 45.0 * uz[i, j - 1]
                               - 9.0 * uz[i, j - 2]
                               + uz[i, j - 3]
                               ) / (60.0 * h)

            # Fallback (lower-order) at boundaries for velocity
            # j = 0,1,2
            for j in (0, 1, 2):
                vel_x[i, j] = (ux[i, j + 1] - ux[i, j]) / (dt[j + 1] - dt[j])
                vel_y[i, j] = (uy[i, j + 1] - uy[i, j]) / (dt[j + 1] - dt[j])
                vel_z[i, j] = (uz[i, j + 1] - uz[i, j]) / (dt[j + 1] - dt[j])
            # j = n_times-3, n_times-2, n_times-1
            for j in (n_times - 3, n_times - 2, n_times - 1):
                vel_x[i, j] = (ux[i, j] - ux[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_y[i, j] = (uy[i, j] - uy[i, j - 1]) / (dt[j] - dt[j - 1])
                vel_z[i, j] = (uz[i, j] - uz[i, j - 1]) / (dt[j] - dt[j - 1])

            # --- Acceleration (second derivative) interior points (6th-order) ---
            for j in range(3, n_times - 3):
                acc_x[i, j] = (2.0 * ux[i, j + 3]
                               - 27.0 * ux[i, j + 2]
                               + 270.0 * ux[i, j + 1]
                               - 490.0 * ux[i, j]
                               + 270.0 * ux[i, j - 1]
                               - 27.0 * ux[i, j - 2]
                               + 2.0 * ux[i, j - 3]
                               ) / (180.0 * h * h)
                acc_y[i, j] = (2.0 * uy[i, j + 3]
                               - 27.0 * uy[i, j + 2]
                               + 270.0 * uy[i, j + 1]
                               - 490.0 * uy[i, j]
                               + 270.0 * uy[i, j - 1]
                               - 27.0 * uy[i, j - 2]
                               + 2.0 * uy[i, j - 3]
                               ) / (180.0 * h * h)
                acc_z[i, j] = (2.0 * uz[i, j + 3]
                               - 27.0 * uz[i, j + 2]
                               + 270.0 * uz[i, j + 1]
                               - 490.0 * uz[i, j]
                               + 270.0 * uz[i, j - 1]
                               - 27.0 * uz[i, j - 2]
                               + 2.0 * uz[i, j - 3]
                               ) / (180.0 * h * h)

            # Fallback (lower-order) at boundaries for acceleration
            for j in (0, 1, 2, n_times - 3, n_times - 2, n_times - 1):
                if 0 < j < n_times - 1:
                    # central 2nd-order
                    acc_x[i, j] = (ux[i, j + 1] - 2.0 * ux[i, j] + ux[i, j - 1]) / (h * h)
                    acc_y[i, j] = (uy[i, j + 1] - 2.0 * uy[i, j] + uy[i, j - 1]) / (h * h)
                    acc_z[i, j] = (uz[i, j + 1] - 2.0 * uz[i, j] + uz[i, j - 1]) / (h * h)
                else:
                    # one-sided 2nd-order
                    k0 = 0 if j < 3 else n_times - 3
                    acc_x[i, j] = (ux[i, k0] - 2.0 * ux[i, k0 + 1] + ux[i, k0 + 2]) / (h * h)
                    acc_y[i, j] = (uy[i, k0] - 2.0 * uy[i, k0 + 1] + uy[i, k0 + 2]) / (h * h)
                    acc_z[i, j] = (uz[i, k0] - 2.0 * uz[i, k0 + 1] + uz[i, k0 + 2]) / (h * h)

        # Compute magnitudes
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
        acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        return vel_mag, acc_mag, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z

    @staticmethod
    @njit(parallel=True)
    def compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz):
        """
        Compute the signed von Mises stress by assigning a sign to the existing von Mises stress.
        Signed von Mises = sigma_vm * ((sx + sy + sz + 1e-6) / |sx + sy + sz + 1e-6|)
        """
        # Calculate the sum of normal stresses
        normal_stress_sum = actual_sx + actual_sy + actual_sz

        # Add a small value (1e-6) to avoid division by zero
        signed_von_mises = sigma_vm * (normal_stress_sum + 1e-6) / np.abs(normal_stress_sum + 1e-6)

        return signed_von_mises

    @staticmethod
    @njit(parallel=True)
    def compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz):
        """
        Calculates the three principal stresses from the six components of stress.

        -- How This Function Works --
        This function takes 2D arrays of the six standard stress components as input.
        For each point in these arrays, it uses an analytical method (Cardano's formula for
        solving cubic equations) to calculate the three principal stresses.

        Args:
            sx (np.ndarray): 2D array of normal stresses in the X-direction. Shape is (num_nodes, num_time_points).
            sy (np.ndarray): 2D array of normal stresses in the Y-direction.
            sz (np.ndarray): 2D array of normal stresses in the Z-direction.
            sxy (np.ndarray): 2D array of XY shear stresses.
            syz (np.ndarray): 2D array of YZ shear stresses.
            sxz (np.ndarray): 2D array of XZ shear stresses.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three 2D arrays (s1, s2, s3).
            - s1: The maximum (most positive) principal stress at each point.
            - s2: The intermediate principal stress at each point.
            - s3: The minimum (most negative) principal stress at each point.
        """
        # Read the dimensions (e.g., height and width) from the input stress array.
        # This tells us how many nodes and time points we need to process.
        num_nodes, num_time_points = actual_sx.shape

        # Create empty 2D arrays filled with zeros to hold our final results.
        # Pre-allocating memory like this is more efficient than building the arrays on the fly.
        s1_out = np.zeros((num_nodes, num_time_points), dtype=constants.NP_DTYPE)
        s2_out = np.zeros_like(s1_out)
        s3_out = np.zeros_like(s1_out)

        # --- Pre-calculate mathematical constants to avoid recalculating them inside the loop ---
        two_pi_3 = 2.0943951023931953  # This is 2 * pi / 3, used in the trigonometric solution.
        tiny_p = 1.0e-12  # A very small number used as a tolerance for a special case.

        # These nested loops ensure we perform the calculation for every node at every time step.
        # `prange` is Numba's "parallel range," which splits the work of this outer loop
        # across multiple CPU cores automatically.
        for i in prange(num_nodes):
            for j in range(num_time_points):
                # For the current point (node `i`, time `j`), get the six stress values.
                s_x = actual_sx[i, j]
                s_y = actual_sy[i, j]
                s_z = actual_sz[i, j]
                s_xy = actual_sxy[i, j]
                s_yz = actual_syz[i, j]
                s_xz = actual_sxz[i, j]

                # --- Step 1: Calculate Stress Invariants (I1, I2, I3) ---
                # To find the principal stresses, we first calculate three special values called
                # "invariants." They are called this because their values don't change even if you
                # rotate the object. They are fundamental properties of the stress state.
                I1 = s_x + s_y + s_z
                I2 = (s_x * s_y + s_y * s_z + s_z * s_x
                      - s_xy ** 2 - s_yz ** 2 - s_xz ** 2)
                I3 = (s_x * s_y * s_z
                      + 2 * s_xy * s_yz * s_xz
                      - s_x * s_yz ** 2
                      - s_y * s_xz ** 2
                      - s_z * s_xy ** 2)

                # --- Step 2: Formulate the Cubic Equation ---
                # The three principal stresses are the mathematical roots of a cubic polynomial equation
                # defined by the invariants. To solve it, we convert it into a simpler "depressed"
                # form: y³ + p*y + q = 0. The variables `p` and `q` are the coefficients for this equation.
                p = I2 - I1 ** 2 / 3.0
                q = (2.0 * I1 ** 3) / 27.0 - (I1 * I2) / 3.0 + I3

                # --- Step 3: Check for the special "Hydrostatic" case ---
                # This `if` statement checks for a simple case where an object is stressed equally
                # in all directions (like being deep underwater). In this case, `p` and `q` are
                # effectively zero, and all three principal stresses are equal.
                # This check handles that case directly and avoids division-by-zero errors in the
                # more complex calculations below, making the function more robust.
                if abs(p) < tiny_p and abs(q) < tiny_p:
                    s_hydro = I1 / 3.0
                    s1_out[i, j] = s_hydro
                    s2_out[i, j] = s_hydro
                    s3_out[i, j] = s_hydro
                    continue  # Skip to the next point in the loop

                # --- Step 4: Solve the Cubic Equation using the Trigonometric Method ---
                # For the general case, the roots are found using a reliable and stable trigonometric
                # formula (related to Viète's formulas).
                minus_p_over_3 = -p / 3.0
                # For real stresses, minus_p_over_3 must be non-negative, so sqrt is safe.
                sqrt_m = math.sqrt(minus_p_over_3)
                cos_arg = q / (2.0 * sqrt_m ** 3)

                # This is a numerical safety check. Due to tiny computer precision errors, `cos_arg`
                # might be slightly outside the valid [-1, 1] range for the `acos` function.
                # This code nudges it back into range to prevent a crash.
                if cos_arg > 1.0:
                    cos_arg = 1.0
                elif cos_arg < -1.0:
                    cos_arg = -1.0

                # These lines are the core of the trigonometric solution formula.
                phi = math.acos(cos_arg) / 3.0
                amp = 2.0 * sqrt_m

                # The final formulas give us the three roots, which are our principal stresses.
                s1 = I1 / 3.0 + amp * math.cos(phi)
                s2 = I1 / 3.0 + amp * math.cos(phi - two_pi_3)
                s3 = I1 / 3.0 + amp * math.cos(phi + two_pi_3)

                # --- Step 5: Sort the Results ---
                # The formulas don't guarantee which of s1, s2, or s3 is the largest.
                # This block of code performs a simple sort to ensure that s1 is always the
                # maximum value, s2 is the middle, and s3 is the minimum. This is the standard
                # engineering convention.
                if s1 < s2:
                    s1, s2 = s2, s1
                if s2 < s3:
                    s2, s3 = s3, s2
                if s1 < s2:
                    s1, s2 = s2, s1

                # --- Step 6: Store the Final Results ---
                # Assign the sorted principal stresses to their correct place in our output arrays.
                s1_out[i, j] = s1
                s2_out[i, j] = s2
                s3_out[i, j] = s3

        # After the loops have finished, return the three complete 2D arrays of results.
        return s1_out, s2_out, s3_out

    @staticmethod
    @njit
    def rainflow_counter(series):
        n = len(series)
        cycles = []
        stack = []
        for i in range(n):
            s = series[i]
            stack.append(s)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                if (s1 - s0) * (s1 - s2) >= 0:
                    stack.pop(-2)
                else:
                    break
            if len(stack) >= 4:
                s0, s1, s2, s3 = stack[-4], stack[-3], stack[-2], stack[-1]
                if abs(s1 - s2) <= abs(s0 - s1):
                    cycles.append((abs(s1 - s2), 0.5))
                    stack.pop(-3)
        # Count residuals
        for i in range(len(stack) - 1):
            cycles.append((abs(stack[i] - stack[i + 1]), 0.5))
        # Convert cycles to ranges and counts
        ranges = np.array([c[0] for c in cycles])
        counts = np.array([c[1] for c in cycles])
        return ranges, counts

    @staticmethod
    @njit(parallel=True)
    def compute_potential_damage_for_all_nodes(sigma_vm, coeff_A, coeff_m):
        num_nodes = sigma_vm.shape[0]
        damages = np.zeros(num_nodes, dtype=constants.NP_DTYPE)
        for i in prange(num_nodes):
            series = sigma_vm[i, :]
            ranges, counts = rainflow_counter(series)
            # Compute damage
            damage = np.sum(counts / (coeff_A / ((ranges + 1e-10) ** coeff_m)))
            damages[i] = damage
        return damages
    # endregion

    # region Core Computations (PyTorch/Numpy)
    @staticmethod
    def map_steady_state_stresses(steady_stress, steady_node_ids, modal_node_ids):
        """Map steady-state stress data to modal node IDs."""
        # Create a mapping from steady_node_ids to steady_stress
        steady_node_dict = dict(zip(steady_node_ids.flatten(), steady_stress.flatten()))
        # Create an array for mapped steady stress
        mapped_steady_stress = np.array([steady_node_dict.get(node_id, 0.0) for node_id in modal_node_ids],
                                        dtype=constants.NP_DTYPE)
        return mapped_steady_stress

    def compute_normal_stresses(self, start_idx, end_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(self.modal_sx[start_idx:end_idx, :], self.modal_coord)
        actual_sy = torch.matmul(self.modal_sy[start_idx:end_idx, :], self.modal_coord)
        actual_sz = torch.matmul(self.modal_sz[start_idx:end_idx, :], self.modal_coord)
        actual_sxy = torch.matmul(self.modal_sxy[start_idx:end_idx, :], self.modal_coord)
        actual_syz = torch.matmul(self.modal_syz[start_idx:end_idx, :], self.modal_coord)
        actual_sxz = torch.matmul(self.modal_sxz[start_idx:end_idx, :], self.modal_coord)

        # Add steady-state stresses if included
        if self.is_steady_state_included:
            actual_sx += self.steady_sx[start_idx:end_idx].unsqueeze(1)
            actual_sy += self.steady_sy[start_idx:end_idx].unsqueeze(1)
            actual_sz += self.steady_sz[start_idx:end_idx].unsqueeze(1)
            actual_sxy += self.steady_sxy[start_idx:end_idx].unsqueeze(1)
            actual_syz += self.steady_syz[start_idx:end_idx].unsqueeze(1)
            actual_sxz += self.steady_sxz[start_idx:end_idx].unsqueeze(1)

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), \
            actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    def compute_normal_stresses_for_a_single_node(self, selected_node_idx):
        """Compute actual stresses using matrix multiplication."""
        actual_sx = torch.matmul(self.modal_sx[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sy = torch.matmul(self.modal_sy[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sz = torch.matmul(self.modal_sz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sxy = torch.matmul(self.modal_sxy[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_syz = torch.matmul(self.modal_syz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)
        actual_sxz = torch.matmul(self.modal_sxz[selected_node_idx: selected_node_idx + 1, :], self.modal_coord)

        # Add steady-state stresses if included
        if self.is_steady_state_included:
            actual_sx += self.steady_sx[selected_node_idx].unsqueeze(0)
            actual_sy += self.steady_sy[selected_node_idx].unsqueeze(0)
            actual_sz += self.steady_sz[selected_node_idx].unsqueeze(0)
            actual_sxy += self.steady_sxy[selected_node_idx].unsqueeze(0)
            actual_syz += self.steady_syz[selected_node_idx].unsqueeze(0)
            actual_sxz += self.steady_sxz[selected_node_idx].unsqueeze(0)

        return actual_sx.cpu().numpy(), actual_sy.cpu().numpy(), actual_sz.cpu().numpy(), actual_sxy.cpu().numpy(), actual_syz.cpu().numpy(), actual_sxz.cpu().numpy()

    def compute_deformations(self, start_idx, end_idx):
        """
        Compute actual nodal displacements (deformations) for all nodes in [start_idx, end_idx].
        This method multiplies the modal deformation modes with the modal coordinate matrix.
        """
        if self.modal_deformations_ux is None:
            return None  # No deformations data available

        # Compute displacements
        actual_ux = torch.matmul(self.modal_deformations_ux[start_idx:end_idx, :], self.modal_coord)
        actual_uy = torch.matmul(self.modal_deformations_uy[start_idx:end_idx, :], self.modal_coord)
        actual_uz = torch.matmul(self.modal_deformations_uz[start_idx:end_idx, :], self.modal_coord)
        return (actual_ux.cpu().numpy(), actual_uy.cpu().numpy(), actual_uz.cpu().numpy())

    def set_plasticity_context(self, context: Optional[PlasticityRuntimeContext]):
        """Assign the runtime plasticity context (``None`` disables plasticity)."""
        self.plasticity_context = context
        if context and context.method in {"neuber", "glinka"}:
            self.max_over_time_svm_corrected = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
        else:
            self.max_over_time_svm_corrected = None
    # endregion

    # region Internal Batch Processing Helpers
    def _setup_calculation_jobs(self, calculate_von_mises, calculate_max_principal_stress,
                                calculate_min_principal_stress, calculate_deformation,
                                calculate_velocity, calculate_acceleration, calculate_damage):
        """
        Initializes a dictionary of calculation jobs, their memmap files, and result metadata.
        This centralizes the configuration for all possible calculations.
        """
        os.makedirs(self.output_directory, exist_ok=True)

        def _prepare_memmap(path, shape):
            path = os.path.abspath(path)
            # Ensure previous run artifacts are removed so Windows can recreate the file
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                # If removal fails we'll still try to overwrite in memmap
                pass
            return np.memmap(path, dtype=constants.RESULT_DTYPE, mode='w+', shape=shape)

        jobs = {}

        if calculate_von_mises:
            self.max_over_time_svm = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['von_mises'] = {
                'max_memmap': _prepare_memmap(os.path.join(self.output_directory, 'max_von_mises_stress.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_max_von_mises_stress.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "SVM_Max",
                'csv_header_time': "Time_of_SVM_Max",
                'csv_value_name': 'max_von_mises_stress.csv',
                'csv_time_name': 'time_of_max_von_mises_stress.csv'
            }

        if calculate_max_principal_stress:
            self.max_over_time_s1 = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['s1_max'] = {
                'max_memmap': _prepare_memmap(os.path.join(self.output_directory, 'max_s1_stress.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_max_s1_stress.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "S1_Max",
                'csv_header_time': "Time_of_S1_Max",
                'csv_value_name': 'max_s1_stress.csv',
                'csv_time_name': 'time_of_max_s1_stress.csv'
            }

        if calculate_min_principal_stress:
            self.min_over_time_s3 = np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['s3_min'] = {
                'min_memmap': _prepare_memmap(os.path.join(self.output_directory, 'min_s3_stress.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_min_s3_stress.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "S3_Min",
                'csv_header_time': "Time_of_S3_Min",
                'csv_value_name': 'min_s3_stress.csv',
                'csv_time_name': 'time_of_min_s3_stress.csv'
            }

        if calculate_deformation:
            self.max_over_time_def = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['deformation'] = {
                'max_memmap': _prepare_memmap(os.path.join(self.output_directory, 'max_deformation.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_max_deformation.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "DEF_Max",
                'csv_header_time': "Time_of_DEF_Max",
                'csv_value_name': 'max_deformation.csv',
                'csv_time_name': 'time_of_max_deformation.csv'
            }

        if calculate_velocity:
            self.max_over_time_vel = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['velocity'] = {
                'max_memmap': _prepare_memmap(os.path.join(self.output_directory, 'max_velocity.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_max_velocity.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "VEL_Max",
                'csv_header_time': "Time_of_VEL_Max",
                'csv_value_name': 'max_velocity.csv',
                'csv_time_name': 'time_of_max_velocity.csv'
            }

        if calculate_acceleration:
            self.max_over_time_acc = -np.inf * np.ones(self.modal_coord.shape[1], dtype=constants.NP_DTYPE)
            jobs['acceleration'] = {
                'max_memmap': _prepare_memmap(os.path.join(self.output_directory, 'max_acceleration.dat'),
                                              (self.modal_sx.shape[0],)),
                'time_memmap': _prepare_memmap(os.path.join(self.output_directory, 'time_of_max_acceleration.dat'),
                                               (self.modal_sx.shape[0],)),
                'csv_header_val': "ACC_Max",
                'csv_header_time': "Time_of_ACC_Max",
                'csv_value_name': 'max_acceleration.csv',
                'csv_time_name': 'time_of_max_acceleration.csv'
            }

        if calculate_damage:
            jobs['damage'] = {
                'damage_memmap': _prepare_memmap(os.path.join(self.output_directory, 'potential_damage_results.dat'),
                                                 (self.modal_sx.shape[0],)),
                'csv_header_val': "Potential Damage (Damage Index)",
                'csv_value_name': 'potential_damage_results.csv'
            }

        if self.plasticity_context and self.plasticity_context.method in {'neuber', 'glinka'}:
            jobs['plasticity'] = {
                'corrected_memmap': _prepare_memmap(
                    os.path.join(self.output_directory, 'corrected_von_mises.dat'),
                    (self.modal_sx.shape[0],)
                ),
                'time_memmap': _prepare_memmap(
                    os.path.join(self.output_directory, 'time_of_max_corrected_von_mises.dat'),
                    (self.modal_sx.shape[0],)
                ),
                'plastic_strain_memmap': _prepare_memmap(
                    os.path.join(self.output_directory, 'plastic_strain.dat'),
                    (self.modal_sx.shape[0],)
                ),
                'csv_header_val': "Corrected SVM (MPa)",
                'csv_header_time': "Time_of_Corrected_SVM_Max",
                'csv_header_strain': "Plastic_Strain",
                'csv_value_name': 'corrected_von_mises.csv',
                'csv_time_name': 'time_of_max_corrected_von_mises.csv',
                'csv_strain_name': 'plastic_strain.csv'
            }

        return jobs

    def _process_stress_chunk(self, jobs, time_values, start_idx, end_idx, actual_sx, actual_sy, actual_sz, actual_sxy,
                              actual_syz, actual_sxz):
        """Processes all stress-derived calculations for a given chunk of nodes."""
        # --- Von Mises Stress Calculation ---
        if 'von_mises' in jobs or 'damage' in jobs or 'plasticity' in jobs:
            start_time = time.time()
            sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                     actual_sxz)
            print(f"Elapsed time for von Mises stresses: {(time.time() - start_time):.3f} seconds")

            if 'von_mises' in jobs:
                job = jobs['von_mises']
                self.max_over_time_svm = np.maximum(self.max_over_time_svm, np.max(sigma_vm, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(sigma_vm, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(sigma_vm, axis=1)]

            if 'plasticity' in jobs:
                self._apply_plasticity_scalar_chunk(
                    jobs['plasticity'], sigma_vm, time_values, start_idx, end_idx
                )

        # --- Principal Stress Calculation ---
        if 's1_max' in jobs or 's3_min' in jobs:
            start_time = time.time()
            s1, _, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                        actual_sxz)
            print(f"Elapsed time for principal stresses: {(time.time() - start_time):.3f} seconds")

            if 's1_max' in jobs:
                job = jobs['s1_max']
                self.max_over_time_s1 = np.maximum(self.max_over_time_s1, np.max(s1, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(s1, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(s1, axis=1)]

            if 's3_min' in jobs:
                job = jobs['s3_min']
                self.min_over_time_s3 = np.minimum(self.min_over_time_s3, np.min(s3, axis=0))
                job['min_memmap'][start_idx:end_idx] = np.min(s3, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmin(s3, axis=1)]

        # --- Damage Calculation ---
        if 'damage' in jobs:
            start_time = time.time()
            job = jobs['damage']
            signed_von_mises = self.compute_signed_von_mises_stress(sigma_vm, actual_sx, actual_sy, actual_sz)
            # Use fatigue parameters if they have been set; otherwise, fall back to defaults.
            coeff_A = getattr(self, 'fatigue_A', 1)
            coeff_m = getattr(self, 'fatigue_m', -3)
            potential_damages = self.compute_potential_damage_for_all_nodes(signed_von_mises, coeff_A, coeff_m)
            job['damage_memmap'][start_idx:end_idx] = potential_damages
            print(f"Elapsed time for damage index calculation: {(time.time() - start_time):.3f} seconds")

    def _process_kinematics_chunk(self, jobs, time_values, start_idx, end_idx):
        """Processes deformation, velocity, and acceleration for a given chunk."""
        if self.modal_deformations_ux is None:
            return

        ux, uy, uz = self.compute_deformations(start_idx, end_idx)

        # --- Deformation ---
        if 'deformation' in jobs:
            start_time = time.time()
            job = jobs['deformation']
            def_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
            self.max_over_time_def = np.maximum(self.max_over_time_def, np.max(def_mag, axis=0))
            job['max_memmap'][start_idx:end_idx] = np.max(def_mag, axis=1)
            job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(def_mag, axis=1)]
            print(f"Elapsed time for deformation magnitude and time: {(time.time() - start_time):.3f} seconds")

        # --- Velocity & Acceleration ---
        if 'velocity' in jobs or 'acceleration' in jobs:
            start_time = time.time()
            vel_mag, acc_mag, _, _, _, _, _, _ = self._vel_acc_from_disp(ux, uy, uz, self.time_values)
            print(
                f"Elapsed time for calculation of velocity/acceleration components: {(time.time() - start_time):.3f} seconds")

            if 'velocity' in jobs:
                start_time = time.time()
                job = jobs['velocity']
                self.max_over_time_vel = np.maximum(self.max_over_time_vel, np.max(vel_mag, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(vel_mag, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(vel_mag, axis=1)]
                print(f"Elapsed time for velocity magnitude and time: {(time.time() - start_time):.3f} seconds")

            if 'acceleration' in jobs:
                start_time = time.time()
                job = jobs['acceleration']
                self.max_over_time_acc = np.maximum(self.max_over_time_acc, np.max(acc_mag, axis=0))
                job['max_memmap'][start_idx:end_idx] = np.max(acc_mag, axis=1)
                job['time_memmap'][start_idx:end_idx] = time_values[np.argmax(acc_mag, axis=1)]
                print(f"Elapsed time for acceleration magnitude and time: {(time.time() - start_time):.3f} seconds")
    # endregion

    # region Main Methods

    def _apply_plasticity_scalar_chunk(self, plasticity_job, sigma_vm, time_values, start_idx, end_idx):
        """Apply Neuber/Glinka corrections for the current chunk."""
        ctx = self.plasticity_context
        if ctx is None:
            return

        node_count = sigma_vm.shape[0]
        if ctx.temperatures is not None:
            local_temperatures = ctx.temperatures[start_idx:end_idx]
        else:
            local_temperatures = np.full(node_count, ctx.default_temperature, dtype=np.float64)

        peak_indices = np.argmax(sigma_vm, axis=1)
        peak_values = sigma_vm[np.arange(node_count), peak_indices]

        if ctx.method == 'neuber':
            corrected, plastic_strain = apply_neuber_correction(
                peak_values, local_temperatures, ctx.material_db,
                tol=ctx.tolerance, max_iterations=ctx.max_iterations
            )
        else:
            corrected, plastic_strain = apply_glinka_correction(
                peak_values, local_temperatures, ctx.material_db,
                tol=ctx.tolerance, max_iterations=ctx.max_iterations
            )

        plasticity_job['corrected_memmap'][start_idx:end_idx] = corrected
        plasticity_job['time_memmap'][start_idx:end_idx] = time_values[peak_indices]
        plasticity_job['plastic_strain_memmap'][start_idx:end_idx] = plastic_strain

        if self.max_over_time_svm_corrected is not None:
            np.maximum.at(self.max_over_time_svm_corrected, peak_indices, corrected)

    def _apply_ibg_single_node(self, node_index: int, stress_components) -> Optional[dict]:
        """Run IBG correction for a single node time history."""
        ctx = self.plasticity_context
        if ctx is None or ctx.method != 'ibg':
            return None

        sx, sy, sz, sxy, syz, sxz = [np.asarray(comp).reshape(-1) for comp in stress_components]
        stress_history = np.column_stack((sx, sy, sz, sxy, syz, sxz))
        if ctx.temperatures is not None:
            node_temp = float(ctx.temperatures[node_index])
        else:
            node_temp = float(ctx.default_temperature)
        temp_history = np.full(stress_history.shape[0], node_temp, dtype=np.float64)

        corrected_tensor, plastic_strain = apply_ibg_correction(
            stress_history, temp_history, ctx.material_db, use_plateau=ctx.use_plateau
        )
        corrected_vm = np.empty(stress_history.shape[0], dtype=np.float64)
        for step in range(stress_history.shape[0]):
            corrected_vm[step] = von_mises_from_voigt(corrected_tensor[step])

        delta_eps = np.empty_like(plastic_strain)
        if plastic_strain.size > 0:
            delta_eps[0] = plastic_strain[0]
            if plastic_strain.size > 1:
                delta_eps[1:] = plastic_strain[1:] - plastic_strain[:-1]

        return {
            'corrected_vm': corrected_vm,
            'plastic_strain': plastic_strain,
            'delta_plastic_strain': delta_eps,
            'corrected_tensor': corrected_tensor,
            'temperature': temp_history,
            'method': 'ibg',
        }

    def _apply_scalar_plasticity_single_node(self, node_index: int, sigma_vm_series: np.ndarray) -> Optional[dict]:
        """Run Neuber/Glinka correction per time step for a node's VM history."""
        ctx = self.plasticity_context
        if ctx is None or ctx.method not in {'neuber', 'glinka'}:
            return None

        if sigma_vm_series is None or sigma_vm_series.size == 0:
            return None

        if ctx.temperatures is not None:
            node_temp = float(ctx.temperatures[node_index])
        else:
            node_temp = float(ctx.default_temperature)
        temp_series = np.full(sigma_vm_series.shape[0], node_temp, dtype=np.float64)

        if ctx.method == 'neuber':
            corrected, plastic_strain = apply_neuber_correction(sigma_vm_series, temp_series, ctx.material_db,
                                                                tol=ctx.tolerance, max_iterations=ctx.max_iterations,
                                                                use_plateau=ctx.use_plateau)
        else:
            corrected, plastic_strain = apply_glinka_correction(sigma_vm_series, temp_series, ctx.material_db,
                                                                tol=ctx.tolerance, max_iterations=ctx.max_iterations,
                                                                use_plateau=ctx.use_plateau)

        delta_eps = np.empty_like(plastic_strain)
        if plastic_strain.size > 0:
            delta_eps[0] = plastic_strain[0]
            if plastic_strain.size > 1:
                delta_eps[1:] = plastic_strain[1:] - plastic_strain[:-1]

        return {
            'corrected_vm': corrected,
            'plastic_strain': plastic_strain,
            'delta_plastic_strain': delta_eps,
            'elastic_vm': sigma_vm_series,
            'temperature': temp_series,
            'method': ctx.method,
        }

    def process_results_in_batch(self,
                                 time_values,
                                 df_node_ids,
                                 node_coords,
                                 calculate_damage=False,
                                 calculate_von_mises=False,
                                 calculate_max_principal_stress=False,
                                 calculate_min_principal_stress=False,
                                 calculate_deformation=False,
                                 calculate_velocity=False,
                                 calculate_acceleration=False):
        """
        Processes stress and deformation results in batches to manage memory usage.
        This method coordinates the setup, execution, and finalization of calculations.
        """
        # --- 1. Initialization and Memory Estimation ---
        print("--- Starting Batch Processing ---")
        num_nodes, _ = self.modal_sx.shape
        num_time_points = self.modal_coord.shape[1]

        # Display memory information based on compute device
        if self._is_gpu_mode():
            gpu_info = self._get_gpu_memory_info()
            gpu_name = torch.cuda.get_device_name(self.device)
            self.total_memory = gpu_info['total'] / (1024 ** 3)
            self.available_memory = gpu_info['available_for_chunks'] / (1024 ** 3)
            self.allocated_memory = self.available_memory
            used_pct = (gpu_info['used'] / gpu_info['total']) * 100 if gpu_info['total'] > 0 else 0
            print(f"Compute Device: GPU ({gpu_name})")
            print(f"Total GPU VRAM: {self.total_memory:.2f} GB")
            print(f"Currently Used: {gpu_info['used'] / (1024 ** 3):.2f} GB ({used_pct:.1f}%)")
            print(f"Available for Processing: {self.available_memory:.2f} GB")
        else:
            my_virtual_memory = psutil.virtual_memory()
            self.total_memory = my_virtual_memory.total / (1024 ** 3)
            self.available_memory = my_virtual_memory.available / (1024 ** 3)
            self.allocated_memory = my_virtual_memory.available * constants.RAM_PERCENT / (1024 ** 3)
            print(f"Compute Device: CPU")
            print(f"Total system RAM: {self.total_memory:.2f} GB")
            print(f"Available system RAM: {self.available_memory:.2f} GB")
            print(f"Allocated for Processing: {self.allocated_memory:.2f} GB")

        chunk_size = self._estimate_chunk_size(
            num_time_points, calculate_von_mises, calculate_max_principal_stress, calculate_damage,
            calculate_deformation, calculate_velocity, calculate_acceleration)
        num_iterations = (num_nodes + chunk_size - 1) // chunk_size

        # Calculate memory per node based on device type
        is_kinematics = calculate_deformation or calculate_velocity or calculate_acceleration
        if self._is_gpu_mode():
            gpu_mem_per_node = self._get_gpu_memory_per_node(num_time_points, is_kinematics)
            gpu_mem_per_iter = self._estimate_memory_required_per_iteration(chunk_size, gpu_mem_per_node)
            print(f"Processing {num_nodes} nodes in {num_iterations} iterations (chunk size: {chunk_size}).")
            print(f"Estimated GPU VRAM per iteration: {gpu_mem_per_iter:.2f} GB")
        else:
            ram_per_node = self._get_memory_per_node(
                num_time_points, calculate_von_mises, calculate_max_principal_stress, calculate_damage,
                calculate_deformation, calculate_velocity, calculate_acceleration)
            ram_per_iter = self._estimate_memory_required_per_iteration(chunk_size, ram_per_node)
            print(f"Processing {num_nodes} nodes in {num_iterations} iterations (chunk size: {chunk_size}).")
            print(f"Estimated RAM per iteration: {ram_per_iter:.2f} GB")
        print()  # Blank line for readability

        # --- 2. Setup Calculation Jobs and Memmap Files ---
        calculation_jobs = self._setup_calculation_jobs(
            calculate_von_mises, calculate_max_principal_stress, calculate_min_principal_stress,
            calculate_deformation, calculate_velocity, calculate_acceleration, calculate_damage
        )

        is_stress_needed = any(k in calculation_jobs for k in ['von_mises', 's1_max', 's3_min', 'damage'])
        is_kinematics_needed = any(k in calculation_jobs for k in ['deformation', 'velocity', 'acceleration'])

        # --- 3. Main Processing Loop ---
        for i, start_idx in enumerate(range(0, num_nodes, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_nodes)
            print(f"\n--- Iteration {i + 1}/{num_iterations} (Nodes {start_idx}-{end_idx - 1}) ---")

            actual_stresses = None
            if is_stress_needed:
                start_time = time.time()
                actual_stresses = self.compute_normal_stresses(start_idx, end_idx)
                print(f"Elapsed time for normal stresses: {(time.time() - start_time):.3f} seconds")

            if is_stress_needed:
                self._process_stress_chunk(calculation_jobs, time_values, start_idx, end_idx, *actual_stresses)

            if is_kinematics_needed:
                self._process_kinematics_chunk(calculation_jobs, time_values, start_idx, end_idx)

            # --- Memory Management and Progress Update ---
            start_time = time.time()
            del actual_stresses
            gc.collect()
            if self._is_gpu_mode():
                torch.cuda.empty_cache()
            print(f"Elapsed time for garbage collection: {(time.time() - start_time):.3f} seconds")

            progress_percentage = ((i + 1) / num_iterations) * 100
            self.progress_signal.emit(int(progress_percentage))
            QApplication.processEvents()

            # Log memory status based on compute device
            memory_status = self._get_current_memory_usage_str()
            print(f"Iteration {i + 1} complete. {memory_status}. Progress: {progress_percentage:.1f}%")

        # --- 4. Finalization ---
        print("\n--- Finalizing Results ---")
        self._finalize_and_convert_results(calculation_jobs, df_node_ids, node_coords)
        print("--- Batch Processing Finished ---")

    def process_results_for_a_single_node(self,
                                          selected_node_idx,
                                          selected_node_id,
                                          _df_node_ids, # will be used in future for multiple node plots
                                          calculate_von_mises=False,
                                          calculate_max_principal_stress=False,
                                          calculate_min_principal_stress=False,
                                          calculate_deformation=False,
                                          calculate_velocity=False,
                                          calculate_acceleration=False):
        """
        Process results for a single node and return the stress data for plotting.

        Parameters:
        - selected_node_idx: The index of the node to process.
        - calculate_von_mises: Boolean flag to compute Von Mises stress.
        - calculate_max_principal_stress: Boolean flag to compute Max Principal Stress.
        - calculate_min_principal_stress: Boolean flag to compute Min Principal Stress.
        - calculate deformation,velocity,acceleration flag are also used for computing the related parameters

        Returns:
        - time_points: Array of time points for the selected node.
        - stress_values: Array of stress values (either Von Mises or Max/Min Principal Stress).
        """

        metadata = {}

        is_stress_calc_needed = calculate_von_mises or calculate_max_principal_stress or calculate_min_principal_stress
        if is_stress_calc_needed:
            # region Compute normal stresses for the selected node
            actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz = \
                self.compute_normal_stresses_for_a_single_node(selected_node_idx)
            # endregion

            if calculate_von_mises:
                # Compute Von Mises stress for the selected node
                sigma_vm = self.compute_von_mises_stress(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                         actual_sxz)
                print(f"Von Mises Stress calculated for Node {selected_node_id}\n")
                plasticity_info = None
                # If IBG is selected, compute IBG overlay; otherwise allow per-step Neuber/Glinka overlay
                if self.plasticity_context is not None:
                    if self.plasticity_context.method == 'ibg':
                        plasticity_info = self._apply_ibg_single_node(
                            selected_node_idx,
                            (actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz, actual_sxz)
                        )
                    elif self.plasticity_context.method in {'neuber', 'glinka'}:
                        sigma_vm_series = sigma_vm[0, :]
                        plasticity_info = self._apply_scalar_plasticity_single_node(selected_node_idx, sigma_vm_series)
                if plasticity_info is not None:
                    plasticity_info['elastic_vm'] = sigma_vm[0, :]
                    metadata['plasticity'] = plasticity_info

                return np.arange(sigma_vm.shape[1]), sigma_vm[0, :], metadata  # time_points, stress_values

            if calculate_max_principal_stress or calculate_min_principal_stress:
                s1, _, s3 = self.compute_principal_stresses(actual_sx, actual_sy, actual_sz, actual_sxy, actual_syz,
                                                             actual_sxz)
                if calculate_max_principal_stress:
                    # Compute Principal Stresses for the selected node
                    print(f"Max Principal Stresses calculated for Node {selected_node_id}\n")
                    return np.arange(s1.shape[1]), s1[0, :], metadata  # time_indices, stress_values

                if calculate_min_principal_stress:
                    print(f"Min Principal Stresses calculated for Node {selected_node_id}\n")
                    return np.arange(s3.shape[1]), s3[0, :], metadata  # S₃ min history

        if calculate_deformation or calculate_velocity or calculate_acceleration:
            if self.modal_deformations_ux is None:
                print("Deformation data missing – velocity/acceleration/deformation calculations are skipped.")
            else:
                ux, uy, uz = self.compute_deformations(selected_node_idx, selected_node_idx+1)

                if calculate_deformation:
                    def_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
                    deformation_data = {
                        'Magnitude': def_mag[0, :],
                        'X': ux[0, :],
                        'Y': uy[0, :],
                        'Z': uz[0, :]
                    }
                    print(f"Deformation calculated for Node {selected_node_id}\n")
                    return np.arange(def_mag.shape[1]), deformation_data, metadata

                if calculate_velocity or calculate_acceleration:
                    vel_mag, acc_mag, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z = \
                        self._vel_acc_from_disp(ux, uy, uz, self.time_values)
                    if calculate_velocity:
                        velocity_data = {
                            'Magnitude': vel_mag[0, :],
                            'X': vel_x[0, :],
                            'Y': vel_y[0, :],
                            'Z': vel_z[0, :]
                        }
                        print(f"Velocity calculated for Node {selected_node_id}\n")
                        return np.arange(vel_mag.shape[1]), velocity_data, metadata
                    if calculate_acceleration:
                        acceleration_data = {
                            'Magnitude': acc_mag[0, :],
                            'X': acc_x[0, :],
                            'Y': acc_y[0, :],
                            'Z': acc_z[0, :]
                        }
                        print(f"Acceleration calculated for Node {selected_node_id}\n")
                        return np.arange(acc_mag.shape[1]), acceleration_data, metadata

        # Return none if no output is requested
        return None, None, metadata
    # endregion

    # region File I/O Utilities
    def _write_csv(self, node_ids, node_coords, values, csv_filename, header):
        """Write a CSV file with NodeID and optional coordinate columns."""
        try:
            # Create a DataFrame for NodeID and the computed stress data
            df_out = pd.DataFrame({
                'NodeID': node_ids,
                header: values
            })
            # If node_coords is available, include the X, Y, Z coordinates
            if node_coords is not None:
                df_coords = pd.DataFrame(node_coords, columns=['X', 'Y', 'Z'])
                df_out = pd.concat([df_out, df_coords], axis=1)
            # Save to CSV
            df_out.to_csv(csv_filename, index=False)

            print(f"Successfully written {csv_filename}.")
        except Exception as e:
            print(f"Error writing {csv_filename}: {e}")

    def _finalize_and_convert_results(self, jobs, df_node_ids, node_coords):
        """Flushes all memmap files and converts them to CSV."""

        def _finalize_memmap(memmap_obj):
            if memmap_obj is None:
                return None, None
            try:
                memmap_obj.flush()
            except Exception:
                pass
            data_copy = np.asarray(memmap_obj, dtype=float).copy()
            path = getattr(memmap_obj, 'filename', None)
            return data_copy, path

        for job_name, job_data in jobs.items():
            if job_name == 'plasticity':
                corrected_values, corrected_path = _finalize_memmap(job_data.get('corrected_memmap'))
                time_values, time_path = _finalize_memmap(job_data.get('time_memmap'))
                strain_values, strain_path = _finalize_memmap(job_data.get('plastic_strain_memmap'))

                if corrected_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        corrected_values,
                        os.path.join(self.output_directory, job_data['csv_value_name']),
                        job_data['csv_header_val']
                    )
                if time_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        time_values,
                        os.path.join(self.output_directory, job_data['csv_time_name']),
                        job_data['csv_header_time']
                    )
                if strain_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        strain_values,
                        os.path.join(self.output_directory, job_data['csv_strain_name']),
                        job_data['csv_header_strain']
                    )

                for p in (corrected_path, time_path, strain_path):
                    if p:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

                job_data['corrected_memmap'] = None
                job_data['time_memmap'] = None
                job_data['plastic_strain_memmap'] = None
                continue

            if job_name == 'damage':
                damage_values, damage_path = _finalize_memmap(job_data.get('damage_memmap'))
                if damage_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        damage_values,
                        os.path.join(self.output_directory, job_data['csv_value_name']),
                        job_data['csv_header_val']
                    )
                if damage_path:
                    try:
                        os.remove(damage_path)
                    except OSError:
                        pass
                job_data['damage_memmap'] = None
                continue

            if job_name == 's3_min':
                min_values, min_path = _finalize_memmap(job_data.get('min_memmap'))
                time_values, time_path = _finalize_memmap(job_data.get('time_memmap'))
                if min_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        min_values,
                        os.path.join(self.output_directory, job_data['csv_value_name']),
                        job_data['csv_header_val']
                    )
                if time_values is not None:
                    self._write_csv(
                        df_node_ids, node_coords,
                        time_values,
                        os.path.join(self.output_directory, job_data['csv_time_name']),
                        job_data['csv_header_time']
                    )
                for p in (min_path, time_path):
                    if p:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                job_data['min_memmap'] = None
                job_data['time_memmap'] = None
                continue

            # Handles max value cases (s1, svm, def, vel, acc)
            max_values, max_path = _finalize_memmap(job_data.get('max_memmap'))
            time_values, time_path = _finalize_memmap(job_data.get('time_memmap'))
            if max_values is not None:
                self._write_csv(
                    df_node_ids, node_coords,
                    max_values,
                    os.path.join(self.output_directory, job_data['csv_value_name']),
                    job_data['csv_header_val']
                )
            if time_values is not None:
                self._write_csv(
                    df_node_ids, node_coords,
                    time_values,
                    os.path.join(self.output_directory, job_data['csv_time_name']),
                    job_data['csv_header_time']
                )
            for p in (max_path, time_path):
                if p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            job_data['max_memmap'] = None
            job_data['time_memmap'] = None
    # endregion
