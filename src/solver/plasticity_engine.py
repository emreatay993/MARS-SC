"""
Plasticity correction solvers for Neuber, Glinka, and IBG methods.

This module houses the numeric kernels that will be orchestrated by the MARS-SC
analysis pipeline. CLI utilities from the standalone prototype have been
removed; call the helper functions exposed here instead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # pragma: no cover - optional acceleration
    from numba import njit, prange
except ImportError:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):  # type: ignore
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    def prange(*args):  # type: ignore
        return range(*args)

LOG = logging.getLogger(__name__)

__all__ = [
    "MaterialDB",
    "POISSON_RATIO",
    "apply_neuber_correction",
    "apply_glinka_correction",
    "apply_ibg_correction",
    "von_mises_from_voigt",
    "default_material_db",
]

# --- Constants --------------------------------------------------------------
POISSON_RATIO: float = 0.30  # isotropic
VOIGT_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
EPS = 1e-12

# ============================================================================
# Data model
# ============================================================================

@dataclass(frozen=True)
class MaterialDB:
    """
    Temperature-blended multilinear hardening database.

    Shapes
      TEMP  : (NT,)
      E_tab : (NT,)
      SIG   : (NT, NP)
      EPSP  : (NT, NP)   # plastic strain values per curve point

    Requirements
      - TEMP strictly increasing.
      - For each T-row, SIG increasing and same NP as EPSP.
      - First column of (SIG, EPSP) defines approximate yield: SIG[:, 0].
    """
    TEMP:  np.ndarray        # (NT,)
    E_tab: np.ndarray        # (NT,)
    SIG:   np.ndarray        # (NT, NP)
    EPSP:  np.ndarray        # (NT, NP)

    @staticmethod
    def from_arrays(temp, e_tab, sig, epsp) -> "MaterialDB":
        TEMP = np.asarray(temp,  dtype=np.float64)
        E    = np.asarray(e_tab, dtype=np.float64)
        SIG  = np.asarray(sig,   dtype=np.float64)
        EPSP = np.asarray(epsp,  dtype=np.float64)

        if TEMP.ndim != 1 or E.ndim != 1:
            raise ValueError("TEMP and E_tab must be 1D.")
        if TEMP.size != E.size:
            raise ValueError("TEMP and E_tab length mismatch.")
        if SIG.shape != EPSP.shape:
            raise ValueError("SIG and EPSP shape mismatch.")
        if SIG.shape[0] != TEMP.size:
            raise ValueError("First dimension of SIG/EPSP must match TEMP size.")
        if SIG.shape[1] < 2:
            raise ValueError("Need at least two points per hardening curve.")
        if not np.all(np.diff(TEMP) > 0):
            raise ValueError("TEMP must be strictly increasing.")
        if not np.all(np.diff(SIG, axis=1) > 0):
            raise ValueError("Each SIG row must be strictly increasing.")
        return MaterialDB(TEMP=TEMP, E_tab=E, SIG=SIG, EPSP=EPSP)

    # ----------- utilities: temperature blend --------------------------------

    @njit(cache=True)
    def _bound_T_indices(self, T: float) -> Tuple[int, int, float]:
        """Return (i_low, i_hi, weight) with linear weight in [0,1]."""
        TEMP = self.TEMP
        nT = TEMP.size
        i = int(np.searchsorted(TEMP, T) - 1)
        if i < 0:
            return 0, 0, 0.0
        if i >= nT - 1:
            return nT - 1, nT - 1, 0.0
        w = (T - TEMP[i]) / (TEMP[i + 1] - TEMP[i])
        return i, i + 1, w

    @njit(cache=True)
    def E_of_T(self, T: float) -> float:
        i, j, w = self._bound_T_indices(T)
        return (1.0 - w) * self.E_tab[i] + w * self.E_tab[j]

    @njit(cache=True)
    def yield_of_T(self, T: float) -> float:
        """Approximate yield as first SIG point, blended across T."""
        i, j, w = self._bound_T_indices(T)
        return (1.0 - w) * self.SIG[i, 0] + w * self.SIG[j, 0]

    # ----------- curve ops (per temperature row) -----------------------------

    @staticmethod
    @njit(cache=True)
    def _interp_epsp_on_curve(sig: float, sig_row: np.ndarray, epsp_row: np.ndarray) -> float:
        """εp(σ) on one row. Piecewise-linear. Extrapolates linearly."""
        if sig <= sig_row[0]:
            return 0.0
        n = sig_row.size
        for k in range(n - 1):
            s0, s1 = sig_row[k], sig_row[k + 1]
            if s0 <= sig <= s1:
                e0, e1 = epsp_row[k], epsp_row[k + 1]
                return e0 + (sig - s0) * (e1 - e0) / (s1 - s0 + EPS)
        # extrapolate
        s0, s1 = sig_row[-2], sig_row[-1]
        e0, e1 = epsp_row[-2], epsp_row[-1]
        slope = (e1 - e0) / (s1 - s0 + EPS)
        return e1 + (sig - s1) * slope

    @staticmethod
    @njit(cache=True)
    def _invert_sigma_on_curve(epsp: float, sig_row: np.ndarray, epsp_row: np.ndarray) -> float:
        """σ(εp) on one row. Piecewise-linear inverse. Extrapolates linearly."""
        if epsp <= epsp_row[0]:
            return sig_row[0]
        n = epsp_row.size
        for k in range(n - 1):
            e0, e1 = epsp_row[k], epsp_row[k + 1]
            if e0 <= epsp <= e1:
                s0, s1 = sig_row[k], sig_row[k + 1]
                t = (epsp - e0) / (e1 - e0 + EPS)
                return s0 + t * (s1 - s0)
        # extrapolate
        e0, e1 = epsp_row[-2], epsp_row[-1]
        s0, s1 = sig_row[-2], sig_row[-1]
        slope = (s1 - s0) / (e1 - e0 + EPS)
        return s1 + slope * (epsp - e1)

    @staticmethod
    @njit(cache=True)
    def _plastic_energy_on_curve(sig: float, sig_row: np.ndarray, epsp_row: np.ndarray) -> float:
        """
        Up(σ) = ∫_0^{εp(σ)} σ_flow(εp) dεp on one row via trapezoids.
        """
        if sig <= sig_row[0]:
            return 0.0
        area = 0.0
        n = sig_row.size
        for k in range(n - 1):
            s0, s1 = sig_row[k], sig_row[k + 1]
            e0, e1 = epsp_row[k], epsp_row[k + 1]
            if sig <= s1:
                et = e0 + (sig - s0) * (e1 - e0) / (s1 - s0 + EPS)
                area += 0.5 * (s0 + sig) * (et - e0)
                return area
            area += 0.5 * (s0 + s1) * (e1 - e0)
        # beyond last point
        s0, s1 = sig_row[-2], sig_row[-1]
        e0, e1 = epsp_row[-2], epsp_row[-1]
        slope = (e1 - e0) / (s1 - s0 + EPS)
        et = e1 + slope * (sig - s1)
        area += 0.5 * (s1 + sig) * (et - e1)
        return area

    # ----------- temperature-blended public ops ------------------------------

    @njit(cache=True)
    def epsp_of_T_sigma(self, T: float, sig: float) -> float:
        """εp(T, σ) by blending results of two bracketing rows."""
        i, j, w = self._bound_T_indices(T)
        eL = self._interp_epsp_on_curve(sig, self.SIG[i], self.EPSP[i])
        if i == j:
            return eL
        eU = self._interp_epsp_on_curve(sig, self.SIG[j], self.EPSP[j])
        return (1.0 - w) * eL + w * eU

    @njit(cache=True)
    def sigma_of_T_epsp(self, T: float, epsp: float) -> float:
        """σ_flow(T, εp) by blending inverted values of two rows."""
        i, j, w = self._bound_T_indices(T)
        sL = self._invert_sigma_on_curve(epsp, self.SIG[i], self.EPSP[i])
        if i == j:
            return sL
        sU = self._invert_sigma_on_curve(epsp, self.SIG[j], self.EPSP[j])
        return (1.0 - w) * sL + w * sU

    @njit(cache=True)
    def Up_of_T_sigma(self, T: float, sig: float) -> float:
        """Plastic strain energy density Up(T, σ) by blending."""
        i, j, w = self._bound_T_indices(T)
        aL = self._plastic_energy_on_curve(sig, self.SIG[i], self.EPSP[i])
        if i == j:
            return aL
        aU = self._plastic_energy_on_curve(sig, self.SIG[j], self.EPSP[j])
        return (1.0 - w) * aL + w * aU


# ============================================================================
# Tensor utilities
# ============================================================================

@njit(cache=True)
def shear_modulus(E: float, nu: float = POISSON_RATIO) -> float:
    return E / (2.0 * (1.0 + nu))

@njit(cache=True)
def von_mises_from_voigt(sig6: np.ndarray) -> float:
    sx, sy, sz, txy, tyz, tzx = sig6[0], sig6[1], sig6[2], sig6[3], sig6[4], sig6[5]
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) +
        3.0 * (txy * txy + tyz * tyz + tzx * tzx)
    )

@njit(cache=True)
def deviator(sig6: np.ndarray) -> np.ndarray:
    m = (sig6[0] + sig6[1] + sig6[2]) / 3.0
    out = np.empty(6, dtype=np.float64)
    out[0] = sig6[0] - m
    out[1] = sig6[1] - m
    out[2] = sig6[2] - m
    out[3] = sig6[3]
    out[4] = sig6[4]
    out[5] = sig6[5]
    return out

@njit(cache=True)
def delta_Ue_uniaxial(sig_prev: float, sig_now: float, E: float) -> float:
    """ΔUe for equivalent/uniaxial increment: ½(σk-1+σk)(σk-σk-1)/E."""
    return 0.5 * (sig_prev + sig_now) * (sig_now - sig_prev) / (E + EPS)

@njit(cache=True)
def delta_Ue_tensor(sig_prev: np.ndarray, sig_now: np.ndarray, E: float, nu: float = POISSON_RATIO) -> float:
    """ΔUe = ½(σd_prev+σd_now):(εd_now-εd_prev), with εd = σd/(2G)."""
    dp = deviator(sig_prev)
    dn = deviator(sig_now)
    G  = shear_modulus(E, nu)
    num = 0.0
    for j in range(6):
        num += VOIGT_WEIGHTS[j] * (dp[j] + dn[j]) * (dn[j] - dp[j])
    return num / (4.0 * G + EPS)  # ½ and (1/2G) combined


@njit(cache=True)
def _bound_T_indices_njit(T: float, TEMP: np.ndarray) -> Tuple[int, int, float]:
    nT = TEMP.size
    i = int(np.searchsorted(TEMP, T) - 1)
    if i < 0:
        return 0, 0, 0.0
    if i >= nT - 1:
        return nT - 1, nT - 1, 0.0
    w = (T - TEMP[i]) / (TEMP[i + 1] - TEMP[i])
    return i, i + 1, w

@njit(cache=True)
def E_of_T_njit(T: float, TEMP: np.ndarray, E_tab: np.ndarray) -> float:
    i, j, w = _bound_T_indices_njit(T, TEMP)
    return (1.0 - w) * E_tab[i] + w * E_tab[j]

@njit(cache=True)
def yield_of_T_njit(T: float, TEMP: np.ndarray, SIG: np.ndarray) -> float:
    i, j, w = _bound_T_indices_njit(T, TEMP)
    return (1.0 - w) * SIG[i, 0] + w * SIG[j, 0]

@njit(cache=True)
def _interp_epsp_on_curve_njit(sig: float, sig_row: np.ndarray, epsp_row: np.ndarray, use_plateau: int) -> float:
    if sig <= sig_row[0]:
        return 0.0
    n = sig_row.size
    for k in range(n - 1):
        s0, s1 = sig_row[k], sig_row[k + 1]
        if s0 <= sig <= s1:
            e0, e1 = epsp_row[k], epsp_row[k + 1]
            return e0 + (sig - s0) * (e1 - e0) / (s1 - s0 + EPS)
    if use_plateau:
        return epsp_row[-1]
    else:
        s0, s1 = sig_row[-2], sig_row[-1]
        e0, e1 = epsp_row[-2], epsp_row[-1]
        slope = (e1 - e0) / (s1 - s0 + EPS)
        return e1 + (sig - s1) * slope

@njit(cache=True)
def _invert_sigma_on_curve_njit(epsp: float, sig_row: np.ndarray, epsp_row: np.ndarray, use_plateau: int) -> float:
    if epsp <= epsp_row[0]:
        return sig_row[0]
    n = epsp_row.size
    for k in range(n - 1):
        e0, e1 = epsp_row[k], epsp_row[k + 1]
        if e0 <= epsp <= e1:
            s0, s1 = sig_row[k], sig_row[k + 1]
            t = (epsp - e0) / (e1 - e0 + EPS)
            return s0 + t * (s1 - s0)
    if use_plateau:
        return sig_row[-1]
    else:
        e0, e1 = epsp_row[-2], epsp_row[-1]
        s0, s1 = sig_row[-2], sig_row[-1]
        slope = (s1 - s0) / (e1 - e0 + EPS)
        return s1 + slope * (epsp - e1)

@njit(cache=True)
def _plastic_energy_on_curve_njit(sig: float, sig_row: np.ndarray, epsp_row: np.ndarray, use_plateau: int) -> float:
    if sig <= sig_row[0]:
        return 0.0
    area = 0.0
    n = sig_row.size
    for k in range(n - 1):
        s0, s1 = sig_row[k], sig_row[k + 1]
        e0, e1 = epsp_row[k], epsp_row[k + 1]
        if sig <= s1:
            et = e0 + (sig - s0) * (e1 - e0) / (s1 - s0 + EPS)
            area += 0.5 * (s0 + sig) * (et - e0)
            return area
        area += 0.5 * (s0 + s1) * (e1 - e0)
    if use_plateau:
        # Clamp to last segment area only
        return area
    else:
        s0, s1 = sig_row[-2], sig_row[-1]
        e0, e1 = epsp_row[-2], epsp_row[-1]
        slope = (e1 - e0) / (s1 - s0 + EPS)
        et = e1 + slope * (sig - s1)
        area += 0.5 * (s1 + sig) * (et - e1)
        return area

@njit(cache=True)
def epsp_of_T_sigma_njit(T: float, sig: float, TEMP: np.ndarray, SIG: np.ndarray, EPSP: np.ndarray, use_plateau: int) -> float:
    i, j, w = _bound_T_indices_njit(T, TEMP)
    eL = _interp_epsp_on_curve_njit(sig, SIG[i], EPSP[i], use_plateau)
    if i == j:
        return eL
    eU = _interp_epsp_on_curve_njit(sig, SIG[j], EPSP[j], use_plateau)
    return (1.0 - w) * eL + w * eU

@njit(cache=True)
def sigma_of_T_epsp_njit(T: float, epsp: float, TEMP: np.ndarray, SIG: np.ndarray, EPSP: np.ndarray, use_plateau: int) -> float:
    i, j, w = _bound_T_indices_njit(T, TEMP)
    sL = _invert_sigma_on_curve_njit(epsp, SIG[i], EPSP[i], use_plateau)
    if i == j:
        return sL
    sU = _invert_sigma_on_curve_njit(epsp, SIG[j], EPSP[j], use_plateau)
    return (1.0 - w) * sL + w * sU

@njit(cache=True)
def Up_of_T_sigma_njit(T: float, sig: float, TEMP: np.ndarray, SIG: np.ndarray, EPSP: np.ndarray, use_plateau: int) -> float:
    i, j, w = _bound_T_indices_njit(T, TEMP)
    aL = _plastic_energy_on_curve_njit(sig, SIG[i], EPSP[i], use_plateau)
    if i == j:
        return aL
    aU = _plastic_energy_on_curve_njit(sig, SIG[j], EPSP[j], use_plateau)
    return (1.0 - w) * aL + w * aU

@njit(cache=True, parallel=True)
def solve_neuber_vector_core(sig_e: np.ndarray, T: np.ndarray,
                             TEMP: np.ndarray, E_tab: np.ndarray,
                             SIG: np.ndarray, EPSP: np.ndarray,
                             tol: float = 1e-10, itmax: int = 60, use_plateau: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    n = sig_e.size
    sc = np.empty(n, dtype=np.float64)
    ep = np.empty(n, dtype=np.float64)
    for i in prange(n):
        sigma_e_i = float(sig_e[i])
        Ti = float(T[i])
        if sigma_e_i <= 0.0:
            sc[i] = 0.0
            ep[i] = 0.0
            continue
        sigma = min(sigma_e_i, yield_of_T_njit(Ti, TEMP, SIG))
        if sigma <= 0.0:
            sigma = 1e-6
        for _ in range(itmax):
            E = E_of_T_njit(Ti, TEMP, E_tab)
            eps = epsp_of_T_sigma_njit(Ti, sigma, TEMP, SIG, EPSP, use_plateau)
            r = sigma / E + eps - (sigma_e_i * sigma_e_i) / (sigma * E + EPS)
            dS = 1e-6 * max(abs(sigma), 1.0)
            eps2 = epsp_of_T_sigma_njit(Ti, sigma + dS, TEMP, SIG, EPSP, use_plateau)
            r2 = (sigma + dS) / E + eps2 - (sigma_e_i * sigma_e_i) / ((sigma + dS) * E + EPS)
            der = (r2 - r) / dS
            step = r / (der + EPS)
            sigma_next = sigma - step
            if sigma_next <= 0.0:
                sigma_next = 0.5 * sigma
            if abs(step) / (abs(sigma) + EPS) < tol:
                sigma = sigma_next
                break
            sigma = sigma_next
        sc[i] = sigma
        ep[i] = epsp_of_T_sigma_njit(Ti, sigma, TEMP, SIG, EPSP, use_plateau)
    return sc, ep

@njit(cache=True, parallel=True)
def solve_glinka_vector_core(sig_e: np.ndarray, T: np.ndarray,
                             TEMP: np.ndarray, E_tab: np.ndarray,
                             SIG: np.ndarray, EPSP: np.ndarray,
                             tol: float = 1e-10, itmax: int = 60, use_plateau: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    n = sig_e.size
    sc = np.empty(n, dtype=np.float64)
    ep = np.empty(n, dtype=np.float64)
    for i in prange(n):
        sigma_e_i = float(sig_e[i])
        Ti = float(T[i])
        if sigma_e_i <= 0.0:
            sc[i] = 0.0
            ep[i] = 0.0
            continue
        E = E_of_T_njit(Ti, TEMP, E_tab)
        Ue0 = sigma_e_i * sigma_e_i / (2.0 * E + EPS)
        sigma = min(sigma_e_i, yield_of_T_njit(Ti, TEMP, SIG))
        if sigma <= 0.0:
            sigma = 1e-6
        for _ in range(itmax):
            Upl = Up_of_T_sigma_njit(Ti, sigma, TEMP, SIG, EPSP, use_plateau)
            Uc = sigma * sigma / (2.0 * E + EPS) + Upl
            r = Uc - Ue0
            if abs(r) / (abs(Ue0) + EPS) < tol:
                break
            dS = 1e-6 * max(abs(sigma), 1.0)
            Upl2 = Up_of_T_sigma_njit(Ti, sigma + dS, TEMP, SIG, EPSP, use_plateau)
            Uc2 = (sigma + dS) * (sigma + dS) / (2.0 * E + EPS) + Upl2
            der = (Uc2 - Uc) / dS
            sigma_new = sigma - r / (der + EPS)
            if sigma_new <= 0.0:
                sigma_new = 0.5 * sigma
            if abs(sigma_new - sigma) / (abs(sigma) + EPS) < tol:
                sigma = sigma_new
                break
            sigma = sigma_new
        sc[i] = sigma
        ep[i] = epsp_of_T_sigma_njit(Ti, sigma, TEMP, SIG, EPSP, use_plateau)
    return sc, ep


# ============================================================================
# IBG tensor-history solver
# ============================================================================

@njit(cache=True)
def _delta_epsp_curve_aware_core(dUe: float, T: float, epsp_prev: float,
                                 TEMP: np.ndarray, SIG: np.ndarray, EPSP: np.ndarray,
                                 use_plateau: int = 0) -> float:
    """
    Solve ΔUe = ∫_{εp}^{εp+Δεp} σ_flow(T, εp) dεp using midpoint closure.
    Few fixed iterations. Yield-gated by construction via σ_flow.
    """
    if dUe <= 0.0:
        return 0.0
    sig_y = yield_of_T_njit(T, TEMP, SIG)
    dep = dUe / max(sig_y, 1e-9)
    for _ in range(3):
        s_mid = sigma_of_T_epsp_njit(T, epsp_prev + 0.5 * dep, TEMP, SIG, EPSP, use_plateau)
        if s_mid <= 0.0:
            break
        dep = dUe / (s_mid + EPS)
    return max(dep, 0.0)

@njit(cache=True)
def ibg_solver_tensor_core(sig_hist: np.ndarray,   # (N,6)
                           T_hist:   np.ndarray,
                           TEMP: np.ndarray, E_tab: np.ndarray,
                           SIG: np.ndarray, EPSP: np.ndarray,
                           use_plateau: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Incremental Buczynski–Glinka for a tensor stress history at one point.

    Returns
      sig_corr : (N,6) corrected tensor per step (radial scaling)
      epsp     : (N,)  cumulative equivalent plastic strain
    """
    N = sig_hist.shape[0]
    sig_corr = np.empty_like(sig_hist)
    epsp     = np.empty(N, dtype=np.float64)

    # Precompute elastic von-Mises for denominator and checks
    vm_el = np.empty(N, dtype=np.float64)
    for k in range(N):
        vm_el[k] = von_mises_from_voigt(sig_hist[k])

    sig_corr[0] = sig_hist[0]
    epsp[0]     = 0.0
    eps_prev    = 0.0

    for k in range(1, N):
        Tk = float(T_hist[k])
        Ek = E_of_T_njit(Tk, TEMP, E_tab)
        dUe = delta_Ue_tensor(sig_hist[k - 1], sig_hist[k], Ek)

        # purely elastic if below yield and no prior plasticity
        if max(vm_el[k - 1], vm_el[k]) <= yield_of_T_njit(Tk, TEMP, SIG) and eps_prev <= 0.0:
            sig_corr[k] = sig_hist[k]
            epsp[k]     = eps_prev
            continue

        # curve-aware Δεp
        k_factor = 0.8  # (e.g., 80% - YOU MUST TUNE THIS VALUE)
        # --- K-Factor Correction ---
        # This is an empirical factor (0.0 < k_factor <= 1.0) tuned to match
        # transient FEA. It acknowledges that the elastic dUe is an
        # overestimate of the energy that should be converted to plastic work.

        dUe_corrected = k_factor * max(dUe, 0.0)

        dep = _delta_epsp_curve_aware_core(max(dUe, 0.0), Tk, eps_prev, TEMP, SIG, EPSP, use_plateau)

        # Update total plastic strain *before* scaling
        eps_prev += dep
        epsp[k]   = eps_prev
        # --------------------------------------------------------

        # 2-pass scale iteration to couple denominator with corrected stress
        scale = 1.0
        for _ in range(2):
            vm_corr = scale * vm_el[k]
            # Use current flow stress at *updated* accumulated plastic strain
            flow    = sigma_of_T_epsp_njit(Tk, eps_prev, TEMP, SIG, EPSP, use_plateau)
            denom   = max(vm_corr, flow, 1e-9)
            # Use *total* plastic strain (eps_prev) in scaling formula
            scale   = 1.0 / (1.0 + Ek * eps_prev / denom)
            # ------------------------------------------------------------------

        # with deviatoric-only update:
        m = (sig_hist[k, 0] + sig_hist[k, 1] + sig_hist[k, 2]) / 3.0           # (1) mean stress
        hyd = np.array([m, m, m, 0.0, 0.0, 0.0], dtype=np.float64)      # (2) hydrostatic part
        dev = sig_hist[k] - hyd                                                # (3) deviatoric part
        sig_corr[k] = hyd + scale * dev                                        # (4) scale deviatoric only

        # INFO: Deviatoric-only radial scaling for J2 plasticity
        # Context:
        #   - sig_hist[k] is the trial Cauchy stress at step k in Voigt order:
        #     [σxx, σyy, σzz, τxy, τyz, τzx].
        #   - 'scale' is the scalar contraction factor computed from the IBG update.
        # Why:
        #   - J2/von Mises plasticity is insensitive to hydrostatic (mean) stress.
        #   - Plastic flow changes shape (deviatoric part) but not volume (mean stress).
        #   - Therefore, scale ONLY the deviatoric part; keep hydrostatic part unchanged.
        #
        # Steps:
        #   1) Compute mean (hydrostatic) stress m = (σxx + σyy + σzz) / 3.
        #   2) Build the hydrostatic stress vector hyd = [m, m, m, 0, 0, 0].
        #   3) Deviatoric part dev = sig_hist[k] - hyd.
        #   4) Apply radial scaling to dev, then re-add hyd.
        # Result:
        #   - Preserves plastic incompressibility and avoids artificial pressure changes.

    return sig_corr, epsp


# ============================================================================
# Convenience wrappers (vectorised helpers)
# ============================================================================

def apply_neuber_correction(
    sigma_equivalent: np.ndarray,
    temperature: np.ndarray,
    material: MaterialDB,
    tol: float = 1e-10,
    max_iterations: int = 60,
    use_plateau: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Neuber correction to an array of equivalent stresses.

    Args:
        sigma_equivalent: Elastic equivalent stress amplitudes per node.
        temperature: Temperature per node (same shape as ``sigma_equivalent``).
        material: Prepared ``MaterialDB`` instance.
        tol: Relative tolerance for fixed-point iteration.
        max_iterations: Maximum iterations for the solver.

    Returns:
        Tuple of ``(corrected_sigma, plastic_strain)`` arrays.
    """
    if sigma_equivalent.shape != temperature.shape:
        raise ValueError("sigma_equivalent and temperature arrays must match in shape.")

    LOG.debug("Running Neuber correction on %d entries", sigma_equivalent.size)
    sig = np.asarray(sigma_equivalent, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    return solve_neuber_vector_core(sig, temp, material.TEMP, material.E_tab, material.SIG, material.EPSP,
                                    tol=tol, itmax=max_iterations, use_plateau=1 if use_plateau else 0)


def apply_glinka_correction(
    sigma_equivalent: np.ndarray,
    temperature: np.ndarray,
    material: MaterialDB,
    tol: float = 1e-10,
    max_iterations: int = 60,
    use_plateau: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Glinka energy-density correction to an array of equivalent stresses.

    Parameters mirror ``apply_neuber_correction``.
    """
    if sigma_equivalent.shape != temperature.shape:
        raise ValueError("sigma_equivalent and temperature arrays must match in shape.")

    LOG.debug("Running Glinka correction on %d entries", sigma_equivalent.size)
    sig = np.asarray(sigma_equivalent, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    return solve_glinka_vector_core(sig, temp, material.TEMP, material.E_tab, material.SIG, material.EPSP,
                                    tol=tol, itmax=max_iterations, use_plateau=1 if use_plateau else 0)


def apply_ibg_correction(
    stress_history: np.ndarray,
    temperature_history: np.ndarray,
    material: MaterialDB,
    use_plateau: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Incremental Buczynski–Glinka correction to a stress history.

    Args:
        stress_history: Array of shape ``(N, 6)`` in Voigt order.
        temperature_history: Array of shape ``(N,)`` with nodal temperatures.
        material: Prepared ``MaterialDB`` instance.

    Returns:
        Tuple ``(corrected_history, plastic_strain_history)``.
    """
    if stress_history.ndim != 2 or stress_history.shape[1] != 6:
        raise ValueError("stress_history must have shape (N, 6) with Voigt ordering.")
    if temperature_history.shape[0] != stress_history.shape[0]:
        raise ValueError("temperature_history length must match stress_history rows.")

    LOG.debug("Running IBG correction on %d history steps", stress_history.shape[0])
    sig = np.asarray(stress_history, dtype=np.float64)
    temp = np.asarray(temperature_history, dtype=np.float64)
    return ibg_solver_tensor_core(sig, temp, material.TEMP, material.E_tab, material.SIG, material.EPSP,
                                  use_plateau=1 if use_plateau else 0)


def default_material_db() -> MaterialDB:
    """Return a trivial single-temperature material definition for smoke tests."""
    temp = np.array([22.0], dtype=np.float64)
    youngs = np.array([70_000.0], dtype=np.float64)
    sig = np.array([[400.0, 1_004.3]], dtype=np.float64)
    epsp = np.array([[0.0, 0.3007]], dtype=np.float64)
    return MaterialDB.from_arrays(temp, youngs, sig, epsp)


# ============================================================================
# MARS-SC (Solution Combination) Integration
# ============================================================================

def apply_plasticity_to_envelope(
    max_values: np.ndarray,
    temperature: np.ndarray,
    material: MaterialDB,
    method: str = "neuber",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply plasticity correction to envelope (max over combinations) stress values.
    
    This is the primary integration point for MARS-SC when processing envelope
    results. The envelope represents the maximum stress at each node across all
    load combinations.
    
    Args:
        max_values: Array of maximum equivalent stress values (e.g., von Mises)
                   over all combinations, shape (num_nodes,).
        temperature: Array of temperatures at each node, same shape as max_values.
        material: Prepared MaterialDB instance with hardening data.
        method: Correction method - "neuber" or "glinka".
        **kwargs: Additional arguments passed to the correction function.
    
    Returns:
        Tuple of (corrected_stress, plastic_strain) arrays.
    
    Example:
        >>> from solver.stress_engine import StressCombinationEngine
        >>> result = engine.compute_full_analysis()
        >>> corrected, strain = apply_plasticity_to_envelope(
        ...     result.max_over_combo, temperature_field, material_db
        ... )
    """
    if method.lower() == "neuber":
        return apply_neuber_correction(max_values, temperature, material, **kwargs)
    elif method.lower() == "glinka":
        return apply_glinka_correction(max_values, temperature, material, **kwargs)
    else:
        raise ValueError(f"Unknown plasticity method: {method}. Use 'neuber' or 'glinka'.")


def apply_plasticity_to_combination_history(
    stress_history: np.ndarray,
    temperature: float,
    material: MaterialDB,
    method: str = "neuber",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply plasticity correction to a single node's stress history over combinations.
    
    For MARS-SC combination history mode, this applies plasticity correction to
    the stress values computed at each combination for a specific node.
    
    Note: Unlike time-history analysis, combination results are independent
    (no path dependency), so Neuber/Glinka is applied per-combination.
    
    Args:
        stress_history: Array of equivalent stress values at each combination,
                       shape (num_combinations,).
        temperature: Scalar temperature for this node.
        material: Prepared MaterialDB instance.
        method: Correction method - "neuber" or "glinka".
        **kwargs: Additional arguments.
    
    Returns:
        Tuple of (corrected_stress, plastic_strain) arrays, same shape as input.
    """
    # Create temperature array matching stress history length
    temp_array = np.full_like(stress_history, temperature, dtype=np.float64)
    
    if method.lower() == "neuber":
        return apply_neuber_correction(stress_history, temp_array, material, **kwargs)
    elif method.lower() == "glinka":
        return apply_glinka_correction(stress_history, temp_array, material, **kwargs)
    else:
        raise ValueError(f"Unknown plasticity method: {method}. Use 'neuber' or 'glinka'.")


def apply_plasticity_to_single_combination(
    stress_values: np.ndarray,
    temperature: np.ndarray,
    material: MaterialDB,
    method: str = "neuber",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply plasticity correction to stress values from a single combination.
    
    For MARS-SC, this corrects the stress field computed at a specific
    combination index (all nodes, one combination).
    
    Args:
        stress_values: Array of equivalent stress values at each node,
                      shape (num_nodes,).
        temperature: Array of temperatures at each node.
        material: Prepared MaterialDB instance.
        method: Correction method - "neuber" or "glinka".
        **kwargs: Additional arguments.
    
    Returns:
        Tuple of (corrected_stress, plastic_strain) arrays.
    """
    if method.lower() == "neuber":
        return apply_neuber_correction(stress_values, temperature, material, **kwargs)
    elif method.lower() == "glinka":
        return apply_glinka_correction(stress_values, temperature, material, **kwargs)
    else:
        raise ValueError(f"Unknown plasticity method: {method}. Use 'neuber' or 'glinka'.")


def correct_combination_tensor_history(
    stress_tensor_history: np.ndarray,
    temperature_history: np.ndarray,
    material: MaterialDB,
    use_plateau: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply IBG (Incremental Buczynski-Glinka) correction to full tensor history.
    
    This is for advanced use cases where the full 6-component stress tensor
    is available for each combination and path-dependent effects are desired.
    
    Note: For linear combination of static results, path dependency is generally
    not physically meaningful. Use Neuber/Glinka for envelope corrections instead.
    
    Args:
        stress_tensor_history: Array of shape (num_combinations, 6) with stress
                              tensors in Voigt order [Sxx, Syy, Szz, Sxy, Syz, Sxz].
        temperature_history: Array of shape (num_combinations,) with temperatures.
        material: Prepared MaterialDB instance.
        use_plateau: If True, clamp extrapolation to curve endpoints.
    
    Returns:
        Tuple of:
            - corrected_tensor: Corrected stress tensors (num_combinations, 6)
            - plastic_strain: Cumulative plastic strain (num_combinations,)
            - von_mises_corrected: Von Mises of corrected tensors (num_combinations,)
    """
    corrected_tensor, plastic_strain = apply_ibg_correction(
        stress_tensor_history, 
        temperature_history, 
        material,
        use_plateau=use_plateau
    )
    
    # Compute von Mises of corrected tensors
    vm_corrected = np.array([
        von_mises_from_voigt(corrected_tensor[i])
        for i in range(corrected_tensor.shape[0])
    ])
    
    return corrected_tensor, plastic_strain, vm_corrected


def compute_damage_index_combination(
    max_stress: np.ndarray,
    corrected_stress: np.ndarray,
    plastic_strain: np.ndarray,
    yield_stress: float = 400.0
) -> np.ndarray:
    """
    Compute a simple damage/severity index for combination envelope results.
    
    This provides a normalized measure of how critical each node is based on
    stress levels and plasticity.
    
    Args:
        max_stress: Envelope max stress values (num_nodes,).
        corrected_stress: Plasticity-corrected stress values (num_nodes,).
        plastic_strain: Plastic strain values (num_nodes,).
        yield_stress: Reference yield stress for normalization.
    
    Returns:
        Damage index array (num_nodes,), normalized 0-1 where 1 is most critical.
    """
    # Simple damage index combining stress ratio and plastic strain
    stress_ratio = corrected_stress / (yield_stress + EPS)
    strain_factor = 1.0 + 10.0 * plastic_strain  # Amplify plastic strain contribution
    
    raw_index = stress_ratio * strain_factor
    
    # Normalize to 0-1 range
    max_index = np.max(raw_index)
    if max_index > 0:
        return raw_index / max_index
    return raw_index
