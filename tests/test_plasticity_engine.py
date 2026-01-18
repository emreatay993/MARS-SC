"""
Tests for the Plasticity Engine.

Tests Neuber, Glinka, and IBG plasticity corrections, including
MARS-SC integration functions.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from solver.plasticity_engine import (
    MaterialDB,
    POISSON_RATIO,
    apply_neuber_correction,
    apply_glinka_correction,
    apply_ibg_correction,
    von_mises_from_voigt,
    default_material_db,
    shear_modulus,
    deviator,
    apply_plasticity_to_envelope,
    apply_plasticity_to_combination_history,
    apply_plasticity_to_single_combination,
    compute_damage_index_combination,
)


class TestMaterialDB:
    """Tests for MaterialDB creation and operations."""
    
    def test_from_arrays_valid(self):
        """Test creating MaterialDB from valid arrays."""
        temp = np.array([20.0, 100.0, 200.0])
        e_tab = np.array([200000.0, 195000.0, 185000.0])
        sig = np.array([
            [300.0, 400.0, 500.0],
            [280.0, 380.0, 480.0],
            [250.0, 350.0, 450.0],
        ])
        epsp = np.array([
            [0.0, 0.01, 0.05],
            [0.0, 0.012, 0.055],
            [0.0, 0.015, 0.06],
        ])
        
        db = MaterialDB.from_arrays(temp, e_tab, sig, epsp)
        
        assert db.TEMP.shape == (3,)
        assert db.E_tab.shape == (3,)
        assert db.SIG.shape == (3, 3)
        assert db.EPSP.shape == (3, 3)
    
    def test_from_arrays_invalid_temp_order(self):
        """Test that non-increasing temperature raises error."""
        temp = np.array([100.0, 50.0, 200.0])  # Not strictly increasing
        e_tab = np.array([200000.0, 195000.0, 185000.0])
        sig = np.array([[300.0, 400.0], [280.0, 380.0], [250.0, 350.0]])
        epsp = np.array([[0.0, 0.01], [0.0, 0.012], [0.0, 0.015]])
        
        with pytest.raises(ValueError, match="strictly increasing"):
            MaterialDB.from_arrays(temp, e_tab, sig, epsp)
    
    def test_from_arrays_invalid_sig_order(self):
        """Test that non-increasing stress raises error."""
        temp = np.array([20.0, 100.0])
        e_tab = np.array([200000.0, 195000.0])
        sig = np.array([[400.0, 300.0], [280.0, 380.0]])  # First row not increasing
        epsp = np.array([[0.0, 0.01], [0.0, 0.012]])
        
        with pytest.raises(ValueError, match="strictly increasing"):
            MaterialDB.from_arrays(temp, e_tab, sig, epsp)
    
    def test_from_arrays_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        temp = np.array([20.0, 100.0])
        e_tab = np.array([200000.0, 195000.0, 180000.0])  # Wrong length
        sig = np.array([[300.0, 400.0], [280.0, 380.0]])
        epsp = np.array([[0.0, 0.01], [0.0, 0.012]])
        
        with pytest.raises(ValueError):
            MaterialDB.from_arrays(temp, e_tab, sig, epsp)


class TestDefaultMaterialDB:
    """Tests for the default test material."""
    
    def test_default_material_creation(self):
        """Test that default material can be created."""
        db = default_material_db()
        
        assert db.TEMP.size == 1
        assert db.E_tab[0] == 70000.0  # Aluminum-like
    
    def test_default_material_yield(self):
        """Test yield stress of default material."""
        db = default_material_db()
        
        # First point of SIG is yield
        assert db.SIG[0, 0] == 400.0


class TestVonMisesFromVoigt:
    """Tests for von Mises calculation from Voigt notation."""
    
    def test_uniaxial_tension(self):
        """Test von Mises for uniaxial tension."""
        sig6 = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vm = von_mises_from_voigt(sig6)
        
        np.testing.assert_almost_equal(vm, 100.0)
    
    def test_pure_shear(self):
        """Test von Mises for pure shear."""
        sig6 = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
        vm = von_mises_from_voigt(sig6)
        
        expected = np.sqrt(3) * 100.0
        np.testing.assert_almost_equal(vm, expected)
    
    def test_hydrostatic(self):
        """Test von Mises for hydrostatic (should be zero)."""
        p = 100.0
        sig6 = np.array([p, p, p, 0.0, 0.0, 0.0])
        vm = von_mises_from_voigt(sig6)
        
        np.testing.assert_almost_equal(vm, 0.0)


class TestShearModulus:
    """Tests for shear modulus calculation."""
    
    def test_default_poisson(self):
        """Test shear modulus with default Poisson's ratio."""
        E = 200000.0
        G = shear_modulus(E)
        
        expected = E / (2 * (1 + POISSON_RATIO))
        np.testing.assert_almost_equal(G, expected)
    
    def test_custom_poisson(self):
        """Test shear modulus with custom Poisson's ratio."""
        E = 200000.0
        nu = 0.25
        G = shear_modulus(E, nu)
        
        expected = E / (2 * (1 + nu))
        np.testing.assert_almost_equal(G, expected)


class TestDeviator:
    """Tests for deviatoric stress calculation."""
    
    def test_pure_deviatoric(self):
        """Test deviator of already deviatoric stress."""
        sig6 = np.array([100.0, -50.0, -50.0, 30.0, 20.0, 10.0])
        dev = deviator(sig6)
        
        # Mean should be zero for deviatoric part
        mean = (dev[0] + dev[1] + dev[2]) / 3.0
        np.testing.assert_almost_equal(mean, 0.0)
    
    def test_hydrostatic_has_zero_deviator(self):
        """Test that hydrostatic stress has zero deviatoric part."""
        p = 100.0
        sig6 = np.array([p, p, p, 0.0, 0.0, 0.0])
        dev = deviator(sig6)
        
        np.testing.assert_array_almost_equal(dev, np.zeros(6))


class TestNeuberCorrection:
    """Tests for Neuber plasticity correction."""
    
    def test_below_yield_no_correction(self):
        """Test that stress below yield is not significantly corrected."""
        db = default_material_db()  # Yield = 400 MPa
        
        sigma_e = np.array([200.0])  # Well below yield
        temp = np.array([22.0])
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        # Below yield, correction should be minimal
        np.testing.assert_almost_equal(corrected[0], sigma_e[0], decimal=0)
        np.testing.assert_almost_equal(strain[0], 0.0, decimal=5)
    
    def test_above_yield_correction(self):
        """Test that stress above yield is reduced."""
        db = default_material_db()  # Yield = 400 MPa
        
        sigma_e = np.array([500.0])  # Above yield
        temp = np.array([22.0])
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        # Corrected should be less than elastic
        assert corrected[0] < sigma_e[0]
        # Plastic strain should be positive
        assert strain[0] > 0
    
    def test_vectorized_correction(self):
        """Test vectorized Neuber correction."""
        db = default_material_db()
        
        sigma_e = np.array([200.0, 400.0, 500.0, 600.0])
        temp = np.full(4, 22.0)
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        assert len(corrected) == 4
        assert len(strain) == 4
        # Higher input should give higher output (monotonic)
        assert np.all(np.diff(corrected) >= -1e-6)
    
    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise error."""
        db = default_material_db()
        
        sigma_e = np.array([200.0, 300.0])
        temp = np.array([22.0])  # Wrong shape
        
        with pytest.raises(ValueError):
            apply_neuber_correction(sigma_e, temp, db)


class TestGlinkaCorrection:
    """Tests for Glinka (energy-density) plasticity correction."""
    
    def test_below_yield_no_correction(self):
        """Test that stress below yield is not significantly corrected."""
        db = default_material_db()
        
        sigma_e = np.array([200.0])
        temp = np.array([22.0])
        
        corrected, strain = apply_glinka_correction(sigma_e, temp, db)
        
        np.testing.assert_almost_equal(corrected[0], sigma_e[0], decimal=0)
        np.testing.assert_almost_equal(strain[0], 0.0, decimal=5)
    
    def test_above_yield_correction(self):
        """Test that stress above yield is reduced."""
        db = default_material_db()
        
        sigma_e = np.array([500.0])
        temp = np.array([22.0])
        
        corrected, strain = apply_glinka_correction(sigma_e, temp, db)
        
        assert corrected[0] < sigma_e[0]
        assert strain[0] > 0
    
    def test_glinka_vs_neuber(self):
        """Test that Glinka gives different (usually higher) results than Neuber."""
        db = default_material_db()
        
        sigma_e = np.array([600.0])
        temp = np.array([22.0])
        
        neuber_corr, _ = apply_neuber_correction(sigma_e, temp, db)
        glinka_corr, _ = apply_glinka_correction(sigma_e, temp, db)
        
        # Glinka typically gives slightly higher corrected stress
        # This is a known difference in the methods
        # Just verify they're different
        assert not np.allclose(neuber_corr, glinka_corr)


class TestIBGCorrection:
    """Tests for Incremental Buczynski-Glinka correction."""
    
    def test_basic_history(self):
        """Test IBG with simple loading history."""
        db = default_material_db()
        
        # Simple ramp loading
        stress_history = np.array([
            [0, 0, 0, 0, 0, 0],
            [200, 0, 0, 0, 0, 0],
            [400, 0, 0, 0, 0, 0],
            [500, 0, 0, 0, 0, 0],
        ], dtype=np.float64)
        temp_history = np.full(4, 22.0)
        
        corrected, strain = apply_ibg_correction(stress_history, temp_history, db)
        
        assert corrected.shape == (4, 6)
        assert strain.shape == (4,)
        # Strain should be non-decreasing for monotonic loading
        assert np.all(np.diff(strain) >= -1e-10)
    
    def test_shape_validation(self):
        """Test that wrong shapes raise errors."""
        db = default_material_db()
        
        # Wrong number of columns
        stress_history = np.array([[1, 2, 3]])  # Should be 6 columns
        temp_history = np.array([22.0])
        
        with pytest.raises(ValueError, match="shape"):
            apply_ibg_correction(stress_history, temp_history, db)


class TestMARSSCIntegration:
    """Tests for MARS-SC specific plasticity functions."""
    
    def test_apply_plasticity_to_envelope(self):
        """Test envelope plasticity correction."""
        db = default_material_db()
        
        max_values = np.array([300.0, 450.0, 550.0, 650.0])
        temperature = np.full(4, 22.0)
        
        corrected, strain = apply_plasticity_to_envelope(
            max_values, temperature, db, method="neuber"
        )
        
        assert len(corrected) == 4
        # Values above yield should be reduced
        assert corrected[2] < max_values[2]
        assert corrected[3] < max_values[3]
    
    def test_apply_plasticity_to_envelope_invalid_method(self):
        """Test that invalid method raises error."""
        db = default_material_db()
        
        max_values = np.array([500.0])
        temperature = np.array([22.0])
        
        with pytest.raises(ValueError, match="Unknown plasticity method"):
            apply_plasticity_to_envelope(max_values, temperature, db, method="invalid")
    
    def test_apply_plasticity_to_combination_history(self):
        """Test combination history plasticity correction."""
        db = default_material_db()
        
        stress_history = np.array([300.0, 400.0, 500.0, 450.0, 350.0])
        temperature = 22.0
        
        corrected, strain = apply_plasticity_to_combination_history(
            stress_history, temperature, db, method="glinka"
        )
        
        assert len(corrected) == 5
        assert len(strain) == 5
    
    def test_apply_plasticity_to_single_combination(self):
        """Test single combination plasticity correction."""
        db = default_material_db()
        
        stress_values = np.array([350.0, 420.0, 510.0])
        temperature = np.full(3, 22.0)
        
        corrected, strain = apply_plasticity_to_single_combination(
            stress_values, temperature, db, method="neuber"
        )
        
        assert len(corrected) == 3


class TestDamageIndexComputation:
    """Tests for damage/severity index computation."""
    
    def test_basic_damage_index(self):
        """Test basic damage index calculation."""
        max_stress = np.array([300.0, 450.0, 600.0])
        corrected_stress = np.array([300.0, 430.0, 520.0])
        plastic_strain = np.array([0.0, 0.005, 0.02])
        
        damage = compute_damage_index_combination(
            max_stress, corrected_stress, plastic_strain,
            yield_stress=400.0
        )
        
        assert len(damage) == 3
        # Max damage should be 1.0 (normalized)
        np.testing.assert_almost_equal(np.max(damage), 1.0)
        # Higher stress/strain should have higher damage
        assert damage[2] > damage[1] > damage[0]
    
    def test_all_elastic(self):
        """Test damage index when all elastic."""
        max_stress = np.array([200.0, 250.0, 300.0])
        corrected_stress = max_stress.copy()
        plastic_strain = np.zeros(3)
        
        damage = compute_damage_index_combination(
            max_stress, corrected_stress, plastic_strain,
            yield_stress=400.0
        )
        
        assert len(damage) == 3
        # Damage should be proportional to stress ratio
        assert damage[2] > damage[0]


class TestEdgeCases:
    """Edge case tests for plasticity engine."""
    
    def test_zero_stress(self):
        """Test with zero stress input."""
        db = default_material_db()
        
        sigma_e = np.array([0.0])
        temp = np.array([22.0])
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        np.testing.assert_almost_equal(corrected[0], 0.0)
        np.testing.assert_almost_equal(strain[0], 0.0)
    
    def test_single_element(self):
        """Test with single element."""
        db = default_material_db()
        
        sigma_e = np.array([500.0])
        temp = np.array([22.0])
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        assert corrected.shape == (1,)
        assert strain.shape == (1,)
    
    def test_large_array(self):
        """Test with large array for performance."""
        db = default_material_db()
        
        n = 10000
        sigma_e = np.random.rand(n) * 600 + 100  # 100-700 MPa
        temp = np.full(n, 22.0)
        
        corrected, strain = apply_neuber_correction(sigma_e, temp, db)
        
        assert len(corrected) == n
        assert len(strain) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
