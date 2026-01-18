"""
Tests for the skip substeps feature in DPF reader.

Tests the get_last_substep_ids() method that identifies load steps and returns
only the final substep of each, useful for reducing result sets when RST files
contain many intermediate substeps.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from file_io.dpf_reader import DPFAnalysisReader, DPF_AVAILABLE

# Skip all tests if DPF is not available
pytestmark = pytest.mark.skipif(not DPF_AVAILABLE, reason="DPF not installed")

# Paths to RST files
RST_FILE_1 = os.path.join(os.path.dirname(__file__), '..', 'example_dataset', 'file_analysis1.rst')
RST_FILE_2 = os.path.join(os.path.dirname(__file__), '..', 'example_dataset', 'file_analysis2.rst')


class TestSkipSubsteps:
    """Test the skip substeps functionality in DPF reader."""
    
    @pytest.fixture(scope="class")
    def reader1(self):
        """Create DPF reader for RST file 1."""
        if not os.path.exists(RST_FILE_1):
            pytest.skip(f"RST file not found: {RST_FILE_1}")
        return DPFAnalysisReader(RST_FILE_1)
    
    @pytest.fixture(scope="class")
    def reader2(self):
        """Create DPF reader for RST file 2."""
        if not os.path.exists(RST_FILE_2):
            pytest.skip(f"RST file not found: {RST_FILE_2}")
        return DPFAnalysisReader(RST_FILE_2)
    
    def test_get_all_load_step_ids(self, reader1):
        """Test getting all load step IDs without filtering."""
        all_ids = reader1.get_load_step_ids()
        n_sets = reader1.get_load_step_count()
        
        print(f"\n{'='*60}")
        print("TEST: get_load_step_ids() - All sets")
        print(f"{'='*60}")
        print(f"Total sets (n_sets): {n_sets}")
        print(f"All step IDs: {all_ids}")
        
        assert len(all_ids) == n_sets
        assert all_ids == list(range(1, n_sets + 1))
    
    def test_get_last_substep_ids(self, reader1):
        """Test getting only the last substep of each load step."""
        all_ids = reader1.get_load_step_ids()
        last_substep_ids = reader1.get_last_substep_ids()
        
        print(f"\n{'='*60}")
        print("TEST: get_last_substep_ids() - Filtered sets")
        print(f"{'='*60}")
        print(f"All step IDs ({len(all_ids)} total): {all_ids}")
        print(f"Last substep IDs ({len(last_substep_ids)} total): {last_substep_ids}")
        
        # Last substep IDs should be a subset of all IDs
        for sid in last_substep_ids:
            assert sid in all_ids, f"Last substep ID {sid} not in all IDs"
        
        # Should have fewer or equal number of sets
        assert len(last_substep_ids) <= len(all_ids)
        
        # If there are substeps, we should have fewer IDs
        # (unless each load step has only 1 substep)
        print(f"\nReduction: {len(all_ids)} -> {len(last_substep_ids)} sets")
        print(f"Substeps skipped: {len(all_ids) - len(last_substep_ids)}")
    
    def test_get_last_substep_ids_time_values(self, reader1):
        """Verify that last substep IDs correspond to correct time values."""
        all_ids = reader1.get_load_step_ids()
        last_substep_ids = reader1.get_last_substep_ids()
        
        all_times = reader1.get_time_values()
        last_times = reader1.get_time_values(set_ids=last_substep_ids)
        
        print(f"\n{'='*60}")
        print("TEST: Time values for last substeps")
        print(f"{'='*60}")
        print(f"All time values ({len(all_times)}): {[f'{t:.4g}s' for t in all_times]}")
        print(f"Last substep times ({len(last_times)}): {[f'{t:.4g}s' for t in last_times]}")
        
        # Time values for last substeps should be a subset of all times
        for t in last_times:
            assert t in all_times or any(abs(t - at) < 1e-10 for at in all_times)
        
        # The time values should be the "end times" of each load step
        # (i.e., the highest time in each group)
        print(f"\nLast substep times should be end-of-loadstep times")
    
    def test_get_analysis_data_with_skip_substeps(self, reader1):
        """Test get_analysis_data with skip_substeps=True."""
        # Get analysis data without skipping
        data_all = reader1.get_analysis_data(skip_substeps=False)
        
        # Get analysis data with skipping
        data_filtered = reader1.get_analysis_data(skip_substeps=True)
        
        print(f"\n{'='*60}")
        print("TEST: get_analysis_data(skip_substeps=True)")
        print(f"{'='*60}")
        print(f"\nWithout skip_substeps:")
        print(f"  num_load_steps: {data_all.num_load_steps}")
        print(f"  load_step_ids: {data_all.load_step_ids}")
        print(f"  time_values: {[f'{t:.4g}s' for t in data_all.time_values]}")
        
        print(f"\nWith skip_substeps=True:")
        print(f"  num_load_steps: {data_filtered.num_load_steps}")
        print(f"  load_step_ids: {data_filtered.load_step_ids}")
        print(f"  time_values: {[f'{t:.4g}s' for t in data_filtered.time_values]}")
        
        # Filtered should have fewer or equal load steps
        assert data_filtered.num_load_steps <= data_all.num_load_steps
        
        # Number of steps should match length of IDs and times
        assert data_filtered.num_load_steps == len(data_filtered.load_step_ids)
        assert data_filtered.num_load_steps == len(data_filtered.time_values)
        
        # Filtered IDs should be a subset
        for sid in data_filtered.load_step_ids:
            assert sid in data_all.load_step_ids
        
        print(f"\nReduction: {data_all.num_load_steps} -> {data_filtered.num_load_steps} load steps")
    
    def test_skip_substeps_both_files(self, reader1, reader2):
        """Test skip substeps on both RST files."""
        print(f"\n{'='*60}")
        print("TEST: Skip substeps on both RST files")
        print(f"{'='*60}")
        
        for name, reader in [("Analysis 1", reader1), ("Analysis 2", reader2)]:
            all_ids = reader.get_load_step_ids()
            last_ids = reader.get_last_substep_ids()
            all_times = reader.get_time_values()
            last_times = reader.get_time_values(set_ids=last_ids)
            
            print(f"\n{name}:")
            print(f"  All sets: {len(all_ids)} - IDs: {all_ids}")
            print(f"  Last substeps: {len(last_ids)} - IDs: {last_ids}")
            print(f"  All times: {[f'{t:.4g}s' for t in all_times]}")
            print(f"  Last times: {[f'{t:.4g}s' for t in last_times]}")
            
            assert len(last_ids) <= len(all_ids)
            assert len(last_times) == len(last_ids)
    
    def test_step_substep_mapping(self, reader1):
        """Test the step/substep to cumulative index mapping."""
        reader = reader1  # Alias for use in assertions
        tf_support = reader1.time_freq_support
        n_sets = tf_support.n_sets
        
        print(f"\n{'='*60}")
        print("TEST: Step/Substep to Cumulative Index Mapping")
        print(f"{'='*60}")
        print(f"Total sets: {n_sets}")
        
        # Try to build the step/substep mapping
        step_info = {}
        
        step = 1
        while True:
            substep = 1
            found_any = False
            
            while True:
                try:
                    cumulative_idx = tf_support.get_cumulative_index(step=step, substep=substep)
                    if cumulative_idx is not None and cumulative_idx > 0:
                        if step not in step_info:
                            step_info[step] = []
                        time_val = tf_support.get_frequency(cumulative_index=cumulative_idx)
                        step_info[step].append({
                            'substep': substep,
                            'cumulative_idx': cumulative_idx,
                            'time': time_val
                        })
                        found_any = True
                        substep += 1
                    else:
                        break
                except Exception:
                    break
                
                if substep > 100:  # Safety limit
                    break
            
            if not found_any and step > 1:
                break
            step += 1
            
            if step > 100:  # Safety limit
                break
        
        # Print the mapping
        print(f"\nFound {len(step_info)} load steps:")
        for load_step, substeps in sorted(step_info.items()):
            print(f"\n  Load Step {load_step}:")
            for info in substeps:
                print(f"    Substep {info['substep']}: cumulative_idx={info['cumulative_idx']}, time={info['time']:.4g}s")
            print(f"    -> Last substep: {substeps[-1]['cumulative_idx']} at time {substeps[-1]['time']:.4g}s")
        
        # Verify our get_last_substep_ids returns the correct indices
        last_ids = reader1.get_last_substep_ids()
        expected_last_ids = [step_info[s][-1]['cumulative_idx'] for s in sorted(step_info.keys())]
        
        print(f"\nExpected last substep IDs (from DPF API): {expected_last_ids}")
        print(f"Actual get_last_substep_ids(): {last_ids}")
        
        # Check coverage - did we find all sets via DPF API?
        total_found = sum(len(substeps) for substeps in step_info.values())
        print(f"\nDPF API Coverage: Found {total_found} of {n_sets} sets via step/substep mapping")
        
        if total_found < n_sets * 0.8:
            print("Note: Less than 80% coverage via DPF API")
            print("The implementation will use time analysis instead")
            
            # Check if time analysis found reasonable boundaries
            # For the example file with times [0.2, 0.4, 0.7, 1.0, 1.2, 1.4, 1.7, 2.0]
            # Integer times are at indices 4 (t=1s) and 8 (t=2s)
            all_times = reader.get_time_values()
            integer_time_sets = [i+1 for i, t in enumerate(all_times) if abs(t - round(t)) < 1e-6 and t > 0]
            print(f"Integer time values found at sets: {integer_time_sets}")
            
            if integer_time_sets:
                print(f"Time analysis should return: {integer_time_sets}")
                assert last_ids == integer_time_sets, \
                    f"Expected {integer_time_sets} from time analysis, got {last_ids}"
            else:
                # No integer times, should fall back to all sets
                assert last_ids == list(range(1, n_sets + 1)), \
                    "With no integer times, should return all sets"
        else:
            assert last_ids == expected_last_ids, \
                f"Last substep IDs mismatch: expected {expected_last_ids}, got {last_ids}"
        
        print("\n[OK] get_last_substep_ids() returns correct indices!")


def run_tests_standalone():
    """Run tests as standalone script for easier debugging."""
    print("\n" + "#"*80)
    print("# SKIP SUBSTEPS FEATURE TESTS")
    print("#"*80)
    
    if not DPF_AVAILABLE:
        print("\nERROR: DPF is not available. Install with: pip install ansys-dpf-core")
        return
    
    if not os.path.exists(RST_FILE_1):
        print(f"\nERROR: RST file not found: {RST_FILE_1}")
        return
    
    # Create readers
    reader1 = DPFAnalysisReader(RST_FILE_1)
    reader2 = DPFAnalysisReader(RST_FILE_2) if os.path.exists(RST_FILE_2) else None
    
    # Create test instance
    test = TestSkipSubsteps()
    
    # Run tests
    print("\n>>> Running: test_get_all_load_step_ids")
    test.test_get_all_load_step_ids(reader1)
    
    print("\n>>> Running: test_get_last_substep_ids")
    test.test_get_last_substep_ids(reader1)
    
    print("\n>>> Running: test_get_last_substep_ids_time_values")
    test.test_get_last_substep_ids_time_values(reader1)
    
    print("\n>>> Running: test_get_analysis_data_with_skip_substeps")
    test.test_get_analysis_data_with_skip_substeps(reader1)
    
    print("\n>>> Running: test_step_substep_mapping")
    test.test_step_substep_mapping(reader1)
    
    if reader2:
        print("\n>>> Running: test_skip_substeps_both_files")
        test.test_skip_substeps_both_files(reader1, reader2)
    
    print("\n" + "#"*80)
    print("# ALL SKIP SUBSTEPS TESTS COMPLETED")
    print("#"*80)


if __name__ == "__main__":
    run_tests_standalone()
