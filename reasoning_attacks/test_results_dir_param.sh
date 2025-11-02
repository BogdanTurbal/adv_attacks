#!/bin/bash

# Test script to verify the new results directory parameter functionality

echo "Testing submit_parallel_jobs.sh with results directory parameter..."

# Test 1: Default results directory
echo "Test 1: Default results directory (results_ort_v3_1f)"
echo "Command: ./submit_parallel_jobs.sh 2 percentage 0.0:0.5"
echo "Expected: Uses results_ort_v3_1f as default"

# Test 2: Custom results directory
echo -e "\nTest 2: Custom results directory (results_gcg_v2_0f)"
echo "Command: ./submit_parallel_jobs.sh 2 percentage 0.0:0.5 results_gcg_v2_0f"
echo "Expected: Uses results_gcg_v2_0f as results directory"

# Test 3: Consolidate with default directory
echo -e "\nTest 3: Consolidate with default directory"
echo "Command: ./consolidate_results.sh"
echo "Expected: Consolidates results_ort_v3_1f"

# Test 4: Consolidate with custom directory
echo -e "\nTest 4: Consolidate with custom directory"
echo "Command: ./consolidate_results.sh results_gcg_v2_0f"
echo "Expected: Consolidates results_gcg_v2_0f"

echo -e "\nAll tests are dry runs - no actual jobs will be submitted."
echo "To run actual tests, remove the 'echo' commands and execute the scripts."
