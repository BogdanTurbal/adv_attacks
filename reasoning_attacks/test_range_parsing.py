#!/usr/bin/env python3
"""
Test script for example range parsing functionality
"""

def parse_example_range(range_str: str, total_examples: int):
    """Parse example range string and return start/end indices."""
    if not range_str:
        return 0, total_examples
    
    try:
        start_str, end_str = range_str.split(':')
        
        # Check if percentages (contain decimal point)
        if '.' in start_str or '.' in end_str:
            start_pct = float(start_str)
            end_pct = float(end_str)
            
            if not (0.0 <= start_pct <= 1.0 and 0.0 <= end_pct <= 1.0):
                raise ValueError("Percentages must be between 0.0 and 1.0")
            
            start_idx = int(start_pct * total_examples)
            end_idx = int(end_pct * total_examples)
        else:
            # Integer indices
            start_idx = int(start_str)
            end_idx = int(end_str)
            
            if start_idx < 0 or end_idx > total_examples:
                raise ValueError(f"Indices must be between 0 and {total_examples}")
        
        if start_idx >= end_idx:
            raise ValueError("Start index must be less than end index")
            
        return start_idx, end_idx
        
    except ValueError as e:
        raise ValueError(f"Invalid range format '{range_str}': {e}")


def test_range_parsing():
    """Test the range parsing function with various inputs."""
    total_examples = 1000
    
    test_cases = [
        # (range_str, expected_start, expected_end, description)
        ("0.25:0.5", 250, 500, "25% to 50% of examples"),
        ("0.0:0.25", 0, 250, "First 25% of examples"),
        ("0.75:1.0", 750, 1000, "Last 25% of examples"),
        ("0:100", 0, 100, "First 100 examples"),
        ("100:200", 100, 200, "Examples 100-199"),
        ("500:1000", 500, 1000, "Examples 500-999"),
        ("", 0, 1000, "Empty range (all examples)"),
    ]
    
    print("Testing example range parsing:")
    print(f"Total examples: {total_examples}")
    print("-" * 50)
    
    for range_str, expected_start, expected_end, description in test_cases:
        try:
            start_idx, end_idx = parse_example_range(range_str, total_examples)
            status = "✓" if start_idx == expected_start and end_idx == expected_end else "✗"
            print(f"{status} {range_str:10} -> {start_idx:4}-{end_idx:4} ({description})")
        except ValueError as e:
            print(f"✗ {range_str:10} -> ERROR: {e}")
    
    print("\nTesting error cases:")
    error_cases = [
        ("0.5:0.25", "Invalid range (start > end)"),
        ("1.5:2.0", "Invalid percentage (> 1.0)"),
        ("-1:100", "Invalid index (< 0)"),
        ("0:1001", "Invalid index (> total)"),
        ("invalid", "Invalid format"),
    ]
    
    for range_str, description in error_cases:
        try:
            start_idx, end_idx = parse_example_range(range_str, total_examples)
            print(f"✗ {range_str:10} -> Should have failed but didn't ({description})")
        except ValueError as e:
            print(f"✓ {range_str:10} -> Correctly failed: {e}")


if __name__ == "__main__":
    test_range_parsing()
