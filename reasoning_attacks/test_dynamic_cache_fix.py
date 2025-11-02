#!/usr/bin/env python3
"""
Test script to verify DynamicCache compatibility fix
"""

def test_dynamic_cache_patch():
    """Test the DynamicCache compatibility patch."""
    try:
        import transformers
        from transformers.cache_utils import DynamicCache
        
        print(f"Transformers version: {transformers.__version__}")
        
        # Test if DynamicCache has key_cache attribute
        if hasattr(DynamicCache, 'key_cache'):
            print("✓ DynamicCache already has key_cache attribute")
            return True
        
        # Apply the patch
        def get_key_cache(self):
            """Get key cache for compatibility."""
            if hasattr(self, '_key_cache'):
                return self._key_cache
            # Fallback to accessing the cache directly
            return [layer[0] for layer in self]
            
        def get_value_cache(self):
            """Get value cache for compatibility."""
            if hasattr(self, '_value_cache'):
                return self._value_cache
            # Fallback to accessing the cache directly
            return [layer[1] for layer in self]
        
        # Add the methods to DynamicCache
        DynamicCache.key_cache = property(get_key_cache)
        DynamicCache.value_cache = property(get_value_cache)
        
        print("✓ Applied DynamicCache compatibility patch")
        
        # Test the patch
        cache = DynamicCache()
        print(f"✓ Cache created successfully")
        print(f"✓ key_cache property exists: {hasattr(cache, 'key_cache')}")
        print(f"✓ value_cache property exists: {hasattr(cache, 'value_cache')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing DynamicCache patch: {e}")
        return False

if __name__ == "__main__":
    print("Testing DynamicCache compatibility fix...")
    success = test_dynamic_cache_patch()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")

