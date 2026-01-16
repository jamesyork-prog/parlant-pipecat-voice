#!/usr/bin/env python3
"""
Configuration validation script for MOR/Product Type router.

This script validates router configuration files and provides detailed
feedback about any issues found.

Usage:
    python validate_config.py [config_file]
    
If no config file is specified, validates the default router_config.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Import the config loader for validation logic
try:
    from config_loader import RouterConfigLoader, ConfigValidationError
except ImportError:
    # If running from different directory, try relative import
    import os
    sys.path.append(os.path.dirname(__file__))
    from config_loader import RouterConfigLoader, ConfigValidationError


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file and print results.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False if invalid
    """
    print(f"Validating configuration file: {config_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(config_path).exists():
        print(f"❌ ERROR: Configuration file not found: {config_path}")
        return False
    
    # Try to load and validate
    try:
        loader = RouterConfigLoader(config_path)
        config = loader.load_config(force_reload=True)
        validation_result = loader.get_validation_result()
        
        if validation_result is None:
            print("❌ ERROR: No validation result available")
            return False
        
        # Print validation results
        if validation_result.is_valid:
            print("✅ Configuration is VALID")
            
            if validation_result.warnings:
                print(f"\n⚠️  {len(validation_result.warnings)} warning(s):")
                for warning in validation_result.warnings:
                    print(f"   - {warning}")
            else:
                print("   No warnings found")
                
        else:
            print("❌ Configuration is INVALID")
            print(f"\n{len(validation_result.errors)} error(s) found:")
            for error in validation_result.errors:
                print(f"   - {error}")
            
            if validation_result.warnings:
                print(f"\n{len(validation_result.warnings)} warning(s):")
                for warning in validation_result.warnings:
                    print(f"   - {warning}")
        
        # Print configuration summary
        print("\n" + "=" * 60)
        print("Configuration Summary:")
        print("=" * 60)
        
        mor_config = config.get("mor_gate", {})
        pt_config = config.get("product_type_gate", {})
        router_config = config.get("router_settings", {})
        
        print(f"Third-party vendors: {len(mor_config.get('third_party_vendors', []))}")
        for vendor in mor_config.get('third_party_vendors', []):
            print(f"   - {vendor}")
        
        print(f"\nParkWhiz partners: {len(mor_config.get('parkwhiz_partners', []))}")
        for partner in mor_config.get('parkwhiz_partners', []):
            print(f"   - {partner}")
        
        print(f"\nSpecial partners: {len(mor_config.get('special_partners', {}))}")
        for partner, flag in mor_config.get('special_partners', {}).items():
            print(f"   - {partner}: {flag}")
        
        print(f"\nOn-demand threshold: {pt_config.get('on_demand_threshold_hours', 'not set')} hours")
        print(f"Hot reload enabled: {router_config.get('enable_hot_reload', 'not set')}")
        print(f"Gates enabled: {router_config.get('enable_mor_product_type_gates', 'not set')}")
        
        return validation_result.is_valid
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON syntax")
        print(f"   {e}")
        return False
    
    except Exception as e:
        print(f"❌ ERROR: Unexpected error during validation")
        print(f"   {e}")
        return False


def main():
    """Main entry point for configuration validation."""
    
    # Determine config file to validate
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to router_config.json in same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "router_config.json"
    
    # Validate the configuration
    is_valid = validate_config_file(str(config_path))
    
    # Exit with appropriate code
    if is_valid:
        print("\n✅ Configuration validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Configuration validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()