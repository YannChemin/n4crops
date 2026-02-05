#!/usr/bin/env python3
"""
Calibration Model Validation Script

This script validates that all calibration models produce realistic outputs
with proper statistical variation. It checks for clipping issues and
ensures ranges match literature values.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from calibration import CropCalibration, CropType
except ImportError:
    print("Error: Cannot import calibration module")
    sys.exit(1)


class CalibrationValidator:
    """Validate calibration models for realistic outputs and ranges."""
    
    def __init__(self):
        self.calib = CropCalibration()
        self.issues = []
        
    def validate_all_models(self) -> Dict[str, List]:
        """Validate all calibration models across all crops and stages."""
        print("🔍 CALIBRATION MODEL VALIDATION")
        print("=" * 60)
        
        crops = [CropType.RICE, CropType.MAIZE, CropType.WHEAT]
        stages_to_check = {
            CropType.RICE: ['tillering', 'booting', 'heading', 'filling', 'vegetative', 'reproductive'],
            CropType.MAIZE: ['V12', 'V16', 'V18', 'R1', 'R2', 'R6', 'vegetative'],
            CropType.WHEAT: ['jointing', 'heading_wheat', 'anthesis', 'grain_fill', 'vegetative']
        }
        
        results = {}
        
        for crop in crops:
            crop_results = []
            print(f"\n📊 {crop.name.upper()} MODELS:")
            print("-" * 30)
            
            for stage in stages_to_check.get(crop, []):
                try:
                    models = self.calib.get_models(crop, stage)
                    if not models:
                        continue
                        
                    stage_results = []
                    for param_name, model_info in models.items():
                        validation_result = self.validate_single_model(
                            crop.name, stage, param_name, model_info
                        )
                        stage_results.append(validation_result)
                        
                        # Print status
                        status = "✅" if validation_result['passed'] else "❌"
                        print(f"  {status} {stage}.{param_name}")
                        if not validation_result['passed']:
                            print(f"      Issues: {', '.join(validation_result['issues'])}")
                    
                    crop_results.extend(stage_results)
                    
                except Exception as e:
                    print(f"  ❌ Error checking {stage}: {e}")
                    
            results[crop.name] = crop_results
            
        return results
    
    def validate_single_model(self, crop: str, stage: str, param_name: str, 
                            model_info: Dict) -> Dict:
        """Validate a single calibration model."""
        model_type = model_info['model_type']
        coeffs = model_info['coefficients']
        valid_range = model_info.get('valid_range', (-np.inf, np.inf))
        units = model_info.get('units', '')
        
        # Generate test inputs based on model type and typical index ranges
        test_inputs = self.generate_test_inputs(model_type, model_info['index'])
        
        # Apply model
        try:
            test_outputs = self.calib.apply_model(test_inputs, model_info)
        except Exception as e:
            return {
                'crop': crop,
                'stage': stage,
                'param': param_name,
                'passed': False,
                'issues': [f'MODEL_APPLICATION_ERROR: {e}'],
                'output_range': (0, 0),
                'valid_range': valid_range,
                'clipping_ratio': 1.0
            }
        
        # Check for clipping
        clipped_outputs = np.clip(test_outputs, valid_range[0], valid_range[1])
        clipping_ratio = np.sum(clipped_outputs != test_outputs) / len(test_outputs)
        
        # Identify issues
        issues = []
        
        # Check for excessive clipping
        if clipping_ratio > 0.9:
            issues.append('EXCESSIVE_CLIPPING')
        elif clipping_ratio > 0.5:
            issues.append('HIGH_CLIPPING')
            
        # Check for zero variation
        if np.std(test_outputs) < 0.001:
            issues.append('ZERO_VARIATION')
        elif np.std(test_outputs) < 0.01:
            issues.append('LOW_VARIATION')
            
        # Check if all clipped to same value
        if len(np.unique(clipped_outputs)) == 1:
            issues.append('CONSTANT_AFTER_CLIPPING')
            
        # Check range appropriateness based on units
        range_issues = self.check_range_appropriateness(valid_range, units, param_name)
        issues.extend(range_issues)
        
        # Check if model outputs are within expected biological limits
        bio_issues = self.check_biological_limits(test_outputs, units, param_name)
        issues.extend(bio_issues)
        
        return {
            'crop': crop,
            'stage': stage,
            'param': param_name,
            'passed': len(issues) == 0,
            'issues': issues,
            'output_range': (np.min(test_outputs), np.max(test_outputs)),
            'valid_range': valid_range,
            'clipping_ratio': clipping_ratio,
            'std_dev': np.std(test_outputs)
        }
    
    def generate_test_inputs(self, model_type: str, index_name: str) -> np.ndarray:
        """Generate appropriate test inputs for different vegetation indices."""
        # Define typical ranges for different indices
        index_ranges = {
            'NDVI': (-0.2, 0.9),
            'NDRE': (-0.1, 0.8),
            'GNDVI': (-0.3, 0.8),
            'RVI_II': (-0.2, 1.0),
            'MTCI': (0.0, 6.0),
            'HBSI1': (-0.5, 0.8),
            'HBSI2': (-0.5, 0.8),
            'HBCI8': (-0.2, 0.5),
            'HBCI9': (-0.2, 0.5),
            'TCARI_OSAVI': (0.1, 0.8),
            'default': (-0.2, 1.0)
        }
        
        # Get range for this index or use default
        min_val, max_val = index_ranges.get(index_name, index_ranges['default'])
        
        # Generate 10 test values across the range
        return np.linspace(min_val, max_val, 10)
    
    def check_range_appropriateness(self, valid_range: Tuple, units: str, param_name: str) -> List[str]:
        """Check if valid range is appropriate for the parameter units."""
        issues = []
        min_val, max_val = valid_range
        
        # Check for percentage-based parameters
        if '%' in units:
            if max_val > 15.0:
                issues.append('UNREALISTIC_HIGH_PERCENTAGE')
            if min_val > 8.0:
                issues.append('UNREALISTIC_MIN_PERCENTAGE')
            if max_val < 0.5 and 'leaf_N' in param_name:
                issues.append('TOO_LOW_FOR_LEAF_N')
                
        # Check for kg kg⁻¹ N (efficiency parameters)
        elif 'kg kg⁻¹' in units:
            if max_val < 10.0:
                issues.append('TOO_LOW_FOR_EFFICIENCY')
            if min_val > 200.0:
                issues.append('TOO_HIGH_FOR_EFFICIENCY')
                
        # Check for kg ha⁻¹ (uptake parameters)
        elif 'kg ha' in units:
            if max_val < 1.0:
                issues.append('TOO_LOW_FOR_UPTAKE')
            if min_val > 500.0:
                issues.append('TOO_HIGH_FOR_UPTAKE')
                
        # Check for unitless indices (NNI)
        elif 'unitless' in units or 'NNI' in param_name:
            if max_val < 0.5 or min_val > 2.0:
                issues.append('UNREALISTIC_NNI_RANGE')
                
        return issues
    
    def check_biological_limits(self, outputs: np.ndarray, units: str, param_name: str) -> List[str]:
        """Check if model outputs are within biological limits."""
        issues = []
        
        # Check for negative values where inappropriate
        if '%' in units and np.min(outputs) < 0:
            issues.append('NEGATIVE_PERCENTAGE_VALUES')
        elif 'kg kg⁻¹' in units and np.min(outputs) < 0:
            issues.append('NEGATIVE_EFFICIENCY_VALUES')
        elif 'kg ha' in units and np.min(outputs) < 0:
            issues.append('NEGATIVE_UPTAKE_VALUES')
            
        # Check for unrealistically high values
        if '%' in units and 'leaf_N' in param_name and np.max(outputs) > 6.0:
            issues.append('UNREALISTIC_HIGH_LEAF_N')
        elif '%' in units and 'grain_protein' in param_name and np.max(outputs) > 20.0:
            issues.append('UNREALISTIC_HIGH_GRAIN_PROTEIN')
            
        return issues
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive validation report."""
        total_models = sum(len(crop_results) for crop_results in results.values())
        failed_models = sum(
            len([r for r in crop_results if not r['passed']]) 
            for crop_results in results.values()
        )
        
        report = f"""
# CALIBRATION MODEL VALIDATION REPORT

## Summary
- Total models checked: {total_models}
- Models with issues: {failed_models}
- Success rate: {((total_models - failed_models) / total_models * 100):.1f}%

## Issues by Crop
"""
        
        for crop_name, crop_results in results.items():
            failed_crop_models = [r for r in crop_results if not r['passed']]
            if failed_crop_models:
                report += f"\n### {crop_name.upper()}\n"
                for model in failed_crop_models:
                    report += f"- **{model['stage']}.{model['param']}**: {', '.join(model['issues'])}\n"
                    report += f"  - Output range: {model['output_range']}\n"
                    report += f"  - Valid range: {model['valid_range']}\n"
                    report += f"  - Clipping ratio: {model['clipping_ratio']:.2f}\n"
        
        # Add recommendations
        report += f"""
## Recommendations

### Critical Fixes Required
The following models need immediate attention:

1. **High Clipping Issues** (>90% clipping): These models produce constant values
2. **Unrealistic Ranges**: Valid ranges don't match biological reality
3. **Negative Values**: Models producing negative values where inappropriate

### Validation Framework
Implement automated validation to prevent future issues:
- Range validation during model definition
- Clipping detection and warnings
- Biological limit checks
- Literature-based range verification

### Testing Protocol
- Test all models after calibration changes
- Validate with real field data
- Check statistical variation in outputs
- Verify literature consistency

## Conclusion
{'✅ All models pass validation' if failed_models == 0 else f'❌ {failed_models} models need fixing'}
"""
        
        return report


def main():
    """Main validation routine."""
    validator = CalibrationValidator()
    
    # Run validation
    results = validator.validate_all_models()
    
    # Generate report
    report = validator.generate_report(results)
    
    # Save report
    report_path = Path('validation_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Validation report saved to: {report_path}")
    
    # Print summary
    total_models = sum(len(crop_results) for crop_results in results.values())
    failed_models = sum(
        len([r for r in crop_results if not r['passed']]) 
        for crop_results in results.values()
    )
    
    if failed_models == 0:
        print("\n🎉 All calibration models passed validation!")
    else:
        print(f"\n⚠️  {failed_models} out of {total_models} models have issues")
        print("Please review the validation report for details")


if __name__ == '__main__':
    main()
