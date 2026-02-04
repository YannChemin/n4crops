#!/usr/bin/env python3
"""
N4Crops Example Usage
=====================

This script demonstrates how to use the N4Crops package for nitrogen
estimation in maize, rice, and wheat using hyperspectral imagery.

Examples include:
1. Processing a hyperspectral image for maize NCE/NIE estimation
2. Processing rice imagery for nitrogen content
3. Processing wheat imagery with custom calibration
4. Working with numpy arrays directly
5. Batch processing multiple images
"""

import numpy as np

from n4crops import (
    HyperspectralIndices,
    NitrogenProcessor,
    CropCalibration,
    read_hyperspectral,
    write_raster
)
from n4crops.processor import ProcessingConfig
from n4crops.calibration import CropType, create_calibration_report


def example_1_maize_processing():
    """
    Example 1: Process hyperspectral image for maize nitrogen estimation.

    Based on Olson et al. (2022) methodology:
    - HBSI1 and HBSI2 for NCE (Nitrogen Conversion Efficiency)
    - HBCI8 and HBCI9 for NIE (Nitrogen Internal Efficiency)
    """
    print("\n" + "=" * 60)
    print("Example 1: Maize Nitrogen Processing")
    print("=" * 60)

    # Configure processing for maize at R1 (reproductive) stage
    config = ProcessingConfig(
        crop_type="maize",
        growth_stage="R1",
        apply_soil_mask=True,
        soil_ndvi_threshold=0.3,
        nodata_value=-9999.0,
        calculate_all_indices=False,  # Only calculate relevant indices
    )

    # Create processor
    processor = NitrogenProcessor(config)

    # Process image (replace with your actual image path)
    input_path = "path/to/hyperspectral_image.tif"
    output_dir = "output/maize_results"

    # If you have an actual image:
    # results = processor.process(input_path, output_dir)

    # For demonstration, create synthetic data
    print("\nCreating synthetic hyperspectral data for demonstration...")

    # Simulate hyperspectral data (100 bands from 400-1000nm)
    wavelengths = np.linspace(400, 1000, 100)
    rows, cols = 100, 100

    # Create synthetic reflectance data
    np.random.seed(42)
    image = np.zeros((100, rows, cols), dtype=np.float32)

    for i, wl in enumerate(wavelengths):
        if wl < 700:  # Visible - lower reflectance for vegetation
            image[i] = 0.05 + 0.03 * np.random.random((rows, cols))
        elif wl < 750:  # Red edge
            image[i] = 0.2 + 0.2 * np.random.random((rows, cols))
        else:  # NIR - high reflectance
            image[i] = 0.4 + 0.1 * np.random.random((rows, cols))

    # Add some spatial variation (simulating N stress gradient)
    stress_gradient = np.linspace(0.8, 1.2, cols)
    for i in range(len(wavelengths)):
        image[i] *= stress_gradient

    # Process the array directly
    results = processor.process_array(image, wavelengths)

    print(f"\nCalculated indices:")
    for key, value in results.items():
        if value is not None and isinstance(value, np.ndarray):
            valid = value[np.isfinite(value)]
            if len(valid) > 0:
                print(f"  {key}: mean={valid.mean():.4f}, std={valid.std():.4f}")

    return results


def example_2_rice_processing():
    """
    Example 2: Process hyperspectral image for rice nitrogen estimation.

    Uses indices optimized for rice:
    - RVI II (Xue et al. 2004)
    - Rice NRI (Tian et al. 2011)
    - GNDVI, NDRE, MTCI
    """
    print("\n" + "=" * 60)
    print("Example 2: Rice Nitrogen Processing")
    print("=" * 60)

    # Create synthetic data
    wavelengths = np.linspace(400, 1000, 120)
    rows, cols = 80, 80

    np.random.seed(123)
    image = np.zeros((120, rows, cols), dtype=np.float32)

    # Simulate rice canopy reflectance
    for i, wl in enumerate(wavelengths):
        if wl < 550:  # Blue-green
            image[i] = 0.04 + 0.02 * np.random.random((rows, cols))
        elif wl < 700:  # Green-red
            image[i] = 0.08 + 0.04 * np.random.random((rows, cols))
        elif wl < 750:  # Red edge
            image[i] = 0.25 + 0.15 * np.random.random((rows, cols))
        else:  # NIR
            image[i] = 0.45 + 0.08 * np.random.random((rows, cols))

    # Initialize indices calculator
    hsi = HyperspectralIndices(wavelengths)

    # Calculate rice-specific indices
    rice_indices = hsi.get_rice_n_indices(image)

    print("\nRice nitrogen indices:")
    for name, values in rice_indices.items():
        if values is not None:
            valid = values[np.isfinite(values)]
            print(f"  {name}: mean={valid.mean():.4f}, range=[{valid.min():.4f}, {valid.max():.4f}]")

    # Apply calibration models
    calib = CropCalibration()
    models = calib.get_models(CropType.RICE, 'booting')

    print("\nEstimated nitrogen parameters:")
    for param_name, model_info in models.items():
        index_name = model_info['index']
        if index_name in rice_indices and rice_indices[index_name] is not None:
            n_values = calib.apply_model(rice_indices[index_name], model_info)
            valid = n_values[np.isfinite(n_values)]
            print(f"  {param_name} ({model_info['units']}): "
                  f"mean={valid.mean():.2f}, std={valid.std():.2f}")

    return rice_indices


def example_3_wheat_with_custom_calibration():
    """
    Example 3: Process wheat imagery with custom calibration model.

    Demonstrates how to add and use custom calibration models.
    """
    print("\n" + "=" * 60)
    print("Example 3: Wheat with Custom Calibration")
    print("=" * 60)

    # Create synthetic wheat data
    wavelengths = np.linspace(400, 1000, 100)
    rows, cols = 60, 60

    np.random.seed(456)
    image = np.zeros((100, rows, cols), dtype=np.float32)

    for i, wl in enumerate(wavelengths):
        if wl < 700:
            image[i] = 0.06 + 0.03 * np.random.random((rows, cols))
        elif wl < 750:
            image[i] = 0.22 + 0.18 * np.random.random((rows, cols))
        else:
            image[i] = 0.42 + 0.12 * np.random.random((rows, cols))

    # Initialize
    hsi = HyperspectralIndices(wavelengths)
    calib = CropCalibration()

    # Add a custom calibration model
    calib.add_custom_model(
        CropType.WHEAT,
        'custom_stage',
        'grain_protein',
        {
            'index': 'NDRE',
            'model_type': 'linear',
            'coefficients': {'slope': 22.5, 'intercept': 5.5},
            'r_squared': 0.78,
            'rmse': 0.9,
            'units': '%',
            'reference': 'Custom calibration from local trials',
            'valid_range': (0.1, 0.5),
        }
    )

    # Calculate indices
    wheat_indices = hsi.get_wheat_n_indices(image)

    print("\nWheat nitrogen indices:")
    for name, values in wheat_indices.items():
        if values is not None:
            valid = values[np.isfinite(values)]
            print(f"  {name}: mean={valid.mean():.4f}")

    # Apply custom model
    custom_models = calib.get_models(CropType.WHEAT, 'custom_stage')

    print("\nCustom grain protein estimation:")
    for param_name, model_info in custom_models.items():
        index_name = model_info['index']
        if index_name in wheat_indices:
            n_values = calib.apply_model(wheat_indices[index_name], model_info)
            valid = n_values[np.isfinite(n_values)]
            print(f"  {param_name}: mean={valid.mean():.2f}%, "
                  f"std={valid.std():.2f}%")

    # Print model info
    print("\n" + create_calibration_report(calib, CropType.WHEAT, 'custom_stage'))

    return wheat_indices


def example_4_indices_only():
    """
    Example 4: Calculate vegetation indices without N estimation.

    Useful when you want to explore the spectral indices first
    before applying calibration models.
    """
    print("\n" + "=" * 60)
    print("Example 4: Calculate All Vegetation Indices")
    print("=" * 60)

    # Create sample data
    wavelengths = np.linspace(400, 1000, 150)
    rows, cols = 50, 50

    np.random.seed(789)
    image = np.zeros((150, rows, cols), dtype=np.float32)

    # Simple vegetation spectral signature
    for i, wl in enumerate(wavelengths):
        if wl < 550:
            image[i] = 0.03 + 0.015 * np.random.random((rows, cols))
        elif wl < 700:
            image[i] = 0.07 + 0.025 * np.random.random((rows, cols))
        elif wl < 750:
            image[i] = 0.20 + 0.20 * np.random.random((rows, cols))
        else:
            image[i] = 0.45 + 0.10 * np.random.random((rows, cols))

    # Calculate all indices
    hsi = HyperspectralIndices(wavelengths)
    all_indices = hsi.calculate_all_indices(image)

    print(f"\nCalculated {len([v for v in all_indices.values() if v is not None])} indices:")
    print("-" * 40)

    # Group by category
    biomass_indices = ['NDVI', 'MSAVI', 'RTVI']
    structural_indices = ['HBSI1', 'HBSI2', 'HBSI3', 'PSRI']
    biochemical_indices = ['HBCI8', 'HBCI9', 'HBCI10', 'MCARI', 'TCARI',
                          'OSAVI', 'TCARI_OSAVI', 'DCNI', 'RVI_II']
    rededge_indices = ['NDRE', 'CIRE', 'HREI15', 'HREI16', 'REIP']

    categories = [
        ('Biomass', biomass_indices),
        ('Structural', structural_indices),
        ('Biochemical', biochemical_indices),
        ('Red-edge', rededge_indices),
    ]

    for cat_name, indices in categories:
        print(f"\n{cat_name} Indices:")
        for idx_name in indices:
            if idx_name in all_indices and all_indices[idx_name] is not None:
                valid = all_indices[idx_name][np.isfinite(all_indices[idx_name])]
                if len(valid) > 0:
                    print(f"  {idx_name:15s}: mean={valid.mean():8.4f}, "
                          f"std={valid.std():8.4f}")

    return all_indices


def example_5_compare_crops():
    """
    Example 5: Compare nitrogen indices across different crops.

    Shows how different indices respond to the same spectral data
    when interpreted for different crops.
    """
    print("\n" + "=" * 60)
    print("Example 5: Compare Nitrogen Indices Across Crops")
    print("=" * 60)

    # Create common spectral data
    wavelengths = np.linspace(400, 1000, 100)
    rows, cols = 40, 40

    np.random.seed(101)
    image = np.zeros((100, rows, cols), dtype=np.float32)

    for i, wl in enumerate(wavelengths):
        if wl < 700:
            image[i] = 0.06 + 0.03 * np.random.random((rows, cols))
        elif wl < 750:
            image[i] = 0.25 + 0.15 * np.random.random((rows, cols))
        else:
            image[i] = 0.45 + 0.10 * np.random.random((rows, cols))

    hsi = HyperspectralIndices(wavelengths)
    calib = CropCalibration()

    # Get crop-specific indices
    maize_indices = hsi.get_maize_nie_indices(image)
    rice_indices = hsi.get_rice_n_indices(image)
    wheat_indices = hsi.get_wheat_n_indices(image)

    print("\nCrop-Specific Index Comparison:")
    print("-" * 60)

    # Common indices across crops
    common_indices = ['GNDVI', 'NDRE', 'MTCI']

    print("\nCommon indices (same spectral data):")
    for idx_name in common_indices:
        values = None
        for crop_indices in [maize_indices, rice_indices, wheat_indices]:
            if idx_name in crop_indices and crop_indices[idx_name] is not None:
                values = crop_indices[idx_name]
                break

        if values is not None:
            valid = values[np.isfinite(values)]
            print(f"  {idx_name}: mean={valid.mean():.4f}")

    # Estimated N content for each crop
    print("\nEstimated Leaf N Content by Crop (same spectral data):")

    for crop_type, stage, indices in [
        (CropType.MAIZE, 'vegetative', maize_indices),
        (CropType.RICE, 'booting', rice_indices),
        (CropType.WHEAT, 'jointing', wheat_indices),
    ]:
        try:
            models = calib.get_models(crop_type, stage)
            for param_name, model_info in models.items():
                if 'leaf_N' in param_name.lower() or 'leaf_n' in param_name.lower():
                    index_name = model_info['index']
                    # Find the index in any of the dictionaries
                    index_values = None
                    if index_name in indices:
                        index_values = indices[index_name]
                    elif hasattr(hsi, index_name.lower()):
                        index_values = getattr(hsi, index_name.lower())(image)

                    if index_values is not None:
                        n_values = calib.apply_model(index_values, model_info)
                        valid = n_values[np.isfinite(n_values)]
                        print(f"  {crop_type.value.title()} ({stage}): "
                              f"{valid.mean():.2f} ± {valid.std():.2f} %")
                    break
        except ValueError as e:
            print(f"  {crop_type.value.title()}: No model available for {stage}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# N4Crops Example Usage")
    print("# Nitrogen Estimation from Hyperspectral Imagery")
    print("#" * 60)

    # Run examples
    example_1_maize_processing()
    example_2_rice_processing()
    example_3_wheat_with_custom_calibration()
    example_4_indices_only()
    example_5_compare_crops()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    # List available models
    print("\nAvailable Calibration Models:")
    print("-" * 40)
    calib = CropCalibration()
    available = calib.list_available_models()

    for crop, stages in available.items():
        print(f"\n{crop.upper()}:")
        for stage, params in stages.items():
            print(f"  {stage}: {', '.join(params)}")


if __name__ == '__main__':
    main()
