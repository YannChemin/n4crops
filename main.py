#!/usr/bin/env python3
"""
N4Crops: Main Processing Script
================================

Command-line interface for nitrogen estimation from hyperspectral imagery.

Usage:
    python -m n4crops.main --input image.tif --output results/ --crop maize --stage R1
    python -m n4crops.main --input image.tif --output results/ --crop rice --stage heading
    python -m n4crops.main --input image.tif --output results/ --crop wheat --stage anthesis

For help:
    python -m n4crops.main --help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from .processor import NitrogenProcessor, ProcessingConfig, BatchProcessor
    from .calibration import CropCalibration, CropType, create_calibration_report
    from .utils import generate_report
except ImportError:
    # Allow running as script
    from processor import NitrogenProcessor, ProcessingConfig, BatchProcessor
    from calibration import CropCalibration, CropType, create_calibration_report
    from utils import generate_report


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='N4Crops: Nitrogen estimation from hyperspectral imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single maize image at R1 stage
    python -m n4crops.main -i image.tif -o output/ -c maize -s R1

    # Process rice image at heading stage
    python -m n4crops.main -i image.tif -o output/ -c rice -s heading

    # Process wheat image with all indices
    python -m n4crops.main -i image.tif -o output/ -c wheat -s anthesis --all-indices

    # Batch process directory
    python -m n4crops.main --batch -i input_dir/ -o output/ -c maize -s R1

    # List available models
    python -m n4crops.main --list-models

Supported crops: maize, rice, wheat
        """
    )

    # Input/Output arguments
    parser.add_argument('-i', '--input', type=str,
                        help='Input hyperspectral image or directory (for batch)')
    parser.add_argument('-o', '--output', type=str, default='./output',
                        help='Output directory (default: ./output)')

    # Crop and growth stage
    parser.add_argument('-c', '--crop', type=str, default='maize',
                        choices=['maize', 'rice', 'wheat'],
                        help='Crop type (default: maize)')
    parser.add_argument('-s', '--stage', type=str, default='R1',
                        help='Growth stage (default: R1)')

    # Processing options
    parser.add_argument('--all-indices', action='store_true',
                        help='Calculate all available indices')
    parser.add_argument('--no-mask', action='store_true',
                        help='Disable soil/background masking')
    parser.add_argument('--ndvi-threshold', type=float, default=0.3,
                        help='NDVI threshold for vegetation mask (default: 0.3)')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                        help='NoData value for outputs (default: -9999.0)')

    # Batch processing
    parser.add_argument('--batch', action='store_true',
                        help='Enable batch processing mode')
    parser.add_argument('--pattern', type=str, default='*.tif',
                        help='File pattern for batch processing (default: *.tif)')

    # Utility options
    parser.add_argument('--list-models', action='store_true',
                        help='List available calibration models and exit')
    parser.add_argument('--model-info', type=str, metavar='PARAM',
                        help='Show detailed info for a specific parameter')
    parser.add_argument('--wavelengths', type=str,
                        help='Path to wavelength file (JSON format)')

    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output messages')

    return parser.parse_args()


def list_models():
    """Print available calibration models."""
    calib = CropCalibration()
    available = calib.list_available_models()

    print("\nAvailable Calibration Models")
    print("=" * 60)

    for crop, stages in available.items():
        print(f"\n{crop.upper()}")
        print("-" * 40)
        for stage, params in stages.items():
            print(f"  {stage}:")
            for param in params:
                print(f"    - {param}")

    print("\n")


def show_model_info(crop: str, stage: str, param: str):
    """Show detailed information for a specific model."""
    calib = CropCalibration()

    try:
        crop_type = CropType[crop.upper()]
    except KeyError:
        print(f"Unknown crop type: {crop}")
        return

    model_info = calib.get_model_info(crop_type, stage, param)

    if model_info is None:
        print(f"No model found for {crop}/{stage}/{param}")
        print("\nAvailable models:")
        list_models()
        return

    print(f"\nModel: {crop.upper()} / {stage} / {param}")
    print("=" * 50)
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()


def process_single(args):
    """Process a single image."""
    if not args.input:
        print("Error: Input file required. Use -i or --input")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Create configuration
    config = ProcessingConfig(
        crop_type=args.crop,
        growth_stage=args.stage,
        apply_soil_mask=not args.no_mask,
        soil_ndvi_threshold=args.ndvi_threshold,
        nodata_value=args.nodata,
        calculate_all_indices=args.all_indices,
    )

    # Create processor and run
    processor = NitrogenProcessor(config)

    if not args.quiet:
        print(f"\nN4Crops Processing")
        print("=" * 50)
        print(f"Input: {input_path}")
        print(f"Output: {args.output}")
        print(f"Crop: {args.crop}")
        print(f"Growth stage: {args.stage}")
        print(f"Soil masking: {not args.no_mask}")
        print()

    try:
        output_files = processor.process(input_path, args.output)

        if not args.quiet:
            print("\nOutput files:")
            for name, path in output_files.items():
                print(f"  {name}: {path}")

        # Generate report
        report_path = Path(args.output) / "processing_report.txt"
        generate_report(
            {
                'input': str(input_path),
                'crop': args.crop,
                'stage': args.stage,
                'outputs': {k: str(v) for k, v in output_files.items()},
            },
            report_path
        )

        if not args.quiet:
            print(f"\nProcessing complete. Report saved to: {report_path}")

    except Exception as e:
        print(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def process_batch(args):
    """Process multiple images in batch mode."""
    if not args.input:
        print("Error: Input directory required. Use -i or --input")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Input must be a directory for batch processing: {input_dir}")
        sys.exit(1)

    # Create configuration
    config = ProcessingConfig(
        crop_type=args.crop,
        growth_stage=args.stage,
        apply_soil_mask=not args.no_mask,
        soil_ndvi_threshold=args.ndvi_threshold,
        nodata_value=args.nodata,
        calculate_all_indices=args.all_indices,
    )

    # Create batch processor
    batch = BatchProcessor(config)

    if not args.quiet:
        print(f"\nN4Crops Batch Processing")
        print("=" * 50)
        print(f"Input directory: {input_dir}")
        print(f"File pattern: {args.pattern}")
        print(f"Output directory: {args.output}")
        print(f"Crop: {args.crop}")
        print(f"Growth stage: {args.stage}")
        print()

    try:
        results = batch.process_directory(input_dir, args.output, args.pattern)

        if not args.quiet:
            print("\nBatch processing complete.")
            print(f"Processed {len(results)} files.")

            # Summary
            successful = sum(1 for r in results.values() if 'error' not in r)
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle utility commands
    if args.list_models:
        list_models()
        return

    if args.model_info:
        show_model_info(args.crop, args.stage, args.model_info)
        return

    # Check GDAL availability
    try:
        from osgeo import gdal
    except ImportError:
        print("Error: GDAL is required but not installed.")
        print("Install with: conda install -c conda-forge gdal")
        sys.exit(1)

    # Run processing
    if args.batch:
        process_batch(args)
    else:
        process_single(args)


if __name__ == '__main__':
    main()
