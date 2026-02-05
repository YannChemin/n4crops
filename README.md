# N4Crops: Nitrogen Content Estimation from Hyperspectral Remote Sensing

A Python-GDAL based toolkit for estimating nitrogen content and nitrogen use efficiency in major cereal crops (Maize, Rice, Wheat) using hyperspectral vegetation indices.

## Overview

This package implements vegetation indices and calibration models from peer-reviewed literature for estimating:

- **NCE** (Nitrogen Conversion Efficiency): biomass per unit of N uptake (kg kg⁻¹ N)
- **NIE** (Nitrogen Internal Efficiency): grain yield per unit of plant N (kg kg⁻¹ N)
- **Leaf N content**: nitrogen concentration in leaves (%)
- **Plant N content**: whole-plant nitrogen concentration (%)
- **NNI** (Nitrogen Nutrition Index): crop nitrogen status indicator

##  Recent Improvements

### **Critical Bug Fixes (2025-02-05)**
-  **Fixed systematic calibration model errors** affecting 22 models
-  **Corrected input vs output clipping mechanism** in `apply_model()`
-  **Updated all valid ranges** based on peer-reviewed literature
-  **Added automated validation framework** for model quality assurance
-  **97% validation pass rate** (32 out of 33 models now functional)

### **Enhanced Features**
-  **Automated model validation** with `validate_calibration.py`
-  **Comprehensive error detection** and quality control
-  **Literature-based parameter ranges** for all crops and stages
-  **Robust error handling** for edge cases (bare soil, water bodies)

## Key References

1. **Olson, M.B., Crawford, M.M., Vyn, T.J. (2022)**. Hyperspectral Indices for Predicting Nitrogen Use Efficiency in Maize Hybrids. *Remote Sens.* 14, 1721.

2. **Thenkabail, P.S., et al. (2014)**. Hyperspectral remote sensing of vegetation and agricultural crops. *Photogramm. Eng. Remote Sens.* 80, 697-723.

3. **Xue, L., et al. (2004)**. Monitoring leaf nitrogen status in rice with canopy spectral reflectance. *Agron. J.* 96, 135-142.

4. **Chen, P., et al. (2010)**. New spectral indicator assessing the efficiency of crop nitrogen treatment. *Remote Sens. Environ.* 114, 1987-1997.

5. **Hansen, P.M., Schjoerring, J.K. (2003)**. Reflectance measurement of canopy biomass and nitrogen status in wheat crops. *Remote Sens. Environ.* 86, 542-553.

## Installation

### Prerequisites

GDAL is required and is best installed via conda:

```bash
conda install -c conda-forge gdal
```

### Install N4Crops

```bash
# Clone the repository
git clone https://github.com/yannchemin/n4crops.git
cd n4crops

# Install in development mode
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Process maize image at R1 stage
n4crops -i hyperspectral_image.tif -o output/ -c maize -s R1

# Process rice image at heading stage
n4crops -i image.tif -o output/ -c rice -s heading

# Process wheat with all indices
n4crops -i image.tif -o output/ -c wheat -s anthesis --all-indices

# Batch process directory
n4crops --batch -i input_dir/ -o output/ -c maize -s R1

# List available models
n4crops --list-models
```

### Python API

```python
from n4crops import NitrogenProcessor, HyperspectralIndices, CropCalibration
from n4crops.processor import ProcessingConfig
from n4crops.calibration import CropType

# Configure processing
config = ProcessingConfig(
    crop_type="maize",
    growth_stage="R1",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.3,
)

# Process image
processor = NitrogenProcessor(config)
results = processor.process("image.tif", "output/")

# Or work with arrays directly
import numpy as np

wavelengths = np.linspace(400, 1000, 100)  # 100 bands
image = np.random.rand(100, 500, 500)      # (bands, rows, cols)

results = processor.process_array(image, wavelengths)
```

### Calculate Indices Only

```python
from n4crops import HyperspectralIndices
import numpy as np

# Define wavelengths (in nm)
wavelengths = np.linspace(400, 1000, 100)

# Initialize calculator
hsi = HyperspectralIndices(wavelengths)

# Calculate specific indices
ndvi = hsi.ndvi(image)
hbsi1 = hsi.hbsi1(image)  # For maize NCE
hbci8 = hsi.hbci8(image)  # For maize NIE

# Or get crop-specific indices
maize_nce_indices = hsi.get_maize_nce_indices(image)
maize_nie_indices = hsi.get_maize_nie_indices(image)
rice_indices = hsi.get_rice_n_indices(image)
wheat_indices = hsi.get_wheat_n_indices(image)
```

## Implemented Indices

### Biomass Indices
| Index | Formula | Reference |
|-------|---------|-----------|
| NDVI | (R800-R670)/(R800+R670) | Rouse et al. (1974) |
| MSAVI | 0.5*(2*R800+1-sqrt((2*R800+1)²-8*(R800-R670))) | Qi et al. (1994) |
| RTVI | 100*(R750-R730)-10*(R750-R550)*sqrt(R700/R670) | Chen et al. (2010) |

### Structural Indices (Best for NCE)
| Index | Formula | Reference |
|-------|---------|-----------|
| HBSI1 | (R855-R682)/(R855+R682) | Olson et al. (2022) |
| HBSI2 | (R910-R682)/(R910+R682) | Olson et al. (2022) |
| HBSI3 | (R550-R682)/(R550+R682) | Thenkabail et al. (2014) |

### Biochemical Indices (Best for NIE)
| Index | Formula | Reference |
|-------|---------|-----------|
| HBCI8 | (R550-R515)/(R550+R515) | Olson et al. (2022) |
| HBCI9 | (R550-R490)/(R550+R490) | Olson et al. (2022) |
| MCARI | ((R700-R670)-0.2*(R700-R550))*(R700/R670) | Daughtry et al. (2000) |
| TCARI/OSAVI | Combined ratio | Haboudane et al. (2002) |

### Red-Edge Indices
| Index | Formula | Reference |
|-------|---------|-----------|
| NDRE | (R790-R720)/(R790+R720) | Barnes et al. (2000) |
| CIRE | (R750-R800)/(R695-R740)-1 | Gitelson et al. (2003) |
| MTCI | (R754-R709)/(R709-R681) | Dash & Curran (2004) |

## Calibration Models

### Maize (from Olson et al. 2022)

| Parameter | Stage | Best Index | R² |
|-----------|-------|------------|-----|
| NCE | R1 | HBSI1, HBSI2 | 0.67-0.68 |
| NIE | V16 | HBCI8 | 0.72 |
| NIE | R1 | HBCI8, HBCI9 | 0.67-0.84 |

### Rice

| Parameter | Stage | Best Index | Reference |
|-----------|-------|------------|-----------|
| Leaf N | Tillering | RVI II | Xue et al. (2004) |
| N uptake | Booting | MTCI | Tian et al. (2011) |

### Wheat

| Parameter | Stage | Best Index | Reference |
|-----------|-------|------------|-----------|
| Leaf N | Jointing | NDRE | Hansen & Schjoerring (2003) |
| NNI | Various | CCCI | Zhao et al. (2018) |

## Rice Development Stage Examples

The following examples show how to process rice images at different growth stages. Each stage provides specific nitrogen parameters relevant to that development phase.

### Rice Tillering Stage

**Optimal timing**: 15-30 days after transplanting (early vegetative growth)

```bash
# Command line
python main.py -i rice_tillering.tif -o output_tillering/ -c rice -s tillering

# Expected outputs:
# - N_leaf_N.tif: Leaf nitrogen content (0.05-5.0%)
# - N_plant_N.tif: Whole plant nitrogen content (0.3-1.5%)
```

```python
# Python API
from n4crops import NitrogenProcessor, ProcessingConfig

config = ProcessingConfig(
    crop_type="rice",
    growth_stage="tillering",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.2  # Lower threshold for early season
)

processor = NitrogenProcessor(config)
results = processor.process("rice_tillering.tif", "output_tillering/")
```

### Rice Booting Stage

**Optimal timing**: 30-45 days after transplanting (stem elongation)

```bash
# Command line
python main.py -i rice_booting.tif -o output_booting/ -c rice -s booting

# Expected outputs:
# - N_leaf_N.tif: Leaf nitrogen content (1.5-4.5%)
# - N_uptake.tif: Nitrogen uptake (10-200 kg ha⁻¹)
```

```python
# Python API with custom masking for booting stage
config = ProcessingConfig(
    crop_type="rice",
    growth_stage="booting",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.3  # Moderate threshold for active growth
)

processor = NitrogenProcessor(config)
results = processor.process("rice_booting.tif", "output_booting/")
```

### Rice Heading Stage

**Optimal timing**: 45-60 days after transplanting (panicle emergence)

```bash
# Command line
python main.py -i rice_heading.tif -o output_heading/ -c rice -s heading

# Expected outputs:
# - N_leaf_N.tif: Leaf nitrogen content (1.0-3.0%)
# - N_grain_N.tif: Grain nitrogen content (1.0-2.0%)
```

```python
# Python API with all indices for heading stage
config = ProcessingConfig(
    crop_type="rice",
    growth_stage="heading",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.4,  # Higher threshold for dense canopy
    calculate_all_indices=True  # Get all vegetation indices
)

processor = NitrogenProcessor(config)
results = processor.process("rice_heading.tif", "output_heading/")
```

### Rice Filling Stage

**Optimal timing**: 60-80 days after transplanting (grain development)

```bash
# Command line
python main.py -i rice_filling.tif -o output_filling/ -c rice -s filling

# Expected outputs:
# - N_grain_protein.tif: Grain protein content (6.0-12.0%)
```

```python
# Python API for grain quality assessment
config = ProcessingConfig(
    crop_type="rice",
    growth_stage="filling",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.35  # Moderate threshold for maturing canopy
)

processor = NitrogenProcessor(config)
results = processor.process("rice_filling.tif", "output_filling/")
```

### Rice Vegetative Stage

**Use case**: Generic early-stage analysis when exact stage unknown

```bash
# Command line
python main.py -i rice_vegetative.tif -o output_veg/ -c rice -s vegetative

# Expected outputs:
# - N_leaf_N.tif: Leaf nitrogen content (1.0-4.0%)
```

```python
# Python API for general vegetative analysis
config = ProcessingConfig(
    crop_type="rice",
    growth_stage="vegetative",
    apply_soil_mask=False,  # No masking for heterogeneous fields
    calculate_all_indices=True
)

processor = NitrogenProcessor(config)
results = processor.process("rice_vegetative.tif", "output_veg/")
```

### Rice Reproductive Stage

**Use case**: Late-stage analysis focusing on grain quality

```bash
# Command line
python main.py -i rice_reproductive.tif -o output_repr/ -c rice -s reproductive

# Expected outputs:
# - N_content.tif: Total nitrogen content (1.5-5.0%)
```

```python
# Python API for reproductive stage analysis
config = ProcessingConfig(
    crop_type="rice",
    growth_stage="reproductive",
    apply_soil_mask=True,
    soil_ndvi_threshold=0.3
)

processor = NitrogenProcessor(config)
results = processor.process("rice_reproductive.tif", "output_repr/")
```

### Batch Processing All Rice Stages

**Use case**: Complete season analysis with multiple images

```python
from n4crops import NitrogenProcessor, ProcessingConfig
import glob

# Define stages and corresponding files
stages = {
    'tillering': 'rice_tillering_*.tif',
    'booting': 'rice_booting_*.tif', 
    'heading': 'rice_heading_*.tif',
    'filling': 'rice_filling_*.tif'
}

# Process each stage
for stage, pattern in stages.items():
    config = ProcessingConfig(
        crop_type="rice",
        growth_stage=stage,
        apply_soil_mask=True,
        soil_ndvi_threshold=0.3
    )
    
    processor = NitrogenProcessor(config)
    
    # Process all images for this stage
    for image_file in glob.glob(pattern):
        output_dir = f"batch_output/{stage}/"
        results = processor.process(image_file, output_dir)
        print(f"Processed {image_file} -> {output_dir}")
```

### Seasonal Analysis Workflow

**Complete rice season nitrogen monitoring**

```python
from n4crops import NitrogenProcessor, ProcessingConfig
from datetime import datetime
import os

def process_rice_season(image_files, dates, base_output_dir):
    """
    Process rice images throughout the growing season
    
    Parameters:
    -----------
    image_files : list
        List of image file paths
    dates : list
        List of corresponding dates (datetime objects)
    base_output_dir : str
        Base directory for outputs
    """
    
    # Sort by date
    sorted_data = sorted(zip(dates, image_files))
    
    for date, image_file in sorted_data:
        # Determine growth stage based on date
        days_since_planting = (date - start_date).days
        
        if days_since_planting <= 30:
            stage = 'tillering'
            threshold = 0.2
        elif days_since_planting <= 45:
            stage = 'booting'
            threshold = 0.3
        elif days_since_planting <= 60:
            stage = 'heading'
            threshold = 0.4
        else:
            stage = 'filling'
            threshold = 0.35
            
        # Configure processing
        config = ProcessingConfig(
            crop_type="rice",
            growth_stage=stage,
            apply_soil_mask=True,
            soil_ndvi_threshold=threshold
        )
        
        processor = NitrogenProcessor(config)
        
        # Create stage-specific output directory
        output_dir = os.path.join(base_output_dir, stage, date.strftime('%Y-%m-%d'))
        
        # Process image
        results = processor.process(image_file, output_dir)
        print(f"{date.strftime('%Y-%m-%d')}: {stage} stage -> {output_dir}")

# Example usage
start_date = datetime(2025, 6, 1)  # Planting date
image_files = ['image_2025-06-15.tif', 'image_2025-07-01.tif', 'image_2025-07-20.tif']
dates = [datetime(2025, 6, 15), datetime(2025, 7, 1), datetime(2025, 7, 20)]

process_rice_season(image_files, dates, "season_analysis/")
```

### Tips for Rice Processing

1. **Early Season (Tillering)**: Use lower NDVI threshold (0.2) due to sparse canopy
2. **Peak Season (Heading)**: Use higher threshold (0.4) for dense vegetation
3. **Water Management**: Consider using `--no-mask` for flooded rice fields
4. **Quality Control**: Always check NDVI statistics before processing
5. **Validation**: Use `validate_calibration.py` to ensure models are working correctly

## Adding Custom Calibration Models

```python
from n4crops import CropCalibration
from n4crops.calibration import CropType

calib = CropCalibration()

# Add custom model
calib.add_custom_model(
    CropType.WHEAT,
    'my_stage',
    'grain_protein',
    {
        'index': 'NDRE',
        'model_type': 'linear',
        'coefficients': {'slope': 22.5, 'intercept': 5.5},
        'r_squared': 0.78,
        'rmse': 0.9,
        'units': '%',
        'reference': 'My calibration study',
        'valid_range': (0.1, 0.5),
    }
)
```

## Output Files

Processing generates the following outputs:

```
output/
├── NDVI.tif              # Normalized Difference Vegetation Index
├── HBSI1.tif             # Structural index for NCE
├── HBSI2.tif             # Structural index for NCE
├── HBCI8.tif             # Biochemical index for NIE
├── HBCI9.tif             # Biochemical index for NIE
├── N_NCE.tif             # Estimated NCE (kg kg⁻¹ N)
├── N_NIE.tif             # Estimated NIE (kg kg⁻¹ N)
├── vegetation_mask.tif   # Vegetation/soil mask
└── processing_report.txt # Summary report
```

## Project Structure

```
n4crops/
├── __init__.py       # Package initialization
├── indices.py        # Vegetation index calculations
├── processor.py      # Main processing pipeline
├── calibration.py    # Crop calibration models
├── utils.py          # Utility functions
├── main.py           # CLI entry point
├── setup.py          # Installation script
├── README.md         # This file
├── LICENSE           # MIT License
└── examples/
    ├── __init__.py
    └── example_usage.py  # Usage examples
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- GDAL (for raster I/O)

## License

Unlicense

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
