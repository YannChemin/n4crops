# N4Crops: Nitrogen Content Estimation from Hyperspectral Remote Sensing

A Python-GDAL based toolkit for estimating nitrogen content and nitrogen use efficiency in major cereal crops (Maize, Rice, Wheat) using hyperspectral vegetation indices.

## Overview

This package implements vegetation indices and calibration models from peer-reviewed literature for estimating:

- **NCE** (Nitrogen Conversion Efficiency): biomass per unit of N uptake (kg kg⁻¹ N)
- **NIE** (Nitrogen Internal Efficiency): grain yield per unit of plant N (kg kg⁻¹ N)
- **Leaf N content**: nitrogen concentration in leaves (%)
- **Plant N content**: whole-plant nitrogen concentration (%)
- **NNI** (Nitrogen Nutrition Index): crop nitrogen status indicator

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
git clone https://github.com/yourusername/n4crops.git
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

MIT License

## Citation

If you use this software in your research, please cite:

```bibtex
@article{olson2022hyperspectral,
  title={Hyperspectral Indices for Predicting Nitrogen Use Efficiency in Maize Hybrids},
  author={Olson, Monica B and Crawford, Melba M and Vyn, Tony J},
  journal={Remote Sensing},
  volume={14},
  number={7},
  pages={1721},
  year={2022},
  publisher={MDPI}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
