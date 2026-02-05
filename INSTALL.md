# Installation Guide

## System Requirements

### Python Version
- **Python 3.8 or higher** (recommended: Python 3.9+)

### Operating Systems
- **Linux** (Ubuntu 18.04+, CentOS 7+, Debian 10+)
- **macOS** (10.14+)
- **Windows** (10+ with WSL2 recommended)

## Dependencies

### Core Dependencies
- **GDAL** (Geospatial Data Abstraction Library) - 3.0+
- **NumPy** - 1.19+
- **SciPy** - 1.6+

### Optional Dependencies
- **Matplotlib** - for visualization and plotting
- **Jupyter** - for interactive analysis
- **pytest** - for running tests

## Installation Methods

### Method 1: Conda (Recommended)

Conda is the preferred method as it handles GDAL installation automatically.

```bash
# Create a new conda environment
conda create -n n4crops python=3.9
conda activate n4crops

# Install GDAL and other dependencies
conda install -c conda-forge gdal numpy scipy

# Clone and install N4Crops
git clone https://github.com/yannchemin/n4crops.git
cd n4crops
pip install -e .
```

### Method 2: pip with System GDAL

If you prefer to use pip and have GDAL installed system-wide:

#### Ubuntu/Debian
```bash
# Install system GDAL
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev

# Set environment variables
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Install Python packages
pip install GDAL==$(gdal-config --version)
pip install numpy scipy

# Install N4Crops
git clone https://github.com/yannchemin/n4crops.git
cd n4crops
pip install -e .
```

#### CentOS/RHEL
```bash
# Install system GDAL
sudo yum install gdal gdal-devel

# Set environment variables
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Install Python packages
pip install GDAL==$(gdal-config --version)
pip install numpy scipy

# Install N4Crops
git clone https://github.com/yannchemin/n4crops.git
cd n4crops
pip install -e .
```

#### macOS
```bash
# Install GDAL via Homebrew
brew install gdal

# Install Python packages
pip install GDAL numpy scipy

# Install N4Crops
git clone https://github.com/yannchemin/n4crops.git
cd n4crops
pip install -e .
```

### Method 3: Development Installation

For developers who want to contribute to the project:

```bash
# Clone the repository
git clone https://github.com/yannchemin/n4crops.git
cd n4crops

# Create conda environment
conda create -n n4crops-dev python=3.9
conda activate n4crops-dev

# Install dependencies
conda install -c conda-forge gdal numpy scipy pytest matplotlib jupyter

# Install in development mode with test dependencies
pip install -e ".[dev]"
```

## Verification

### Test Installation

```bash
# Test basic import
python -c "import n4crops; print('N4Crops installed successfully!')"

# Test calibration models
python validate_calibration.py

# Test command line interface
python main.py --help
```

### Expected Output

The validation script should show:
```
🔍 CALIBRATION MODEL VALIDATION
✅ All models passed validation!
🎉 All calibration models passed validation!
```

## Troubleshooting

### Common Issues

#### 1. GDAL Installation Errors

**Symptom**: `ERROR: Could not find a version that satisfies the requirement GDAL`

**Solution**: Use conda instead of pip for GDAL:
```bash
conda install -c conda-forge gdal
```

#### 2. Import Errors

**Symptom**: `ImportError: libgdal.so.20: cannot open shared object file`

**Solution**: Set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/path/to/gdal/lib:$LD_LIBRARY_PATH
```

#### 3. Wavelength Extraction Warnings

**Symptom**: `Warning: Could not extract wavelengths from metadata`

**Solution**: This is normal for some image formats. The system will use default wavelength ranges.

#### 4. Permission Errors

**Symptom**: `Permission denied` when installing

**Solution**: Use user installation or virtual environment:
```bash
pip install --user -e .
# or
python -m pip install --user -e .
```

### Platform-Specific Notes

#### Windows
- **WSL2 is strongly recommended** over native Windows
- GDAL installation on native Windows can be challenging
- Use conda for the smoothest experience

#### macOS
- **Apple Silicon (M1/M2)**: Use conda with miniforge
- **Intel Macs**: Either conda or Homebrew works well

#### Linux
- **Ubuntu/Debian**: Use conda or system package manager
- **CentOS/RHEL**: Enable EPEL repository for GDAL packages

## Configuration

### Environment Variables

Optional environment variables for customization:

```bash
# GDAL data files (optional)
export GDAL_DATA=/path/to/gdal/data

# Python path (if needed)
export PYTHONPATH=/path/to/n4crops:$PYTHONPATH
```

### Default Settings

The system uses sensible defaults:
- **Wavelength range**: 400-1000nm (if not in metadata)
- **NDVI threshold**: 0.3 for soil masking
- **NoData value**: -9999
- **Output format**: GeoTIFF with LZW compression

## Performance Optimization

### Memory Usage
For large hyperspectral images (>1GB), consider:
- Increasing system RAM
- Using solid-state storage
- Processing in smaller tiles

### Processing Speed
- Use SSD storage for I/O operations
- Ensure sufficient RAM for image size
- Consider parallel processing for batch operations

## Support

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the README.md and inline help
- **Examples**: See the command-line help with `--help`

### Contributing
- Fork the repository
- Create a feature branch
- Submit a pull request
- Ensure tests pass with `python validate_calibration.py`

## Version History

### v2.0.0 (2025-02-05)
- **Critical bug fixes** for calibration models
- **Automated validation framework**
- **97% model validation pass rate**
- **Enhanced error handling**

### v1.0.0
- Initial release
- Basic nitrogen estimation functionality
- Support for maize, rice, wheat

---

**Installation complete!** 🎉

You're ready to start using N4Crops for hyperspectral nitrogen estimation.
