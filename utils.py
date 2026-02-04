"""
Utility Functions for N4Crops
=============================

This module provides utility functions for:
- Reading hyperspectral imagery with GDAL
- Writing raster outputs
- Wavelength extraction and interpolation
- Data quality checks
- Statistical summaries
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import json

try:
    from osgeo import gdal, osr, ogr
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    warnings.warn("GDAL not available. Some functionality will be limited.")


def read_hyperspectral(filepath: Union[str, Path],
                       return_wavelengths: bool = True,
                       bands: Optional[List[int]] = None) -> Union[
                           np.ndarray, Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Read hyperspectral image using GDAL.

    Parameters
    ----------
    filepath : str or Path
        Path to the hyperspectral image
    return_wavelengths : bool, optional
        If True, also return wavelength array and metadata (default: True)
    bands : List[int], optional
        Specific band indices to read (0-indexed). If None, read all bands.

    Returns
    -------
    np.ndarray or Tuple
        If return_wavelengths is False: image array (bands, rows, cols)
        If return_wavelengths is True: (image, wavelengths, metadata)

    Example
    -------
    >>> image, wavelengths, meta = read_hyperspectral("image.tif")
    >>> print(f"Shape: {image.shape}, Wavelength range: {wavelengths.min()}-{wavelengths.max()}nm")
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for read_hyperspectral")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ds = gdal.Open(str(filepath), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open file: {filepath}")

    n_bands = ds.RasterCount
    rows = ds.RasterYSize
    cols = ds.RasterXSize

    # Determine which bands to read
    if bands is None:
        bands_to_read = range(n_bands)
    else:
        bands_to_read = bands
        n_bands = len(bands)

    # Read image data
    image = np.zeros((n_bands, rows, cols), dtype=np.float32)
    for i, band_idx in enumerate(bands_to_read):
        band = ds.GetRasterBand(band_idx + 1)  # GDAL bands are 1-indexed
        image[i] = band.ReadAsArray().astype(np.float32)

    if not return_wavelengths:
        ds = None
        return image

    # Extract wavelengths
    wavelengths = _extract_wavelengths_from_dataset(ds, ds.RasterCount)
    if bands is not None:
        wavelengths = wavelengths[bands]

    # Build metadata dictionary
    metadata = {
        'rows': rows,
        'cols': cols,
        'bands': n_bands,
        'projection': ds.GetProjection(),
        'geotransform': ds.GetGeoTransform(),
        'nodata': ds.GetRasterBand(1).GetNoDataValue(),
        'driver': ds.GetDriver().ShortName,
        'filepath': str(filepath),
    }

    ds = None
    return image, wavelengths, metadata


def _extract_wavelengths_from_dataset(ds: 'gdal.Dataset',
                                       n_bands: int) -> np.ndarray:
    """
    Extract wavelength information from GDAL dataset.

    Attempts multiple methods with comprehensive key name support:
    1. ENVI header metadata
    2. Band descriptions (individual wavelengths per band)
    3. Generic metadata fields with various naming conventions
    4. Fuzzy search for wavelength-related keys
    5. Default range if nothing found

    Supported key names include:
    - wavelength, wavelengths, Wavelength, Wavelengths, WAVELENGTH, WAVELENGTHS
    - wl, Wl, WL, wls, Wls, WLS
    - lambda, Lambda, LAMBDA, lambdas, Lambdas, LAMBDAS
    - center_wavelength, central_wavelength, band_wavelength
    - And variations with underscores

    Parameters
    ----------
    ds : gdal.Dataset
        GDAL dataset object
    n_bands : int
        Number of bands in the dataset

    Returns
    -------
    np.ndarray
        Array of wavelengths in nanometers
    """
    import re

    # Comprehensive list of possible wavelength key names
    WAVELENGTH_KEYS = [
        # Standard names
        'wavelength', 'wavelengths', 'Wavelength', 'Wavelengths',
        'WAVELENGTH', 'WAVELENGTHS',
        # Abbreviations
        'wl', 'Wl', 'WL', 'wls', 'Wls', 'WLS',
        # Lambda variations
        'lambda', 'Lambda', 'LAMBDA', 'lambdas', 'Lambdas', 'LAMBDAS',
        # Descriptive names
        'center_wavelength', 'center_wavelengths',
        'Center_Wavelength', 'Center_Wavelengths',
        'CENTER_WAVELENGTH', 'CENTER_WAVELENGTHS',
        'central_wavelength', 'central_wavelengths',
        'Central_Wavelength', 'Central_Wavelengths',
        'band_wavelength', 'band_wavelengths',
        'Band_Wavelength', 'Band_Wavelengths',
        'BAND_WAVELENGTH', 'BAND_WAVELENGTHS',
        # With underscores
        'wave_length', 'wave_lengths', 'Wave_Length', 'Wave_Lengths',
        # Spectral
        'spectral_wavelength', 'spectral_wavelengths',
        'Spectral_Wavelength', 'Spectral_Wavelengths',
        # Common variations
        'bandcenters', 'band_centers', 'BandCenters',
    ]

    wavelengths = None

    # Method 1: ENVI metadata domain
    envi_meta = ds.GetMetadata('ENVI')
    if envi_meta:
        for key in WAVELENGTH_KEYS:
            if key in envi_meta:
                wavelengths = _parse_wavelength_string(envi_meta[key])
                if wavelengths is not None and len(wavelengths) == n_bands:
                    return wavelengths

    # Method 2: Band descriptions
    wavelengths = _extract_from_band_descriptions(ds, n_bands)
    if wavelengths is not None:
        return wavelengths

    # Method 3: Check multiple metadata domains
    domains_to_check = ['', 'IMAGE_STRUCTURE', 'DERIVED_SUBDATASETS',
                        'SUBDATASETS', 'xml:', 'GEOLOCATION', 'RPC']

    for domain in domains_to_check:
        try:
            meta = ds.GetMetadata(domain) if domain else ds.GetMetadata()
            if meta:
                # Try exact key matches
                for key in WAVELENGTH_KEYS:
                    if key in meta:
                        wavelengths = _parse_wavelength_string(meta[key])
                        if wavelengths is not None and len(wavelengths) == n_bands:
                            return wavelengths
        except:
            continue

    # Method 4: Fuzzy search in default metadata
    meta = ds.GetMetadata()
    if meta:
        search_terms = ['wave', 'wl', 'lambda', 'spectral', 'band']
        for key, value in meta.items():
            key_lower = key.lower()
            if any(term in key_lower for term in search_terms):
                wavelengths = _parse_wavelength_string(value)
                if wavelengths is not None and len(wavelengths) == n_bands:
                    return wavelengths

    # Method 5: Check band-level metadata
    wavelengths = _extract_from_band_metadata(ds, n_bands, WAVELENGTH_KEYS)
    if wavelengths is not None:
        return wavelengths

    # Fallback: Default wavelength range
    warnings.warn(
        f"Could not extract wavelengths from metadata. "
        f"Using default range 400-1000nm for {n_bands} bands."
    )
    return np.linspace(400, 1000, n_bands)


def _parse_wavelength_string(wl_str: str) -> Optional[np.ndarray]:
    """
    Parse a wavelength string into an array of floats.

    Handles various formats:
    - Comma-separated: "400, 410, 420, ..."
    - Space-separated: "400 410 420 ..."
    - ENVI format: "{400, 410, 420, ...}"
    - Semicolon-separated: "400; 410; 420; ..."
    - Tab-separated
    - With units: "400nm, 410nm, ..." or "0.4um, 0.41um, ..."

    Parameters
    ----------
    wl_str : str
        Wavelength string to parse

    Returns
    -------
    Optional[np.ndarray]
        Array of wavelengths in nm, or None if parsing failed
    """
    import re

    if not wl_str or not isinstance(wl_str, str):
        return None

    # Clean up the string
    wl_str = wl_str.strip()

    # Remove ENVI-style braces and brackets
    wl_str = wl_str.strip('{}[]()').strip()

    # Remove newlines and normalize whitespace
    wl_str = ' '.join(wl_str.split())

    # Remove units (nm, nanometer, um, micrometer, etc.)
    wl_str = re.sub(r'\s*(nm|nanometer|nanometre|um|µm|micrometer|micrometre)\s*',
                    ' ', wl_str, flags=re.IGNORECASE)

    # Try different separators
    separators = [',', ';', '\t', ' ']

    for sep in separators:
        if sep in wl_str:
            parts = [p.strip() for p in wl_str.split(sep) if p.strip()]
            try:
                values = [float(p) for p in parts]
                wavelengths = np.array(values)

                # Check if values might be in micrometers and convert to nm
                if wavelengths.max() < 10:  # Likely in micrometers
                    wavelengths = wavelengths * 1000

                return wavelengths
            except ValueError:
                continue

    # Try parsing as space-separated numbers
    try:
        parts = wl_str.split()
        values = [float(p) for p in parts]
        wavelengths = np.array(values)

        if wavelengths.max() < 10:
            wavelengths = wavelengths * 1000

        return wavelengths
    except ValueError:
        pass

    return None


def _extract_from_band_descriptions(ds: 'gdal.Dataset',
                                     n_bands: int) -> Optional[np.ndarray]:
    """
    Extract wavelengths from individual band descriptions.

    Handles various formats in band descriptions:
    - "550" or "550.0"
    - "550nm" or "550 nm"
    - "Band 550" or "band_550"
    - "B550" or "b550"
    - "wavelength: 550"
    - "wl=550"
    - "lambda=550"

    Parameters
    ----------
    ds : gdal.Dataset
        GDAL dataset
    n_bands : int
        Number of bands

    Returns
    -------
    Optional[np.ndarray]
        Array of wavelengths or None
    """
    import re

    wavelengths = []

    for i in range(n_bands):
        band = ds.GetRasterBand(i + 1)
        desc = band.GetDescription()

        if desc:
            wl = _parse_single_wavelength(desc)
            if wl is not None:
                wavelengths.append(wl)
            else:
                return None  # Can't parse this band
        else:
            return None  # No description

    if len(wavelengths) == n_bands:
        return np.array(wavelengths)

    return None


def _extract_from_band_metadata(ds: 'gdal.Dataset', n_bands: int,
                                 keys: List[str]) -> Optional[np.ndarray]:
    """
    Extract wavelengths from individual band metadata.

    Parameters
    ----------
    ds : gdal.Dataset
        GDAL dataset
    n_bands : int
        Number of bands
    keys : List[str]
        List of possible wavelength key names

    Returns
    -------
    Optional[np.ndarray]
        Array of wavelengths or None
    """
    wavelengths = []

    for i in range(n_bands):
        band = ds.GetRasterBand(i + 1)
        band_meta = band.GetMetadata()

        if not band_meta:
            return None

        wl_found = False
        for key in keys:
            if key in band_meta:
                try:
                    wl = float(band_meta[key])
                    # Convert if in micrometers
                    if wl < 10:
                        wl = wl * 1000
                    wavelengths.append(wl)
                    wl_found = True
                    break
                except ValueError:
                    continue

        if not wl_found:
            return None

    if len(wavelengths) == n_bands:
        return np.array(wavelengths)

    return None


def _parse_single_wavelength(text: str) -> Optional[float]:
    """
    Parse a single wavelength value from various text formats.

    Parameters
    ----------
    text : str
        Text that may contain a wavelength value

    Returns
    -------
    Optional[float]
        Parsed wavelength in nm, or None if parsing failed
    """
    import re

    text = text.strip()

    # Patterns to try (in order of specificity)
    patterns = [
        # "wavelength: 550" or "wl=550" or "lambda: 550"
        r'(?:wavelength|wl|lambda|wave)\s*[:=]\s*([0-9.]+)',
        # "550nm" or "550 nm" or "550.5nm"
        r'([0-9.]+)\s*(?:nm|nanometer|nanometre)',
        # "550um" or "0.55 um" - micrometers
        r'([0-9.]+)\s*(?:um|µm|micrometer|micrometre)',
        # "Band 550" or "band_550" or "B550"
        r'(?:band|b)[_\s]*([0-9.]+)',
        # Just a number at start
        r'^([0-9.]+)',
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))

                # Handle micrometers (pattern index 2)
                if i == 2 or (0.1 <= value <= 10):
                    value = value * 1000  # Convert to nm

                # Sanity check: wavelength should be in reasonable range (200-3000nm)
                if 200 <= value <= 3000:
                    return value

            except ValueError:
                continue

    return None


def write_raster(data: np.ndarray,
                 output_path: Union[str, Path],
                 geotransform: Tuple[float, ...],
                 projection: str,
                 nodata: float = -9999.0,
                 dtype: Optional[int] = None,
                 band_names: Optional[List[str]] = None,
                 compress: bool = True) -> Path:
    """
    Write array to GeoTIFF file.

    Parameters
    ----------
    data : np.ndarray
        2D or 3D array to write. Shape: (rows, cols) or (bands, rows, cols)
    output_path : str or Path
        Output file path
    geotransform : tuple
        GDAL geotransform (6 elements)
    projection : str
        WKT projection string
    nodata : float, optional
        NoData value (default: -9999.0)
    dtype : int, optional
        GDAL data type. If None, determined from array dtype.
    band_names : List[str], optional
        Names for each band
    compress : bool, optional
        Apply LZW compression (default: True)

    Returns
    -------
    Path
        Path to the created file

    Example
    -------
    >>> write_raster(ndvi_array, "ndvi.tif", geotransform, projection)
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for write_raster")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle 2D vs 3D arrays
    if data.ndim == 2:
        n_bands = 1
        rows, cols = data.shape
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        n_bands, rows, cols = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {data.shape}")

    # Determine GDAL data type
    if dtype is None:
        dtype_map = {
            np.dtype('float32'): gdal.GDT_Float32,
            np.dtype('float64'): gdal.GDT_Float64,
            np.dtype('int16'): gdal.GDT_Int16,
            np.dtype('int32'): gdal.GDT_Int32,
            np.dtype('uint8'): gdal.GDT_Byte,
            np.dtype('uint16'): gdal.GDT_UInt16,
            np.dtype('uint32'): gdal.GDT_UInt32,
        }
        dtype = dtype_map.get(data.dtype, gdal.GDT_Float32)

    # Create options
    options = ['TILED=YES']
    if compress:
        options.append('COMPRESS=LZW')

    # Create dataset
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(output_path), cols, rows, n_bands, dtype, options)

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    # Write bands
    for i in range(n_bands):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(data[i])
        band.SetNoDataValue(nodata)

        if band_names and i < len(band_names):
            band.SetDescription(band_names[i])

    ds.FlushCache()
    ds = None

    return output_path


def create_wavelength_file(wavelengths: np.ndarray,
                           output_path: Union[str, Path],
                           units: str = 'nm') -> Path:
    """
    Create a wavelength metadata file (JSON format).

    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelengths
    output_path : str or Path
        Output file path
    units : str, optional
        Wavelength units (default: 'nm')

    Returns
    -------
    Path
        Path to the created file
    """
    output_path = Path(output_path)

    metadata = {
        'wavelengths': wavelengths.tolist(),
        'units': units,
        'n_bands': len(wavelengths),
        'min': float(wavelengths.min()),
        'max': float(wavelengths.max()),
        'mean_spacing': float(np.mean(np.diff(wavelengths))),
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


def load_wavelength_file(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load wavelengths from a JSON metadata file.

    Parameters
    ----------
    filepath : str or Path
        Path to wavelength file

    Returns
    -------
    np.ndarray
        Array of wavelengths
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)

    return np.array(metadata['wavelengths'])


def interpolate_bands(image: np.ndarray,
                      source_wavelengths: np.ndarray,
                      target_wavelengths: np.ndarray) -> np.ndarray:
    """
    Interpolate hyperspectral image to new wavelength grid.

    Parameters
    ----------
    image : np.ndarray
        Source image (bands, rows, cols)
    source_wavelengths : np.ndarray
        Original wavelengths
    target_wavelengths : np.ndarray
        Target wavelengths

    Returns
    -------
    np.ndarray
        Interpolated image with shape (len(target_wavelengths), rows, cols)
    """
    from scipy import interpolate

    n_bands, rows, cols = image.shape
    n_target = len(target_wavelengths)

    # Reshape for interpolation
    image_2d = image.reshape(n_bands, -1)

    # Create interpolation function
    interp_func = interpolate.interp1d(
        source_wavelengths, image_2d, axis=0,
        kind='linear', bounds_error=False, fill_value='extrapolate'
    )

    # Interpolate
    result_2d = interp_func(target_wavelengths)

    # Reshape back
    return result_2d.reshape(n_target, rows, cols)


def calculate_statistics(data: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate statistics for a raster array.

    Parameters
    ----------
    data : np.ndarray
        Input array (2D)
    mask : np.ndarray, optional
        Boolean mask (True = valid pixels)

    Returns
    -------
    Dict
        Dictionary of statistics
    """
    if mask is not None:
        valid_data = data[mask]
    else:
        valid_data = data[np.isfinite(data)]

    if len(valid_data) == 0:
        return {
            'count': 0,
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'p25': np.nan,
            'p75': np.nan,
        }

    return {
        'count': len(valid_data),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'std': float(np.std(valid_data)),
        'median': float(np.median(valid_data)),
        'p25': float(np.percentile(valid_data, 25)),
        'p75': float(np.percentile(valid_data, 75)),
    }


def extract_plot_values(image: np.ndarray,
                        shapefile_path: Union[str, Path],
                        wavelengths: Optional[np.ndarray] = None) -> Dict:
    """
    Extract mean values for each polygon in a shapefile.

    Parameters
    ----------
    image : np.ndarray
        Hyperspectral image (bands, rows, cols)
    shapefile_path : str or Path
        Path to shapefile with plot boundaries
    wavelengths : np.ndarray, optional
        Wavelength array for reference

    Returns
    -------
    Dict
        Dictionary with plot ID keys and spectral values
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL/OGR is required for extract_plot_values")

    # This is a placeholder - full implementation would require
    # rasterizing the shapefile and extracting zonal statistics
    raise NotImplementedError(
        "Plot extraction requires additional implementation. "
        "Consider using rasterstats or rasterio for zonal statistics."
    )


def apply_quality_mask(image: np.ndarray,
                       wavelengths: np.ndarray,
                       min_reflectance: float = 0.0,
                       max_reflectance: float = 1.0,
                       check_atmospheric: bool = True) -> np.ndarray:
    """
    Create quality mask based on reflectance values.

    Parameters
    ----------
    image : np.ndarray
        Hyperspectral image (bands, rows, cols)
    wavelengths : np.ndarray
        Wavelength array
    min_reflectance : float
        Minimum valid reflectance
    max_reflectance : float
        Maximum valid reflectance
    check_atmospheric : bool
        Check for atmospheric absorption bands

    Returns
    -------
    np.ndarray
        Boolean mask (True = good quality)
    """
    n_bands, rows, cols = image.shape

    # Start with all pixels valid
    mask = np.ones((rows, cols), dtype=bool)

    # Check reflectance range
    for i in range(n_bands):
        band_data = image[i]
        mask &= (band_data >= min_reflectance)
        mask &= (band_data <= max_reflectance)
        mask &= np.isfinite(band_data)

    # Check for saturated pixels in visible bands
    vis_bands = (wavelengths >= 400) & (wavelengths <= 700)
    if np.any(vis_bands):
        vis_mean = np.mean(image[vis_bands], axis=0)
        mask &= (vis_mean > 0.01)  # Not too dark
        mask &= (vis_mean < 0.95)  # Not saturated

    # Check atmospheric absorption regions if requested
    if check_atmospheric:
        # Water vapor bands around 940nm and 1140nm
        for center, width in [(940, 20), (1140, 30)]:
            in_band = (wavelengths >= center - width) & (wavelengths <= center + width)
            if np.any(in_band):
                band_mean = np.mean(image[in_band], axis=0)
                # Flag pixels with unrealistic values in absorption bands
                mask &= (band_mean < 0.8)

    return mask


def resample_image(image: np.ndarray,
                   scale_factor: float,
                   method: str = 'average') -> np.ndarray:
    """
    Resample image to different spatial resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image (bands, rows, cols)
    scale_factor : float
        Scale factor (< 1 for downsampling, > 1 for upsampling)
    method : str
        Resampling method ('average', 'nearest', 'bilinear')

    Returns
    -------
    np.ndarray
        Resampled image
    """
    from scipy import ndimage

    n_bands, rows, cols = image.shape
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    result = np.zeros((n_bands, new_rows, new_cols), dtype=image.dtype)

    zoom_factors = (1, scale_factor, scale_factor)

    if method == 'nearest':
        order = 0
    elif method == 'bilinear':
        order = 1
    else:  # average
        order = 1

    for i in range(n_bands):
        result[i] = ndimage.zoom(image[i], scale_factor, order=order)

    return result


def generate_report(results: Dict,
                    output_path: Union[str, Path],
                    format: str = 'txt') -> Path:
    """
    Generate a summary report of processing results.

    Parameters
    ----------
    results : Dict
        Processing results dictionary
    output_path : str or Path
        Output file path
    format : str
        Output format ('txt' or 'json')

    Returns
    -------
    Path
        Path to the generated report
    """
    output_path = Path(output_path)

    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        json_results = {k: convert(v) for k, v in results.items()}

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

    else:  # txt format
        lines = ["N4Crops Processing Report", "=" * 50, ""]

        for key, value in results.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, np.ndarray):
                lines.append(f"{key}: array with shape {value.shape}")
            else:
                lines.append(f"{key}: {value}")
            lines.append("")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    return output_path
