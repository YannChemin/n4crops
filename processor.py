"""
Nitrogen Processor: GDAL-based Hyperspectral Image Processing
==============================================================

This module provides the main processing pipeline for estimating nitrogen
content and nitrogen use efficiency from hyperspectral imagery using GDAL.

Features:
- Read hyperspectral imagery in various formats (ENVI, GeoTIFF, etc.)
- Apply atmospheric correction (optional)
- Calculate vegetation indices
- Estimate nitrogen parameters using calibration models
- Export results as GeoTIFF rasters

References:
-----------
[1] Olson et al. (2022) - Remote Sens. 14, 1721
[2] Thenkabail et al. (2014) - Photogramm. Eng. Remote Sens. 80, 697-723
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    warnings.warn("GDAL not available. Some functionality will be limited.")

from .indices import HyperspectralIndices
from .calibration import CropCalibration, CropType


@dataclass
class ProcessingConfig:
    """Configuration for nitrogen processing pipeline."""
    crop_type: str = "maize"  # maize, rice, wheat
    growth_stage: str = "R1"  # V16, V18, R1, R2
    apply_soil_mask: bool = True
    soil_ndvi_threshold: float = 0.3
    nodata_value: float = -9999.0
    output_dtype: str = "float32"
    calculate_all_indices: bool = False
    indices_to_calculate: List[str] = field(default_factory=lambda: [
        'HBSI1', 'HBSI2', 'HBCI8', 'HBCI9', 'NDVI', 'NDRE'
    ])


@dataclass
class ImageMetadata:
    """Metadata for hyperspectral image."""
    rows: int
    cols: int
    bands: int
    wavelengths: np.ndarray
    projection: str
    geotransform: Tuple[float, ...]
    nodata: Optional[float]
    driver: str
    filepath: str


class NitrogenProcessor:
    """
    Main processor for nitrogen estimation from hyperspectral imagery.

    This class provides a complete pipeline for:
    1. Reading hyperspectral imagery using GDAL
    2. Extracting wavelength information
    3. Calculating vegetation indices
    4. Estimating nitrogen parameters (NCE, NIE, pN)
    5. Exporting results

    Attributes
    ----------
    config : ProcessingConfig
        Configuration settings for processing
    calibration : CropCalibration
        Calibration models for nitrogen estimation

    Example
    -------
    >>> from n4crops import NitrogenProcessor, ProcessingConfig
    >>> config = ProcessingConfig(crop_type="maize", growth_stage="R1")
    >>> processor = NitrogenProcessor(config)
    >>> results = processor.process("hyperspectral_image.tif", "output_dir")
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the NitrogenProcessor.

        Parameters
        ----------
        config : ProcessingConfig, optional
            Processing configuration. If None, uses default settings.
        """
        if not GDAL_AVAILABLE:
            raise ImportError(
                "GDAL is required for NitrogenProcessor. "
                "Install with: conda install -c conda-forge gdal"
            )

        self.config = config or ProcessingConfig()
        self.calibration = CropCalibration()
        self._indices_calculator = None
        self._metadata = None

    def read_image(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Read hyperspectral image using GDAL.

        Parameters
        ----------
        filepath : str or Path
            Path to the hyperspectral image

        Returns
        -------
        Tuple[np.ndarray, ImageMetadata]
            Image array (bands, rows, cols) and metadata
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")

        ds = gdal.Open(str(filepath), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Could not open image: {filepath}")

        # Read image data
        bands = ds.RasterCount
        rows = ds.RasterYSize
        cols = ds.RasterXSize

        image = np.zeros((bands, rows, cols), dtype=np.float32)
        for i in range(bands):
            band = ds.GetRasterBand(i + 1)
            image[i] = band.ReadAsArray().astype(np.float32)

        # Extract wavelengths from metadata
        wavelengths = self._extract_wavelengths(ds, bands)

        # Get spatial reference info
        projection = ds.GetProjection()
        geotransform = ds.GetGeoTransform()

        # Get nodata value
        nodata = ds.GetRasterBand(1).GetNoDataValue()

        metadata = ImageMetadata(
            rows=rows,
            cols=cols,
            bands=bands,
            wavelengths=wavelengths,
            projection=projection,
            geotransform=geotransform,
            nodata=nodata,
            driver=ds.GetDriver().ShortName,
            filepath=str(filepath)
        )

        ds = None  # Close dataset
        self._metadata = metadata

        return image, metadata

    def _extract_wavelengths(self, ds: 'gdal.Dataset', n_bands: int) -> np.ndarray:
        """
        Extract wavelength information from image metadata.

        Attempts to read wavelengths from multiple sources and naming conventions:
        1. ENVI header metadata
        2. Band descriptions (individual wavelengths per band)
        3. Generic metadata with various key names
        4. Multiple metadata domains

        Supported key names: wavelength, wavelengths, Wavelength, Wavelengths,
        WAVELENGTH, WAVELENGTHS, wl, Wl, WL, lambda, Lambda, LAMBDA,
        center_wavelength, central_wavelength, band_wavelength, etc.

        Parameters
        ----------
        ds : gdal.Dataset
            GDAL dataset object
        n_bands : int
            Number of bands

        Returns
        -------
        np.ndarray
            Array of wavelengths in nm
        """
        # Common wavelength key names (case variations and abbreviations)
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
            'band_wavelength', 'band_wavelengths',
            'Band_Wavelength', 'Band_Wavelengths',
            # With underscores
            'wave_length', 'wave_lengths',
            # Spectral
            'spectral_wavelength', 'spectral_wavelengths',
        ]

        wavelengths = None

        # Method 1: Try ENVI metadata domain
        wavelengths = self._try_envi_wavelengths(ds, WAVELENGTH_KEYS, n_bands)

        # Method 2: Try band descriptions
        if wavelengths is None:
            wavelengths = self._try_band_descriptions(ds, n_bands)

        # Method 3: Try generic metadata (default domain)
        if wavelengths is None:
            wavelengths = self._try_metadata_domain(ds, '', WAVELENGTH_KEYS, n_bands)

        # Method 4: Try other common metadata domains
        if wavelengths is None:
            for domain in ['IMAGE_STRUCTURE', 'DERIVED_SUBDATASETS', 'xml:',
                          'SUBDATASETS', 'RPC', 'GEOLOCATION']:
                wavelengths = self._try_metadata_domain(ds, domain, WAVELENGTH_KEYS, n_bands)
                if wavelengths is not None:
                    break

        # Method 5: Try to find wavelength info in any metadata key containing 'wl' or 'wave'
        if wavelengths is None:
            wavelengths = self._search_metadata_fuzzy(ds, n_bands)

        # Fallback: create default wavelengths
        if wavelengths is None:
            warnings.warn(
                "Could not extract wavelengths from metadata. "
                f"Using default range 400-1000nm for {n_bands} bands."
            )
            wavelengths = np.linspace(400, 1000, n_bands)

        return wavelengths

    def _try_envi_wavelengths(self, ds: 'gdal.Dataset', keys: List[str],
                              n_bands: int) -> Optional[np.ndarray]:
        """Try to extract wavelengths from ENVI metadata domain."""
        envi_metadata = ds.GetMetadata('ENVI')
        if not envi_metadata:
            return None

        for key in keys:
            if key in envi_metadata:
                wl_str = envi_metadata[key]
                wavelengths = self._parse_wavelength_string(wl_str)
                if wavelengths is not None and len(wavelengths) == n_bands:
                    return wavelengths

        return None

    def _try_band_descriptions(self, ds: 'gdal.Dataset',
                               n_bands: int) -> Optional[np.ndarray]:
        """
        Try to extract wavelengths from individual band descriptions.

        Handles various formats:
        - "550" or "550.0"
        - "550nm" or "550 nm"
        - "Band 550" or "band_550"
        - "B550" or "b550"
        - "wavelength: 550"
        - "wl=550"
        """
        import re

        wavelengths = []

        for i in range(n_bands):
            band = ds.GetRasterBand(i + 1)
            desc = band.GetDescription()

            if not desc:
                # Also check band metadata
                band_meta = band.GetMetadata()
                for key in ['wavelength', 'Wavelength', 'wl', 'lambda', 'center_wavelength']:
                    if key in band_meta:
                        desc = band_meta[key]
                        break

            if desc:
                wl = self._parse_single_wavelength(desc)
                if wl is not None:
                    wavelengths.append(wl)
                else:
                    # Can't parse this band, abort
                    return None
            else:
                return None

        if len(wavelengths) == n_bands:
            return np.array(wavelengths)

        return None

    def _parse_single_wavelength(self, text: str) -> Optional[float]:
        """
        Parse a single wavelength value from various text formats.

        Parameters
        ----------
        text : str
            Text that may contain a wavelength value

        Returns
        -------
        Optional[float]
            Parsed wavelength or None if parsing failed
        """
        import re

        text = text.strip()

        # Patterns to try (in order of specificity)
        patterns = [
            # "wavelength: 550" or "wl=550" or "lambda: 550"
            r'(?:wavelength|wl|lambda|wave)\s*[:=]\s*([0-9.]+)',
            # "550nm" or "550 nm" or "550.5nm"
            r'([0-9.]+)\s*(?:nm|nanometer|nanometre)',
            # "Band 550" or "band_550" or "B550"
            r'(?:band|b)[_\s]*([0-9.]+)',
            # Just a number (must be in reasonable wavelength range)
            r'^([0-9.]+)$',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Sanity check: wavelength should be in reasonable range
                    # Allow 200-3000nm for hyperspectral (UV to SWIR)
                    if 200 <= value <= 3000:
                        return value
                    # Could be in micrometers - convert
                    elif 0.2 <= value <= 3.0:
                        return value * 1000  # Convert µm to nm
                except ValueError:
                    continue

        return None

    def _try_metadata_domain(self, ds: 'gdal.Dataset', domain: str,
                             keys: List[str], n_bands: int) -> Optional[np.ndarray]:
        """Try to extract wavelengths from a specific metadata domain."""
        try:
            metadata = ds.GetMetadata(domain) if domain else ds.GetMetadata()
        except:
            return None

        if not metadata:
            return None

        for key in keys:
            if key in metadata:
                wl_str = metadata[key]
                wavelengths = self._parse_wavelength_string(wl_str)
                if wavelengths is not None and len(wavelengths) == n_bands:
                    return wavelengths

        return None

    def _search_metadata_fuzzy(self, ds: 'gdal.Dataset',
                               n_bands: int) -> Optional[np.ndarray]:
        """
        Search for wavelength data using fuzzy key matching.

        Looks for any metadata key containing 'wave', 'wl', or 'lambda'.
        """
        metadata = ds.GetMetadata()
        if not metadata:
            return None

        search_terms = ['wave', 'wl', 'lambda', 'spectral']

        for key, value in metadata.items():
            key_lower = key.lower()
            if any(term in key_lower for term in search_terms):
                wavelengths = self._parse_wavelength_string(value)
                if wavelengths is not None and len(wavelengths) == n_bands:
                    return wavelengths

        return None

    def _parse_wavelength_string(self, wl_str: str) -> Optional[np.ndarray]:
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
            Array of wavelengths or None if parsing failed
        """
        import re

        if not wl_str or not isinstance(wl_str, str):
            return None

        # Clean up the string
        wl_str = wl_str.strip()

        # Remove ENVI-style braces
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

                    # Check if values might be in micrometers and convert
                    if wavelengths.max() < 10:  # Likely in micrometers
                        wavelengths = wavelengths * 1000

                    return wavelengths
                except ValueError:
                    continue

        # Try as single comma-free list of numbers
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

    def apply_soil_mask(self, image: np.ndarray,
                        wavelengths: np.ndarray) -> np.ndarray:
        """
        Create and apply soil mask based on NDVI threshold.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)
        wavelengths : np.ndarray
            Wavelength array

        Returns
        -------
        np.ndarray
            Boolean mask (True = vegetation, False = soil/background)
        """
        hsi = HyperspectralIndices(wavelengths)
        ndvi = hsi.ndvi(image)
        mask = ndvi > self.config.soil_ndvi_threshold
        return mask

    def process(self, input_path: Union[str, Path],
                output_dir: Union[str, Path],
                wavelengths: Optional[np.ndarray] = None) -> Dict[str, Path]:
        """
        Process hyperspectral image and estimate nitrogen parameters.

        Parameters
        ----------
        input_path : str or Path
            Path to input hyperspectral image
        output_dir : str or Path
            Directory for output files
        wavelengths : np.ndarray, optional
            Override wavelengths if not in image metadata

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping output names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read image
        print(f"Reading image: {input_path}")
        image, metadata = self.read_image(input_path)

        if wavelengths is not None:
            metadata.wavelengths = wavelengths

        print(f"Image shape: {image.shape}")
        print(f"Wavelength range: {metadata.wavelengths.min():.1f} - "
              f"{metadata.wavelengths.max():.1f} nm")

        # Initialize indices calculator
        self._indices_calculator = HyperspectralIndices(metadata.wavelengths)

        # Apply soil mask
        if self.config.apply_soil_mask:
            print("Applying soil mask...")
            veg_mask = self.apply_soil_mask(image, metadata.wavelengths)
        else:
            veg_mask = np.ones((metadata.rows, metadata.cols), dtype=bool)

        # Calculate indices
        print("Calculating vegetation indices...")
        indices = self._calculate_indices(image)

        # Apply mask to indices
        for name, index_array in indices.items():
            if index_array is not None:
                indices[name] = np.where(veg_mask, index_array,
                                         self.config.nodata_value)

        # Estimate nitrogen parameters
        print(f"Estimating nitrogen parameters for {self.config.crop_type}...")
        n_params = self._estimate_nitrogen(indices)

        # Apply mask to N parameters
        for name, param_array in n_params.items():
            if param_array is not None:
                n_params[name] = np.where(veg_mask, param_array,
                                          self.config.nodata_value)

        # Write outputs
        print("Writing output files...")
        output_files = {}

        # Write indices
        for name, index_array in indices.items():
            if index_array is not None:
                out_path = output_dir / f"{name}.tif"
                self._write_raster(index_array, out_path, metadata)
                output_files[name] = out_path

        # Write N parameters
        for name, param_array in n_params.items():
            if param_array is not None:
                out_path = output_dir / f"N_{name}.tif"
                self._write_raster(param_array, out_path, metadata)
                output_files[f"N_{name}"] = out_path

        # Write vegetation mask
        mask_path = output_dir / "vegetation_mask.tif"
        self._write_raster(veg_mask.astype(np.uint8), mask_path, metadata,
                          dtype=gdal.GDT_Byte)
        output_files['vegetation_mask'] = mask_path

        print(f"Processing complete. {len(output_files)} files written.")
        return output_files

    def _calculate_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate vegetation indices based on configuration."""
        indices = {}

        if self.config.calculate_all_indices:
            return self._indices_calculator.calculate_all_indices(image)

        # Get crop-specific indices
        crop_type = self.config.crop_type.lower()

        if crop_type == 'maize':
            indices.update(self._indices_calculator.get_maize_nce_indices(image))
            indices.update(self._indices_calculator.get_maize_nie_indices(image))
        elif crop_type == 'rice':
            indices.update(self._indices_calculator.get_rice_n_indices(image))
        elif crop_type == 'wheat':
            indices.update(self._indices_calculator.get_wheat_n_indices(image))

        # Add any additional requested indices
        for idx_name in self.config.indices_to_calculate:
            if idx_name not in indices:
                method_name = idx_name.lower().replace('/', '_')
                if hasattr(self._indices_calculator, method_name):
                    try:
                        indices[idx_name] = getattr(
                            self._indices_calculator, method_name)(image)
                    except Exception as e:
                        print(f"Warning: Could not calculate {idx_name}: {e}")

        return indices

    def _estimate_nitrogen(self, indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Estimate nitrogen parameters using calibration models.

        Parameters
        ----------
        indices : Dict[str, np.ndarray]
            Dictionary of calculated vegetation indices

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of estimated nitrogen parameters
        """
        crop_type = CropType[self.config.crop_type.upper()]
        growth_stage = self.config.growth_stage

        n_params = {}

        # Get calibration models
        models = self.calibration.get_models(crop_type, growth_stage)

        for param_name, model_info in models.items():
            index_name = model_info['index']
            if index_name in indices and indices[index_name] is not None:
                n_params[param_name] = self.calibration.apply_model(
                    indices[index_name], model_info
                )
            else:
                print(f"Warning: Index {index_name} not available for {param_name}")

        return n_params

    def _write_raster(self, data: np.ndarray, output_path: Path,
                      metadata: ImageMetadata,
                      dtype: int = None) -> None:
        """
        Write array to GeoTIFF file.

        Parameters
        ----------
        data : np.ndarray
            2D array to write
        output_path : Path
            Output file path
        metadata : ImageMetadata
            Spatial metadata for georeferencing
        dtype : int, optional
            GDAL data type (default: GDT_Float32)
        """
        if dtype is None:
            dtype = gdal.GDT_Float32

        driver = gdal.GetDriverByName('GTiff')

        if len(data.shape) == 2:
            n_bands = 1
            rows, cols = data.shape
        else:
            n_bands, rows, cols = data.shape

        ds = driver.Create(
            str(output_path),
            cols, rows, n_bands,
            dtype,
            options=['COMPRESS=LZW', 'TILED=YES']
        )

        ds.SetProjection(metadata.projection)
        ds.SetGeoTransform(metadata.geotransform)

        if n_bands == 1:
            band = ds.GetRasterBand(1)
            band.WriteArray(data)
            band.SetNoDataValue(self.config.nodata_value)
        else:
            for i in range(n_bands):
                band = ds.GetRasterBand(i + 1)
                band.WriteArray(data[i])
                band.SetNoDataValue(self.config.nodata_value)

        ds.FlushCache()
        ds = None

    def process_array(self, image: np.ndarray,
                      wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a numpy array directly (without file I/O).

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)
        wavelengths : np.ndarray
            Array of wavelengths in nm

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing indices and N parameter estimates
        """
        self._indices_calculator = HyperspectralIndices(wavelengths)

        results = {}

        # Apply soil mask
        if self.config.apply_soil_mask:
            veg_mask = self.apply_soil_mask(image, wavelengths)
        else:
            veg_mask = np.ones(image.shape[1:], dtype=bool)

        results['vegetation_mask'] = veg_mask

        # Calculate indices
        indices = self._calculate_indices(image)

        # Apply mask
        for name, index_array in indices.items():
            if index_array is not None:
                results[f'index_{name}'] = np.where(
                    veg_mask, index_array, np.nan)

        # Estimate N parameters
        n_params = self._estimate_nitrogen(indices)

        for name, param_array in n_params.items():
            if param_array is not None:
                results[f'N_{name}'] = np.where(
                    veg_mask, param_array, np.nan)

        return results


class BatchProcessor:
    """
    Batch processor for multiple hyperspectral images.

    Example
    -------
    >>> from n4crops import BatchProcessor, ProcessingConfig
    >>> config = ProcessingConfig(crop_type="wheat")
    >>> batch = BatchProcessor(config)
    >>> results = batch.process_directory("input_images/", "output/")
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize batch processor.

        Parameters
        ----------
        config : ProcessingConfig, optional
            Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.processor = NitrogenProcessor(config)

    def process_directory(self, input_dir: Union[str, Path],
                          output_dir: Union[str, Path],
                          pattern: str = "*.tif") -> Dict[str, Dict[str, Path]]:
        """
        Process all matching images in a directory.

        Parameters
        ----------
        input_dir : str or Path
            Input directory containing images
        output_dir : str or Path
            Output directory for results
        pattern : str
            Glob pattern for matching input files

        Returns
        -------
        Dict[str, Dict[str, Path]]
            Dictionary mapping input filenames to their output files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        results = {}

        for input_file in sorted(input_dir.glob(pattern)):
            print(f"\nProcessing: {input_file.name}")

            # Create output subdirectory
            file_output_dir = output_dir / input_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                output_files = self.processor.process(input_file, file_output_dir)
                results[input_file.name] = output_files
            except Exception as e:
                print(f"Error processing {input_file.name}: {e}")
                results[input_file.name] = {'error': str(e)}

        return results

    def process_file_list(self, file_list: List[Union[str, Path]],
                          output_dir: Union[str, Path]) -> Dict[str, Dict[str, Path]]:
        """
        Process a list of image files.

        Parameters
        ----------
        file_list : List[str or Path]
            List of input file paths
        output_dir : str or Path
            Output directory for results

        Returns
        -------
        Dict[str, Dict[str, Path]]
            Dictionary mapping input filenames to their output files
        """
        output_dir = Path(output_dir)
        results = {}

        for input_file in file_list:
            input_file = Path(input_file)
            print(f"\nProcessing: {input_file.name}")

            file_output_dir = output_dir / input_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                output_files = self.processor.process(input_file, file_output_dir)
                results[input_file.name] = output_files
            except Exception as e:
                print(f"Error processing {input_file.name}: {e}")
                results[input_file.name] = {'error': str(e)}

        return results
