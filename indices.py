"""
Hyperspectral Vegetation Indices for Nitrogen Estimation
=========================================================

This module implements hyperspectral vegetation indices (HSI) for estimating
nitrogen content, nitrogen use efficiency (NUE), nitrogen conversion efficiency
(NCE), and nitrogen internal efficiency (NIE) in cereal crops.

Indices are organized by category:
- Biomass indices (NDVI, MSAVI, RTVI)
- Structural indices (HBSI1, HBSI2, HBSI3, PSRI)
- Biochemical/Chlorophyll indices (HBCI8, HBCI9, HBCI10, MCARI, TCARI, etc.)
- Red-edge indices (NDRE, CIRE, HREI15, HREI16)
- Crop-specific indices for Rice and Wheat

References:
-----------
[1] Olson et al. (2022) - Remote Sens. 14, 1721
[2] Thenkabail et al. (2014) - Photogramm. Eng. Remote Sens. 80, 697-723
[3] Daughtry et al. (2000) - Remote Sens. Environ. 74, 229-239
[4] Chen et al. (2010) - Remote Sens. Environ. 114, 1987-1997
[5] Gitelson et al. (2003) - J. Plant Physiol. 160, 271-282
[6] Haboudane et al. (2002) - Remote Sens. Environ. 81, 416-426
[7] Xue et al. (2004) - Agron. J. 96, 135-142
[8] Tian et al. (2011) - Field Crops Res. 120, 299-310
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class BandInfo:
    """Information about spectral band configuration."""
    wavelength: float  # Center wavelength in nm
    bandwidth: float   # FWHM bandwidth in nm
    band_index: int    # Index in the image array


class HyperspectralIndices:
    """
    Calculator for hyperspectral vegetation indices related to nitrogen.

    This class provides methods to calculate various vegetation indices
    from hyperspectral imagery for nitrogen estimation in crops.

    Attributes
    ----------
    wavelengths : np.ndarray
        Array of wavelengths corresponding to image bands (in nm)
    tolerance : float
        Tolerance for wavelength matching (default: 5 nm)

    Example
    -------
    >>> import numpy as np
    >>> wavelengths = np.arange(400, 1001, 5)  # 400-1000nm at 5nm resolution
    >>> hsi = HyperspectralIndices(wavelengths)
    >>> # Assuming 'image' is a 3D array (bands, rows, cols)
    >>> ndvi = hsi.calculate_ndvi(image)
    """

    def __init__(self, wavelengths: np.ndarray, tolerance: float = 5.0):
        """
        Initialize the HyperspectralIndices calculator.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of center wavelengths for each band (in nm)
        tolerance : float, optional
            Tolerance for wavelength matching in nm (default: 5.0)
        """
        self.wavelengths = np.array(wavelengths)
        self.tolerance = tolerance
        self._band_cache = {}

    def _get_band(self, image: np.ndarray, wavelength: float) -> np.ndarray:
        """
        Extract the band closest to the specified wavelength.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image array (bands, rows, cols)
        wavelength : float
            Target wavelength in nm

        Returns
        -------
        np.ndarray
            2D array of reflectance values for the selected band
        """
        if wavelength in self._band_cache:
            idx = self._band_cache[wavelength]
        else:
            idx = np.argmin(np.abs(self.wavelengths - wavelength))
            if np.abs(self.wavelengths[idx] - wavelength) > self.tolerance:
                raise ValueError(
                    f"No band found within {self.tolerance}nm of {wavelength}nm. "
                    f"Closest band is at {self.wavelengths[idx]}nm"
                )
            self._band_cache[wavelength] = idx

        return image[idx].astype(np.float64)

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray,
                     fill_value: float = 0.0) -> np.ndarray:
        """Safely divide arrays, handling division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result[~np.isfinite(result)] = fill_value
        return result

    # =========================================================================
    # BIOMASS INDICES
    # =========================================================================

    def ndvi(self, image: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Vegetation Index (NDVI).

        NDVI = (R800 - R670) / (R800 + R670)

        Reference: Rouse et al. (1974)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            NDVI values (-1 to 1)
        """
        r800 = self._get_band(image, 800)
        r670 = self._get_band(image, 670)
        return self._safe_divide(r800 - r670, r800 + r670)

    def msavi(self, image: np.ndarray) -> np.ndarray:
        """
        Modified Soil Adjusted Vegetation Index (MSAVI).

        MSAVI = 0.5 * (2 * R800 + 1 - sqrt((2 * R800 + 1)^2 - 8 * (R800 - R670)))

        Reference: Qi et al. (1994)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            MSAVI values
        """
        r800 = self._get_band(image, 800)
        r670 = self._get_band(image, 670)

        term1 = 2 * r800 + 1
        term2 = np.sqrt(np.maximum(0, term1**2 - 8 * (r800 - r670)))
        return 0.5 * (term1 - term2)

    def rtvi(self, image: np.ndarray) -> np.ndarray:
        """
        Red-edge Triangular Vegetation Index (RTVI).

        RTVI = 100 * (R750 - R730) - 10 * (R750 - R550) * sqrt(R700/R670)

        Reference: Chen et al. (2010)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            RTVI values
        """
        r750 = self._get_band(image, 750)
        r730 = self._get_band(image, 730)
        r700 = self._get_band(image, 700)
        r670 = self._get_band(image, 670)
        r550 = self._get_band(image, 550)

        ratio = self._safe_divide(r700, r670, fill_value=1.0)
        return 100 * (r750 - r730) - 10 * (r750 - r550) * np.sqrt(np.maximum(0, ratio))

    # =========================================================================
    # STRUCTURAL INDICES (from Olson et al. 2022 / Thenkabail et al. 2014)
    # =========================================================================

    def hbsi1(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biomass Structural Index 1 (HBSI1).

        HBSI1 = (R855 - R682) / (R855 + R682)

        Best for NCE prediction at R1 stage.
        Reference: Olson et al. (2022), Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBSI1 values
        """
        r855 = self._get_band(image, 855)
        r682 = self._get_band(image, 682)
        return self._safe_divide(r855 - r682, r855 + r682)

    def hbsi2(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biomass Structural Index 2 (HBSI2).

        HBSI2 = (R910 - R682) / (R910 + R682)

        Best for NCE prediction at R1 stage.
        Reference: Olson et al. (2022), Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBSI2 values
        """
        r910 = self._get_band(image, 910)
        r682 = self._get_band(image, 682)
        return self._safe_divide(r910 - r682, r910 + r682)

    def hbsi3(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biomass Structural Index 3 (HBSI3).

        HBSI3 = (R550 - R682) / (R550 + R682)

        Reference: Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBSI3 values
        """
        r550 = self._get_band(image, 550)
        r682 = self._get_band(image, 682)
        return self._safe_divide(r550 - r682, r550 + r682)

    def psri(self, image: np.ndarray) -> np.ndarray:
        """
        Plant Senescence Reflectance Index (PSRI).

        PSRI = (R678 - R500) / R750

        Reference: Merzlyak et al. (1999)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            PSRI values
        """
        r678 = self._get_band(image, 678)
        r500 = self._get_band(image, 500)
        r750 = self._get_band(image, 750)
        return self._safe_divide(r678 - r500, r750)

    # =========================================================================
    # BIOCHEMICAL/CHLOROPHYLL INDICES (for N concentration & NIE)
    # =========================================================================

    def hbci8(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biochemical Index 8 (HBCI8).

        HBCI8 = (R550 - R515) / (R550 + R515)

        Best for NIE prediction at V16 and R1 stages.
        Reference: Olson et al. (2022), Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBCI8 values
        """
        r550 = self._get_band(image, 550)
        r515 = self._get_band(image, 515)
        return self._safe_divide(r550 - r515, r550 + r515)

    def hbci9(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biochemical Index 9 (HBCI9).

        HBCI9 = (R550 - R490) / (R550 + R490)

        Best for NIE prediction at R1 stage.
        Reference: Olson et al. (2022), Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBCI9 values
        """
        r550 = self._get_band(image, 550)
        r490 = self._get_band(image, 490)
        return self._safe_divide(r550 - r490, r550 + r490)

    def hbci10(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Biochemical Index 10 (HBCI10).

        HBCI10 = (R720 - R550) / (R720 + R550)

        Reference: Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HBCI10 values
        """
        r720 = self._get_band(image, 720)
        r550 = self._get_band(image, 550)
        return self._safe_divide(r720 - r550, r720 + r550)

    def mcari(self, image: np.ndarray) -> np.ndarray:
        """
        Modified Chlorophyll Absorption Reflectance Index (MCARI).

        MCARI = ((R700 - R670) - 0.2 * (R700 - R550)) * (R700/R670)

        Reference: Daughtry et al. (2000)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            MCARI values
        """
        r700 = self._get_band(image, 700)
        r670 = self._get_band(image, 670)
        r550 = self._get_band(image, 550)

        ratio = self._safe_divide(r700, r670, fill_value=1.0)
        return ((r700 - r670) - 0.2 * (r700 - r550)) * ratio

    def tcari(self, image: np.ndarray) -> np.ndarray:
        """
        Transformed Chlorophyll Absorption Ratio (TCARI).

        TCARI = 3 * ((R700 - R670) - 0.2 * (R700 - R550) * (R700/R670))

        Reference: Haboudane et al. (2002)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            TCARI values
        """
        r700 = self._get_band(image, 700)
        r670 = self._get_band(image, 670)
        r550 = self._get_band(image, 550)

        ratio = self._safe_divide(r700, r670, fill_value=1.0)
        return 3 * ((r700 - r670) - 0.2 * (r700 - r550) * ratio)

    def osavi(self, image: np.ndarray) -> np.ndarray:
        """
        Optimized Soil Adjusted Vegetation Index (OSAVI).

        OSAVI = (1 + 0.16) * (R800 - R670) / (R800 + R670 + 0.16)

        Reference: Rondeaux et al. (1996)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            OSAVI values
        """
        r800 = self._get_band(image, 800)
        r670 = self._get_band(image, 670)
        return self._safe_divide(1.16 * (r800 - r670), r800 + r670 + 0.16)

    def tcari_osavi(self, image: np.ndarray) -> np.ndarray:
        """
        TCARI/OSAVI ratio index.

        Combines TCARI and OSAVI for improved chlorophyll estimation
        while minimizing soil background effects.

        Reference: Haboudane et al. (2002)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            TCARI/OSAVI ratio values
        """
        tcari_val = self.tcari(image)
        osavi_val = self.osavi(image)
        return self._safe_divide(tcari_val, osavi_val)

    def dcni(self, image: np.ndarray) -> np.ndarray:
        """
        Double-peak Canopy Nitrogen Index (DCNI).

        DCNI = (R720 - R700) / (R700 - R670) / (R720 - R670 + 0.03)

        Reference: Chen et al. (2010)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            DCNI values
        """
        r720 = self._get_band(image, 720)
        r700 = self._get_band(image, 700)
        r670 = self._get_band(image, 670)

        term1 = self._safe_divide(r720 - r700, r700 - r670, fill_value=0.0)
        return self._safe_divide(term1, r720 - r670 + 0.03)

    def rvi_ii(self, image: np.ndarray) -> np.ndarray:
        """
        Ratio Vegetation Index II (RVI II).

        RVI_II = R810 / R560

        Reference: Xue et al. (2004) - developed for rice

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            RVI II values
        """
        r810 = self._get_band(image, 810)
        r560 = self._get_band(image, 560)
        return self._safe_divide(r810, r560)

    # =========================================================================
    # RED-EDGE INDICES (for plant stress and N status)
    # =========================================================================

    def ndre(self, image: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Red Edge (NDRE).

        NDRE = (R790 - R720) / (R790 + R720)

        Reference: Barnes et al. (2000)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            NDRE values
        """
        r790 = self._get_band(image, 790)
        r720 = self._get_band(image, 720)
        return self._safe_divide(r790 - r720, r790 + r720)

    def cire(self, image: np.ndarray) -> np.ndarray:
        """
        Chlorophyll Index Red Edge (CIRE).

        CIRE = (R750 - R800) / (R695 - R740) - 1

        Reference: Gitelson et al. (2003)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            CIRE values
        """
        r750 = self._get_band(image, 750)
        r800 = self._get_band(image, 800)
        r695 = self._get_band(image, 695)
        r740 = self._get_band(image, 740)

        return self._safe_divide(r750 - r800, r695 - r740) - 1

    def hrei15(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Red Edge Index 15 (HREI15).

        HREI15 = (R855 - R720) / (R855 + R720)

        Reference: Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HREI15 values
        """
        r855 = self._get_band(image, 855)
        r720 = self._get_band(image, 720)
        return self._safe_divide(r855 - r720, r855 + r720)

    def hrei16(self, image: np.ndarray) -> np.ndarray:
        """
        Hyperspectral Red Edge Index 16 (HREI16).

        HREI16 = (R910 - R705) / (R910 + R705)

        Reference: Thenkabail et al. (2014)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            HREI16 values
        """
        r910 = self._get_band(image, 910)
        r705 = self._get_band(image, 705)
        return self._safe_divide(r910 - r705, r910 + r705)

    def reip(self, image: np.ndarray) -> np.ndarray:
        """
        Red Edge Inflection Point (REIP).

        Linear interpolation method to find the inflection point
        in the red-edge region (680-750nm).

        REIP = 700 + 40 * ((R670 + R780)/2 - R700) / (R740 - R700)

        Reference: Guyot & Baret (1988)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            REIP values (wavelength in nm)
        """
        r670 = self._get_band(image, 670)
        r700 = self._get_band(image, 700)
        r740 = self._get_band(image, 740)
        r780 = self._get_band(image, 780)

        midpoint = (r670 + r780) / 2
        return 700 + 40 * self._safe_divide(midpoint - r700, r740 - r700)

    # =========================================================================
    # CROP-SPECIFIC INDICES
    # =========================================================================

    def lci(self, image: np.ndarray) -> np.ndarray:
        """
        Leaf Chlorophyll Index (LCI) - suitable for all crops.

        LCI = (R850 - R710) / (R850 + R680)

        Reference: Datt (1999)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            LCI values
        """
        r850 = self._get_band(image, 850)
        r710 = self._get_band(image, 710)
        r680 = self._get_band(image, 680)
        return self._safe_divide(r850 - r710, r850 + r680)

    def gndvi(self, image: np.ndarray) -> np.ndarray:
        """
        Green Normalized Difference Vegetation Index (GNDVI).

        GNDVI = (R780 - R550) / (R780 + R550)

        More sensitive to chlorophyll concentration than NDVI.
        Suitable for rice, wheat, and maize.

        Reference: Gitelson et al. (1996)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            GNDVI values
        """
        r780 = self._get_band(image, 780)
        r550 = self._get_band(image, 550)
        return self._safe_divide(r780 - r550, r780 + r550)

    def mtci(self, image: np.ndarray) -> np.ndarray:
        """
        MERIS Terrestrial Chlorophyll Index (MTCI).

        MTCI = (R754 - R709) / (R709 - R681)

        Particularly effective for wheat and rice.
        Reference: Dash & Curran (2004)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            MTCI values
        """
        r754 = self._get_band(image, 754)
        r709 = self._get_band(image, 709)
        r681 = self._get_band(image, 681)
        return self._safe_divide(r754 - r709, r709 - r681)

    def nli(self, image: np.ndarray) -> np.ndarray:
        """
        Non-Linear Index (NLI).

        NLI = (R800^2 - R680) / (R800^2 + R680)

        Reference: Goel & Qin (1994)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            NLI values
        """
        r800 = self._get_band(image, 800)
        r680 = self._get_band(image, 680)
        return self._safe_divide(r800**2 - r680, r800**2 + r680)

    def pssr_a(self, image: np.ndarray) -> np.ndarray:
        """
        Pigment Specific Simple Ratio for chlorophyll a (PSSRa).

        PSSRa = R800 / R680

        Reference: Blackburn (1998)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            PSSRa values
        """
        r800 = self._get_band(image, 800)
        r680 = self._get_band(image, 680)
        return self._safe_divide(r800, r680)

    def pssr_b(self, image: np.ndarray) -> np.ndarray:
        """
        Pigment Specific Simple Ratio for chlorophyll b (PSSRb).

        PSSRb = R800 / R635

        Reference: Blackburn (1998)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            PSSRb values
        """
        r800 = self._get_band(image, 800)
        r635 = self._get_band(image, 635)
        return self._safe_divide(r800, r635)

    # =========================================================================
    # RICE-SPECIFIC INDICES
    # =========================================================================

    def rice_ndvi_green(self, image: np.ndarray) -> np.ndarray:
        """
        Green NDVI optimized for rice nitrogen estimation.

        Uses green band instead of red for better N sensitivity in rice.

        Reference: Xue et al. (2004)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            Green NDVI values
        """
        r810 = self._get_band(image, 810)
        r560 = self._get_band(image, 560)
        return self._safe_divide(r810 - r560, r810 + r560)

    def rice_nri(self, image: np.ndarray) -> np.ndarray:
        """
        Nitrogen Reflectance Index for Rice (NRI).

        NRI = (R570 - R670) / (R570 + R670)

        Reference: Tian et al. (2011)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            NRI values
        """
        r570 = self._get_band(image, 570)
        r670 = self._get_band(image, 670)
        return self._safe_divide(r570 - r670, r570 + r670)

    # =========================================================================
    # WHEAT-SPECIFIC INDICES
    # =========================================================================

    def wheat_nsi(self, image: np.ndarray) -> np.ndarray:
        """
        Nitrogen Sufficiency Index for Wheat (NSI).

        Based on NDRE, optimized for wheat nitrogen status.

        Reference: Raun et al. (2005)

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            NSI values
        """
        return self.ndre(image)

    def wheat_canopy_chlorophyll(self, image: np.ndarray) -> np.ndarray:
        """
        Canopy Chlorophyll Content Index for Wheat.

        CCCI = NDRE / NDVI

        Reference: Barnes et al. (2000), adapted for wheat

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        np.ndarray
            CCCI values
        """
        ndre_val = self.ndre(image)
        ndvi_val = self.ndvi(image)
        return self._safe_divide(ndre_val, ndvi_val)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def calculate_all_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all available indices for the given image.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping index names to their computed arrays
        """
        indices = {}

        index_methods = [
            # Biomass indices
            ('NDVI', self.ndvi),
            ('MSAVI', self.msavi),
            ('RTVI', self.rtvi),
            # Structural indices
            ('HBSI1', self.hbsi1),
            ('HBSI2', self.hbsi2),
            ('HBSI3', self.hbsi3),
            ('PSRI', self.psri),
            # Biochemical indices
            ('HBCI8', self.hbci8),
            ('HBCI9', self.hbci9),
            ('HBCI10', self.hbci10),
            ('MCARI', self.mcari),
            ('TCARI', self.tcari),
            ('OSAVI', self.osavi),
            ('TCARI_OSAVI', self.tcari_osavi),
            ('DCNI', self.dcni),
            ('RVI_II', self.rvi_ii),
            # Red-edge indices
            ('NDRE', self.ndre),
            ('CIRE', self.cire),
            ('HREI15', self.hrei15),
            ('HREI16', self.hrei16),
            ('REIP', self.reip),
            # Additional indices
            ('LCI', self.lci),
            ('GNDVI', self.gndvi),
            ('MTCI', self.mtci),
            ('NLI', self.nli),
            ('PSSRa', self.pssr_a),
            ('PSSRb', self.pssr_b),
        ]

        for name, method in index_methods:
            try:
                indices[name] = method(image)
            except ValueError as e:
                print(f"Warning: Could not calculate {name}: {e}")
                indices[name] = None

        return indices

    def get_maize_nce_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get indices optimized for maize NCE (Nitrogen Conversion Efficiency).

        Based on Olson et al. (2022) findings.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of NCE-relevant indices
        """
        return {
            'HBSI1': self.hbsi1(image),
            'HBSI2': self.hbsi2(image),
            'HBSI3': self.hbsi3(image),
            'NDVI': self.ndvi(image),
        }

    def get_maize_nie_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get indices optimized for maize NIE (Nitrogen Internal Efficiency).

        Based on Olson et al. (2022) findings.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of NIE-relevant indices
        """
        return {
            'HBCI8': self.hbci8(image),
            'HBCI9': self.hbci9(image),
            'HBCI10': self.hbci10(image),
            'GNDVI': self.gndvi(image),
        }

    def get_rice_n_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get indices optimized for rice nitrogen estimation.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of rice N-relevant indices
        """
        return {
            'RVI_II': self.rvi_ii(image),
            'Rice_NRI': self.rice_nri(image),
            'Rice_GNDVI': self.rice_ndvi_green(image),
            'NDRE': self.ndre(image),
            'MTCI': self.mtci(image),
            'GNDVI': self.gndvi(image),
        }

    def get_wheat_n_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get indices optimized for wheat nitrogen estimation.

        Parameters
        ----------
        image : np.ndarray
            Hyperspectral image (bands, rows, cols)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of wheat N-relevant indices
        """
        return {
            'NDRE': self.ndre(image),
            'Wheat_NSI': self.wheat_nsi(image),
            'Wheat_CCCI': self.wheat_canopy_chlorophyll(image),
            'MTCI': self.mtci(image),
            'TCARI_OSAVI': self.tcari_osavi(image),
            'GNDVI': self.gndvi(image),
        }
