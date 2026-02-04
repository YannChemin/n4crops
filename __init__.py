"""
N4Crops: Nitrogen Content Estimation from Hyperspectral Remote Sensing
======================================================================

A Python-GDAL based toolkit for estimating nitrogen content and nitrogen use
efficiency in major cereal crops (Maize, Rice, Wheat) using hyperspectral
vegetation indices.

References:
-----------
1. Olson, M.B., Crawford, M.M., Vyn, T.J. (2022). Hyperspectral Indices for
   Predicting Nitrogen Use Efficiency in Maize Hybrids. Remote Sens. 14, 1721.

2. Thenkabail, P.S., et al. (2014). Hyperspectral remote sensing of vegetation
   and agricultural crops. Photogramm. Eng. Remote Sens. 80, 697-723.

3. Gitelson, A.A., et al. (2003). Relationships between leaf chlorophyll content
   and spectral reflectance. J. Plant Physiol. 160, 271-282.

4. Chen, P., et al. (2010). New spectral indicator assessing the efficiency of
   crop nitrogen treatment. Remote Sens. Environ. 114, 1987-1997.

5. Xue, L., et al. (2004). Monitoring leaf nitrogen status in rice with canopy
   spectral reflectance. Agron. J. 96, 135-142.

Author: Generated for N4Maize project
License: MIT
"""

__version__ = "1.0.0"
__author__ = "N4Maize Project"

from .indices import HyperspectralIndices
from .processor import NitrogenProcessor
from .calibration import CropCalibration
from .utils import read_hyperspectral, write_raster

__all__ = [
    'HyperspectralIndices',
    'NitrogenProcessor',
    'CropCalibration',
    'read_hyperspectral',
    'write_raster'
]
