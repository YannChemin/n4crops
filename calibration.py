"""
Crop Calibration Models for Nitrogen Estimation
================================================

This module provides calibration models for converting vegetation indices
to nitrogen parameters for maize, rice, and wheat.

Models are derived from peer-reviewed literature including:
- Olson et al. (2022) - Maize NCE and NIE
- Thenkabail et al. (2014) - General hyperspectral indices
- Xue et al. (2004) - Rice nitrogen
- Tian et al. (2011) - Rice N accumulation
- Zhao et al. (2018) - Wheat and maize N nutrition index
- Hansen & Schjoerring (2003) - Wheat nitrogen

References:
-----------
[1] Olson, M.B., et al. (2022). Remote Sens. 14, 1721.
[2] Thenkabail, P.S., et al. (2014). Photogramm. Eng. Remote Sens. 80, 697-723.
[3] Xue, L., et al. (2004). Agron. J. 96, 135-142.
[4] Tian, Y., et al. (2011). Field Crops Res. 120, 299-310.
[5] Zhao, B., et al. (2018). Eur. J. Agron. 93, 113-125.
[6] Hansen, P.M., Schjoerring, J.K. (2003). Remote Sens. Environ. 86, 542-553.
[7] Feng, W., et al. (2016). Field Crops Res. 186, 15-27.
"""

import numpy as np
from typing import Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass


class CropType(Enum):
    """Supported crop types."""
    MAIZE = "maize"
    RICE = "rice"
    WHEAT = "wheat"


class GrowthStage(Enum):
    """Growth stages for model selection."""
    # Maize stages
    V12 = "V12"
    V16 = "V16"
    V18 = "V18"
    R1 = "R1"
    R2 = "R2"
    R6 = "R6"
    # Rice stages
    TILLERING = "tillering"
    BOOTING = "booting"
    HEADING = "heading"
    FILLING = "filling"
    # Wheat stages
    JOINTING = "jointing"
    HEADING_WHEAT = "heading_wheat"
    ANTHESIS = "anthesis"
    GRAIN_FILL = "grain_fill"
    # Generic stages
    VEGETATIVE = "vegetative"
    REPRODUCTIVE = "reproductive"


@dataclass
class CalibrationModel:
    """
    Calibration model definition.

    Attributes
    ----------
    index : str
        Name of the vegetation index used
    model_type : str
        Type of model ('linear', 'polynomial', 'exponential', 'power')
    coefficients : dict
        Model coefficients (varies by model type)
    r_squared : float
        Coefficient of determination from calibration
    rmse : float
        Root mean square error from calibration
    units : str
        Units of the output parameter
    reference : str
        Literature reference for the model
    valid_range : tuple
        Valid range for the input index
    """
    index: str
    model_type: str
    coefficients: Dict[str, float]
    r_squared: float
    rmse: float
    units: str
    reference: str
    valid_range: tuple = (-1.0, 1.0)


class CropCalibration:
    """
    Crop-specific calibration models for nitrogen estimation.

    This class provides calibration equations for converting vegetation
    indices to nitrogen parameters (NCE, NIE, pN, leaf N, etc.) for
    maize, rice, and wheat.

    Example
    -------
    >>> calib = CropCalibration()
    >>> models = calib.get_models(CropType.MAIZE, "R1")
    >>> nce = calib.apply_model(hbsi1_array, models['NCE'])
    """

    def __init__(self):
        """Initialize calibration models."""
        self._models = self._initialize_models()

    def _initialize_models(self) -> Dict:
        """
        Initialize all calibration models from literature.

        Returns dictionary structured as:
        {crop_type: {growth_stage: {parameter: CalibrationModel}}}
        """
        models = {
            CropType.MAIZE: self._get_maize_models(),
            CropType.RICE: self._get_rice_models(),
            CropType.WHEAT: self._get_wheat_models(),
        }
        return models

    def _get_maize_models(self) -> Dict:
        """
        Get calibration models for maize.

        Based primarily on Olson et al. (2022) findings:
        - HBSI1 and HBSI2 best for NCE at R1
        - HBCI8 and HBCI9 best for NIE at V16 and R1
        """
        return {
            # V16 stage models
            'V16': {
                'NIE': {
                    'index': 'HBCI8',
                    'model_type': 'linear',
                    'coefficients': {'slope': 2.64, 'intercept': 50.0},
                    'r_squared': 0.72,
                    'rmse': 5.2,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (40.0, 60.0),
                },
                'pN': {
                    'index': 'HBCI8',
                    'model_type': 'linear',
                    'coefficients': {'slope': 0.054, 'intercept': 0.94},
                    'r_squared': 0.60,
                    'rmse': 0.06,
                    'units': '%',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (0.8, 1.2),
                },
            },
            # R1 stage models (most validated)
            'R1': {
                'NCE': {
                    'index': 'HBSI1',
                    'model_type': 'linear',
                    'coefficients': {'slope': 107.0, 'intercept': 0.0},
                    'r_squared': 0.67,
                    'rmse': 14.4,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (60.0, 150.0),
                },
                'NCE_HBSI2': {
                    'index': 'HBSI2',
                    'model_type': 'linear',
                    'coefficients': {'slope': 107.0, 'intercept': 0.0},
                    'r_squared': 0.68,
                    'rmse': 14.4,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (60.0, 150.0),
                },
                'NIE': {
                    'index': 'HBCI8',
                    'model_type': 'linear',
                    'coefficients': {'slope': 52.0, 'intercept': 0.0},
                    'r_squared': 0.67,
                    'rmse': 5.0,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (20.0, 40.0),
                },
                'NIE_HBCI9': {
                    'index': 'HBCI9',
                    'model_type': 'linear',
                    'coefficients': {'slope': 55.0, 'intercept': 0.0},
                    'r_squared': 0.84,
                    'rmse': 5.0,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (20.0, 45.0),
                },
                'pN': {
                    'index': 'HREI16',
                    'model_type': 'linear',
                    'coefficients': {'slope': 0.9, 'intercept': 0.0},
                    'r_squared': 0.72,
                    'rmse': 0.11,
                    'units': '%',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (0.8, 2.5),
                },
            },
            # R2 stage models
            'R2': {
                'NCE': {
                    'index': 'HBSI2',
                    'model_type': 'linear',
                    'coefficients': {'slope': 5.0e-6, 'intercept': 105.0},
                    'r_squared': 0.65,
                    'rmse': 15.0,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (100.0, 110.0),
                },
                'NIE': {
                    'index': 'HBCI9',
                    'model_type': 'linear',
                    'coefficients': {'slope': 4.0, 'intercept': 50.0},
                    'r_squared': 0.65,
                    'rmse': 5.5,
                    'units': 'kg kg⁻¹ N',
                    'reference': 'Olson et al. (2022)',
                    'valid_range': (45.0, 55.0),
                },
            },
            # Generic vegetative stage
            'vegetative': {
                'leaf_N': {
                    'index': 'NDRE',
                    'model_type': 'polynomial',
                    'coefficients': {'a2': -15.2, 'a1': 18.5, 'a0': 1.2},
                    'r_squared': 0.75,
                    'rmse': 0.3,
                    'units': '%',
                    'reference': 'Zhao et al. (2018)',
                    'valid_range': (0.1, 0.6),
                },
                'NNI': {
                    'index': 'MTCI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 0.15, 'intercept': 0.3},
                    'r_squared': 0.72,
                    'rmse': 0.12,
                    'units': 'unitless',
                    'reference': 'Zhao et al. (2018)',
                    'valid_range': (1.0, 6.0),
                },
            },
        }

    def _get_rice_models(self) -> Dict:
        """
        Get calibration models for rice.

        Based on literature from:
        - Xue et al. (2004) - RVI II for rice N
        - Tian et al. (2011) - Rice N accumulation
        - Various studies on NDRE and GNDVI for rice
        """
        return {
            'tillering': {
                'leaf_N': {
                    'index': 'RVI_II',
                    'model_type': 'power',
                    'coefficients': {'a': 1.52, 'b': 0.48},
                    'r_squared': 0.82,
                    'rmse': 0.25,
                    'units': '%',
                    'reference': 'Xue et al. (2004)',
                    'valid_range': (0.05, 5.0),
                },
                'plant_N': {
                    'index': 'GNDVI',
                    'model_type': 'exponential',
                    'coefficients': {'a': 0.5, 'b': 4.2},
                    'r_squared': 0.78,
                    'rmse': 0.18,
                    'units': '%',
                    'reference': 'Tian et al. (2011)',
                    'valid_range': (0.3, 0.8),
                },
            },
            'booting': {
                'leaf_N': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 8.5, 'intercept': 1.0},
                    'r_squared': 0.80,
                    'rmse': 0.22,
                    'units': '%',
                    'reference': 'Tian et al. (2011)',
                    'valid_range': (1.5, 4.5),
                },
                'N_uptake': {
                    'index': 'MTCI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 25.0, 'intercept': -15.0},
                    'r_squared': 0.76,
                    'rmse': 12.0,
                    'units': 'kg ha⁻¹',
                    'reference': 'Tian et al. (2011)',
                    'valid_range': (1.5, 5.5),
                },
            },
            'heading': {
                'leaf_N': {
                    'index': 'Rice_NRI',
                    'model_type': 'linear',
                    'coefficients': {'slope': -12.0, 'intercept': 2.8},
                    'r_squared': 0.75,
                    'rmse': 0.28,
                    'units': '%',
                    'reference': 'Tian et al. (2011)',
                    'valid_range': (1.0, 3.0),
                },
                'grain_N': {
                    'index': 'GNDVI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 2.5, 'intercept': 0.5},
                    'r_squared': 0.70,
                    'rmse': 0.15,
                    'units': '%',
                    'reference': 'Estimated from literature',
                    'valid_range': (1.0, 2.0),
                },
            },
            'filling': {
                'grain_protein': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 20.0, 'intercept': 4.0},
                    'r_squared': 0.72,
                    'rmse': 0.8,
                    'units': '%',
                    'reference': 'Estimated from literature',
                    'valid_range': (6.0, 12.0),
                },
            },
            # Generic stages
            'vegetative': {
                'leaf_N': {
                    'index': 'Rice_GNDVI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 6.0, 'intercept': 0.8},
                    'r_squared': 0.78,
                    'rmse': 0.25,
                    'units': '%',
                    'reference': 'Xue et al. (2004)',
                    'valid_range': (0.3, 0.8),
                },
            },
            'reproductive': {
                'N_content': {
                    'index': 'MTCI',
                    'model_type': 'polynomial',
                    'coefficients': {'a2': 0.5, 'a1': 3.0, 'a0': 1.0},
                    'r_squared': 0.74,
                    'rmse': 0.20,
                    'units': '%',
                    'reference': 'Tian et al. (2011)',
                    'valid_range': (1.0, 6.0),
                },
            },
        }

    def _get_wheat_models(self) -> Dict:
        """
        Get calibration models for wheat.

        Based on literature from:
        - Hansen & Schjoerring (2003) - Wheat N estimation
        - Feng et al. (2016) - Wheat N monitoring
        - Zhao et al. (2018) - NNI estimation
        """
        return {
            'jointing': {
                'leaf_N': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 10.5, 'intercept': 0.8},
                    'r_squared': 0.85,
                    'rmse': 0.20,
                    'units': '%',
                    'reference': 'Hansen & Schjoerring (2003)',
                    'valid_range': (0.15, 0.55),
                },
                'N_uptake': {
                    'index': 'TCARI_OSAVI',
                    'model_type': 'exponential',
                    'coefficients': {'a': 280.0, 'b': -3.5},
                    'r_squared': 0.80,
                    'rmse': 18.0,
                    'units': 'kg ha⁻¹',
                    'reference': 'Feng et al. (2016)',
                    'valid_range': (10.0, 200.0),
                },
                'NNI': {
                    'index': 'Wheat_CCCI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 1.8, 'intercept': 0.2},
                    'r_squared': 0.78,
                    'rmse': 0.10,
                    'units': 'unitless',
                    'reference': 'Zhao et al. (2018)',
                    'valid_range': (0.3, 0.9),
                },
            },
            'heading_wheat': {
                'leaf_N': {
                    'index': 'Wheat_NSI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 9.0, 'intercept': 1.2},
                    'r_squared': 0.82,
                    'rmse': 0.22,
                    'units': '%',
                    'reference': 'Hansen & Schjoerring (2003)',
                    'valid_range': (0.15, 0.50),
                },
                'plant_N': {
                    'index': 'GNDVI',
                    'model_type': 'polynomial',
                    'coefficients': {'a2': -5.0, 'a1': 8.0, 'a0': 0.5},
                    'r_squared': 0.77,
                    'rmse': 0.18,
                    'units': '%',
                    'reference': 'Feng et al. (2016)',
                    'valid_range': (0.35, 0.80),
                },
            },
            'anthesis': {
                'flag_leaf_N': {
                    'index': 'MTCI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 0.65, 'intercept': 0.8},
                    'r_squared': 0.80,
                    'rmse': 0.25,
                    'units': '%',
                    'reference': 'Feng et al. (2016)',
                    'valid_range': (1.5, 5.0),
                },
                'grain_protein_pred': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 25.0, 'intercept': 6.0},
                    'r_squared': 0.70,
                    'rmse': 1.0,
                    'units': '%',
                    'reference': 'Hansen & Schjoerring (2003)',
                    'valid_range': (8.0, 15.0),
                },
            },
            'grain_fill': {
                'grain_N': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 5.5, 'intercept': 1.0},
                    'r_squared': 0.72,
                    'rmse': 0.20,
                    'units': '%',
                    'reference': 'Feng et al. (2016)',
                    'valid_range': (1.0, 2.0),
                },
            },
            # Generic stages
            'vegetative': {
                'leaf_N': {
                    'index': 'NDRE',
                    'model_type': 'linear',
                    'coefficients': {'slope': 10.0, 'intercept': 1.0},
                    'r_squared': 0.82,
                    'rmse': 0.22,
                    'units': '%',
                    'reference': 'Hansen & Schjoerring (2003)',
                    'valid_range': (0.15, 0.55),
                },
            },
            'reproductive': {
                'plant_N': {
                    'index': 'GNDVI',
                    'model_type': 'linear',
                    'coefficients': {'slope': 4.5, 'intercept': 0.8},
                    'r_squared': 0.75,
                    'rmse': 0.20,
                    'units': '%',
                    'reference': 'Feng et al. (2016)',
                    'valid_range': (0.35, 0.75),
                },
            },
        }

    def get_models(self, crop_type: CropType,
                   growth_stage: str) -> Dict[str, Dict]:
        """
        Get calibration models for a specific crop and growth stage.

        Parameters
        ----------
        crop_type : CropType
            Crop type (MAIZE, RICE, WHEAT)
        growth_stage : str
            Growth stage identifier

        Returns
        -------
        Dict[str, Dict]
            Dictionary of available models for the crop/stage
        """
        if crop_type not in self._models:
            raise ValueError(f"Unknown crop type: {crop_type}")

        crop_models = self._models[crop_type]

        # Try exact stage match
        if growth_stage in crop_models:
            return crop_models[growth_stage]

        # Try lowercase
        if growth_stage.lower() in crop_models:
            return crop_models[growth_stage.lower()]

        # Fall back to generic stages
        if growth_stage in ['V12', 'V16', 'V18', 'tillering', 'jointing']:
            if 'vegetative' in crop_models:
                return crop_models['vegetative']
        elif growth_stage in ['R1', 'R2', 'R6', 'booting', 'heading',
                              'heading_wheat', 'anthesis', 'filling', 'grain_fill']:
            if 'reproductive' in crop_models:
                return crop_models['reproductive']

        raise ValueError(
            f"No models available for {crop_type.value} at stage {growth_stage}. "
            f"Available stages: {list(crop_models.keys())}"
        )

    def apply_model(self, index_values: np.ndarray,
                    model_info: Dict) -> np.ndarray:
        """
        Apply a calibration model to vegetation index values.

        Parameters
        ----------
        index_values : np.ndarray
            Array of vegetation index values
        model_info : Dict
            Model information dictionary

        Returns
        -------
        np.ndarray
            Estimated nitrogen parameter values
        """
        model_type = model_info['model_type']
        coeffs = model_info['coefficients']
        valid_range = model_info.get('valid_range', (-np.inf, np.inf))

        # Apply model
        if model_type == 'linear':
            # y = slope * x + intercept
            result = coeffs['slope'] * index_values + coeffs['intercept']

        elif model_type == 'polynomial':
            # y = a2*x^2 + a1*x + a0
            result = (coeffs.get('a2', 0) * index_values**2 +
                      coeffs.get('a1', 0) * index_values +
                      coeffs.get('a0', 0))

        elif model_type == 'exponential':
            # y = a * exp(b * x)
            result = coeffs['a'] * np.exp(coeffs['b'] * index_values)

        elif model_type == 'power':
            # y = a * x^b
            result = coeffs['a'] * np.power(np.maximum(index_values, 0.001), coeffs['b'])

        elif model_type == 'logarithmic':
            # y = a * ln(x) + b
            result = coeffs['a'] * np.log(np.maximum(index_values, 0.001)) + coeffs['b']

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Clip result to valid range (applied to output, not input)
        result = np.clip(result, valid_range[0], valid_range[1])

        return result

    def get_model_info(self, crop_type: CropType, growth_stage: str,
                       parameter: str) -> Optional[Dict]:
        """
        Get detailed information about a specific model.

        Parameters
        ----------
        crop_type : CropType
            Crop type
        growth_stage : str
            Growth stage
        parameter : str
            Nitrogen parameter name

        Returns
        -------
        Optional[Dict]
            Model information or None if not found
        """
        try:
            models = self.get_models(crop_type, growth_stage)
            return models.get(parameter)
        except ValueError:
            return None

    def list_available_models(self) -> Dict[str, Dict[str, list]]:
        """
        List all available calibration models.

        Returns
        -------
        Dict
            Nested dictionary of crop -> stage -> parameters
        """
        available = {}
        for crop_type, stages in self._models.items():
            available[crop_type.value] = {}
            for stage, params in stages.items():
                available[crop_type.value][stage] = list(params.keys())
        return available

    def add_custom_model(self, crop_type: CropType, growth_stage: str,
                         parameter: str, model_info: Dict) -> None:
        """
        Add a custom calibration model.

        Parameters
        ----------
        crop_type : CropType
            Crop type
        growth_stage : str
            Growth stage
        parameter : str
            Nitrogen parameter name
        model_info : Dict
            Model specification dictionary

        Example
        -------
        >>> calib = CropCalibration()
        >>> calib.add_custom_model(
        ...     CropType.MAIZE, 'V14', 'custom_N',
        ...     {
        ...         'index': 'NDVI',
        ...         'model_type': 'linear',
        ...         'coefficients': {'slope': 5.0, 'intercept': 1.0},
        ...         'r_squared': 0.75,
        ...         'rmse': 0.2,
        ...         'units': '%',
        ...         'reference': 'Custom calibration',
        ...     }
        ... )
        """
        if crop_type not in self._models:
            self._models[crop_type] = {}

        if growth_stage not in self._models[crop_type]:
            self._models[crop_type][growth_stage] = {}

        # Validate required fields
        required_fields = ['index', 'model_type', 'coefficients']
        for field in required_fields:
            if field not in model_info:
                raise ValueError(f"Model info missing required field: {field}")

        self._models[crop_type][growth_stage][parameter] = model_info


def create_calibration_report(calib: CropCalibration,
                              crop_type: CropType,
                              growth_stage: str) -> str:
    """
    Create a text report of calibration models.

    Parameters
    ----------
    calib : CropCalibration
        Calibration object
    crop_type : CropType
        Crop type
    growth_stage : str
        Growth stage

    Returns
    -------
    str
        Formatted report string
    """
    try:
        models = calib.get_models(crop_type, growth_stage)
    except ValueError as e:
        return str(e)

    lines = [
        f"Calibration Models for {crop_type.value.title()} at {growth_stage}",
        "=" * 60,
        ""
    ]

    for param_name, model_info in models.items():
        lines.append(f"Parameter: {param_name}")
        lines.append(f"  Index: {model_info['index']}")
        lines.append(f"  Model: {model_info['model_type']}")
        lines.append(f"  Coefficients: {model_info['coefficients']}")
        lines.append(f"  R²: {model_info.get('r_squared', 'N/A')}")
        lines.append(f"  RMSE: {model_info.get('rmse', 'N/A')} {model_info.get('units', '')}")
        lines.append(f"  Reference: {model_info.get('reference', 'N/A')}")
        lines.append("")

    return "\n".join(lines)
