"""
Feature Engineering Configuration Module

This module provides a declarative configuration for all engineered features
in the project. Instead of hardcoding feature engineering logic, all feature
definitions are centralized here for easy maintenance and modification.

Each engineered feature is defined with:
- name: The name of the new feature
- description: Human-readable description
- operation: Type of operation (ratio, log, sum, diversity, etc.)
- dependencies: List of World Bank indicator codes required
- parameters: Additional parameters for the operation
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


# Engineered feature definitions
ENGINEERED_FEATURES = [
    {
        'name': 'urbanization_rate',
        'description': 'Percentage of population living in urban areas',
        'operation': 'ratio_scaled',
        'dependencies': ['SP.URB.TOTL', 'SP.POP.TOTL'],
        'parameters': {
            'numerator': 'SP.URB.TOTL',
            'denominator': 'SP.POP.TOTL',
            'scale': 100
        }
    },
    {
        'name': 'log_gdp_per_capita',
        'description': 'Natural log of GDP per capita (current USD)',
        'operation': 'log1p',
        'dependencies': ['NY.GDP.PCAP.CD'],
        'parameters': {
            'source': 'NY.GDP.PCAP.CD'
        }
    },
    {
        'name': 'trade_openness',
        'description': 'Sum of exports and imports as % of GDP',
        'operation': 'sum',
        'dependencies': ['NE.EXP.GNFS.ZS', 'NE.IMP.GNFS.ZS'],
        'parameters': {
            'columns': ['NE.EXP.GNFS.ZS', 'NE.IMP.GNFS.ZS']
        }
    },
    {
        'name': 'health_to_edu_ratio',
        'description': 'Ratio of health expenditure to education expenditure',
        'operation': 'ratio',
        'dependencies': ['SH.XPD.CHEX.GD.ZS', 'SE.XPD.TOTL.GD.ZS'],
        'parameters': {
            'numerator': 'SH.XPD.CHEX.GD.ZS',
            'denominator': 'SE.XPD.TOTL.GD.ZS'
        }
    },
    {
        'name': 'gender_labor_gap',
        'description': 'Difference between male and female labor force participation',
        'operation': 'difference',
        'dependencies': ['SL.TLF.CACT.MA.ZS', 'SL.TLF.CACT.FE.ZS'],
        'parameters': {
            'minuend': 'SL.TLF.CACT.MA.ZS',
            'subtrahend': 'SL.TLF.CACT.FE.ZS'
        }
    },
    {
        'name': 'economic_diversity',
        'description': 'Economic sector diversity index (inverse Herfindahl)',
        'operation': 'diversity',
        'dependencies': ['NV.AGR.TOTL.ZS', 'NV.IND.TOTL.ZS', 'NV.SRV.TOTL.ZS'],
        'parameters': {
            'columns': ['NV.AGR.TOTL.ZS', 'NV.IND.TOTL.ZS', 'NV.SRV.TOTL.ZS']
        }
    }
]


def check_dependencies(data: pd.DataFrame, feature_config: Dict[str, Any]) -> bool:
    """
    Check if all required dependencies for a feature are present in the dataset.

    Args:
        data: DataFrame containing the raw data
        feature_config: Feature configuration dictionary

    Returns:
        True if all dependencies are present, False otherwise
    """
    dependencies = feature_config.get('dependencies', [])
    return all(dep in data.columns for dep in dependencies)


def apply_ratio_scaled(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Calculate a ratio and scale it.

    Args:
        data: DataFrame containing the source columns
        params: Parameters including 'numerator', 'denominator', and 'scale'

    Returns:
        Series containing the scaled ratio
    """
    numerator = params['numerator']
    denominator = params['denominator']
    scale = params.get('scale', 1)

    return (data[numerator] / data[denominator]) * scale


def apply_log1p(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Apply log(1 + x) transformation.

    Args:
        data: DataFrame containing the source column
        params: Parameters including 'source' column name

    Returns:
        Series containing the log-transformed values
    """
    source = params['source']
    return np.log1p(data[source])


def apply_sum(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Sum multiple columns.

    Args:
        data: DataFrame containing the source columns
        params: Parameters including 'columns' list

    Returns:
        Series containing the sum
    """
    columns = params['columns']
    return data[columns].sum(axis=1)


def apply_ratio(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Calculate a simple ratio.

    Args:
        data: DataFrame containing the source columns
        params: Parameters including 'numerator' and 'denominator'

    Returns:
        Series containing the ratio
    """
    numerator = params['numerator']
    denominator = params['denominator']

    return data[numerator] / data[denominator]


def apply_difference(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Calculate difference between two columns.

    Args:
        data: DataFrame containing the source columns
        params: Parameters including 'minuend' and 'subtrahend'

    Returns:
        Series containing the difference
    """
    minuend = params['minuend']
    subtrahend = params['subtrahend']

    return data[minuend] - data[subtrahend]


def apply_diversity(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Calculate diversity index (1 - Herfindahl index).

    The Herfindahl index is the sum of squared proportions.
    Diversity is 1 minus this, measuring how evenly distributed values are.

    Args:
        data: DataFrame containing the source columns
        params: Parameters including 'columns' list

    Returns:
        Series containing the diversity index
    """
    columns = params['columns']

    # Normalize to get proportions
    total = data[columns].sum(axis=1)
    proportions = data[columns].div(total, axis=0)

    # Calculate Herfindahl index (sum of squared proportions)
    herfindahl = (proportions ** 2).sum(axis=1)

    # Diversity is 1 - Herfindahl
    return 1 - herfindahl


# Operation registry mapping operation names to functions
OPERATION_REGISTRY = {
    'ratio_scaled': apply_ratio_scaled,
    'log1p': apply_log1p,
    'sum': apply_sum,
    'ratio': apply_ratio,
    'difference': apply_difference,
    'diversity': apply_diversity
}


def create_engineered_feature(
    data: pd.DataFrame,
    feature_config: Dict[str, Any]
) -> Optional[pd.Series]:
    """
    Create an engineered feature based on its configuration.

    Args:
        data: DataFrame containing the raw data
        feature_config: Feature configuration dictionary

    Returns:
        Series containing the engineered feature, or None if dependencies missing
    """
    # Check if all dependencies are present
    if not check_dependencies(data, feature_config):
        return None

    # Get the operation function
    operation = feature_config['operation']
    operation_func = OPERATION_REGISTRY.get(operation)

    if operation_func is None:
        raise ValueError(f"Unknown operation: {operation}")

    # Apply the operation
    params = feature_config.get('parameters', {})
    return operation_func(data, params)


def create_all_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features defined in ENGINEERED_FEATURES.

    Args:
        data: DataFrame containing the raw data

    Returns:
        DataFrame with engineered features added
    """
    data_copy = data.copy()

    for feature_config in ENGINEERED_FEATURES:
        feature_name = feature_config['name']
        feature_series = create_engineered_feature(data_copy, feature_config)

        if feature_series is not None:
            data_copy[feature_name] = feature_series
            print(f"  ✓ Created feature: {feature_name}")
        else:
            missing_deps = [
                dep for dep in feature_config['dependencies']
                if dep not in data_copy.columns
            ]
            print(f"  ✗ Skipped feature '{feature_name}' - missing: {missing_deps}")

    return data_copy


def get_engineered_feature_names() -> List[str]:
    """
    Get a list of all engineered feature names.

    Returns:
        List of feature names
    """
    return [feature['name'] for feature in ENGINEERED_FEATURES]


def get_all_dependencies() -> List[str]:
    """
    Get a list of all World Bank indicator codes required for feature engineering.

    Returns:
        List of unique indicator codes
    """
    all_deps = []
    for feature in ENGINEERED_FEATURES:
        all_deps.extend(feature.get('dependencies', []))
    return list(set(all_deps))


def validate_feature_engineering_setup(available_indicators: List[str]) -> Dict[str, Any]:
    """
    Validate that all required indicators are available for feature engineering.

    Args:
        available_indicators: List of available World Bank indicator codes

    Returns:
        Dictionary with validation results
    """
    required_deps = get_all_dependencies()
    available_set = set(available_indicators)
    required_set = set(required_deps)

    missing = required_set - available_set
    available = required_set & available_set

    # Check which features can be created
    creatable_features = []
    uncreatable_features = []

    for feature in ENGINEERED_FEATURES:
        deps = set(feature['dependencies'])
        if deps.issubset(available_set):
            creatable_features.append(feature['name'])
        else:
            uncreatable_features.append({
                'name': feature['name'],
                'missing': list(deps - available_set)
            })

    return {
        'total_required': len(required_deps),
        'available_count': len(available),
        'missing_count': len(missing),
        'missing_indicators': list(missing),
        'creatable_features': creatable_features,
        'uncreatable_features': uncreatable_features
    }
