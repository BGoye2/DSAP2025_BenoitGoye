"""
Feature Category Registry Module

This module provides a centralized registry of feature categories for the project.
Instead of using fragile string matching to identify feature types, features are
explicitly categorized here.

Categories include:
- GDP and economic output
- Education
- Health
- Labor market
- Demographics
- Trade
- Government spending
- Infrastructure
- Engineered features
"""

from typing import Dict, List, Set


# Feature category mappings
FEATURE_CATEGORIES: Dict[str, List[str]] = {
    'gdp': [
        'NY.GDP.PCAP.CD',           # GDP per capita (current US$)
        'NY.GDP.PCAP.PP.CD',        # GDP per capita, PPP (current international $)
        'NY.GDP.MKTP.CD',           # GDP (current US$)
        'NY.GDP.MKTP.KD.ZG',        # GDP growth (annual %)
        'log_gdp_per_capita',       # Engineered: Log GDP per capita
    ],
    'education': [
        'SE.PRM.ENRL',              # Primary school enrollment
        'SE.PRM.ENRL.FE.ZS',        # Primary school enrollment, female (% gross)
        'SE.SEC.ENRL',              # Secondary school enrollment
        'SE.SEC.ENRL.FE.ZS',        # Secondary school enrollment, female (% gross)
        'SE.TER.ENRL',              # Tertiary school enrollment
        'SE.XPD.TOTL.GD.ZS',        # Government expenditure on education (% of GDP)
        'SE.PRM.CMPT.ZS',           # Primary completion rate
        'SE.ADT.LITR.ZS',           # Adult literacy rate
    ],
    'health': [
        'SH.XPD.CHEX.GD.ZS',        # Current health expenditure (% of GDP)
        'SH.XPD.CHEX.PC.CD',        # Current health expenditure per capita (current US$)
        'SH.DYN.MORT',              # Mortality rate, under-5 (per 1,000 live births)
        'SH.DYN.NMRT',              # Neonatal mortality rate (per 1,000 live births)
        'SP.DYN.LE00.IN',           # Life expectancy at birth
        'SH.DTH.COMM.ZS',           # Cause of death, by communicable diseases (% of total)
        'SH.STA.BASS.ZS',           # People with basic sanitation services (% of population)
        'health_to_edu_ratio',      # Engineered: Health to education expenditure ratio
    ],
    'labor': [
        'SL.TLF.CACT.ZS',           # Labor force participation rate
        'SL.TLF.CACT.FE.ZS',        # Labor force participation rate, female
        'SL.TLF.CACT.MA.ZS',        # Labor force participation rate, male
        'SL.UEM.TOTL.ZS',           # Unemployment rate
        'SL.UEM.TOTL.FE.ZS',        # Unemployment, female (% of female labor force)
        'SL.UEM.TOTL.MA.ZS',        # Unemployment, male (% of male labor force)
        'SL.AGR.EMPL.ZS',           # Employment in agriculture (% of total employment)
        'SL.IND.EMPL.ZS',           # Employment in industry (% of total employment)
        'SL.SRV.EMPL.ZS',           # Employment in services (% of total employment)
        'gender_labor_gap',         # Engineered: Gender gap in labor force participation
    ],
    'demographics': [
        'SP.POP.TOTL',              # Total population
        'SP.URB.TOTL',              # Urban population
        'SP.URB.TOTL.IN.ZS',        # Urban population (% of total)
        'SP.POP.GROW',              # Population growth (annual %)
        'SP.POP.65UP.TO.ZS',        # Population ages 65 and above (% of total)
        'SP.POP.0014.TO.ZS',        # Population ages 0-14 (% of total)
        'SP.DYN.TFRT.IN',           # Fertility rate
        'SP.DYN.CBRT.IN',           # Birth rate, crude (per 1,000 people)
        'SP.DYN.CDRT.IN',           # Death rate, crude (per 1,000 people)
        'urbanization_rate',        # Engineered: Urbanization rate
    ],
    'trade': [
        'NE.EXP.GNFS.ZS',           # Exports of goods and services (% of GDP)
        'NE.IMP.GNFS.ZS',           # Imports of goods and services (% of GDP)
        'BX.KLT.DINV.WD.GD.ZS',     # Foreign direct investment, net inflows (% of GDP)
        'BN.CAB.XOKA.GD.ZS',        # Current account balance (% of GDP)
        'trade_openness',           # Engineered: Trade openness (exports + imports)
    ],
    'government': [
        'GC.TAX.TOTL.GD.ZS',        # Tax revenue (% of GDP)
        'GC.XPN.TOTL.GD.ZS',        # Total government expenditure (% of GDP)
        'GC.DOD.TOTL.GD.ZS',        # Government debt (% of GDP)
        'MS.MIL.XPND.GD.ZS',        # Military expenditure (% of GDP)
    ],
    'infrastructure': [
        'EG.ELC.ACCS.ZS',           # Access to electricity (% of population)
        'IT.NET.USER.ZS',           # Internet users (% of population)
        'IT.CEL.SETS.P2',           # Mobile cellular subscriptions (per 100 people)
        'IS.ROD.PAVE.ZS',           # Roads, paved (% of total roads)
    ],
    'economic_structure': [
        'NV.AGR.TOTL.ZS',           # Agriculture, forestry, and fishing, value added (% of GDP)
        'NV.IND.TOTL.ZS',           # Industry (including construction), value added (% of GDP)
        'NV.SRV.TOTL.ZS',           # Services, value added (% of GDP)
        'NY.GDP.DEFL.KD.ZG',        # Inflation, GDP deflator (annual %)
        'FP.CPI.TOTL.ZG',           # Inflation, consumer prices (annual %)
        'economic_diversity',       # Engineered: Economic sector diversity index
    ],
    'income_inequality': [
        'SI.POV.GINI',              # GINI coefficient (TARGET VARIABLE)
        'SI.POV.NAHC',              # Poverty headcount ratio at national poverty lines
    ],
    'other': [
        'year_feature',             # Engineered: Year feature
    ]
}


def get_category_for_feature(feature_code: str) -> str:
    """
    Get the category for a given feature code.

    Args:
        feature_code: Feature or indicator code

    Returns:
        Category name, or 'unknown' if not found
    """
    for category, features in FEATURE_CATEGORIES.items():
        if feature_code in features:
            return category
    return 'unknown'


def get_features_by_category(category: str) -> List[str]:
    """
    Get all features in a specific category.

    Args:
        category: Category name

    Returns:
        List of feature codes in that category
    """
    return FEATURE_CATEGORIES.get(category, [])


def get_features_by_categories(categories: List[str]) -> List[str]:
    """
    Get all features from multiple categories.

    Args:
        categories: List of category names

    Returns:
        Combined list of feature codes from all categories
    """
    features = []
    for category in categories:
        features.extend(get_features_by_category(category))
    return features


def filter_features_by_category(
    feature_list: List[str],
    category: str
) -> List[str]:
    """
    Filter a list of features to only include those in a specific category.

    Args:
        feature_list: List of feature codes to filter
        category: Category to filter by

    Returns:
        Filtered list of features in the category
    """
    category_features = set(get_features_by_category(category))
    return [f for f in feature_list if f in category_features]


def filter_features_by_categories(
    feature_list: List[str],
    categories: List[str]
) -> List[str]:
    """
    Filter a list of features to only include those in specific categories.

    Args:
        feature_list: List of feature codes to filter
        categories: List of categories to filter by

    Returns:
        Filtered list of features in any of the categories
    """
    allowed_features = set(get_features_by_categories(categories))
    return [f for f in feature_list if f in allowed_features]


def get_all_categories() -> List[str]:
    """
    Get a list of all category names.

    Returns:
        List of category names
    """
    return list(FEATURE_CATEGORIES.keys())


def categorize_features(feature_list: List[str]) -> Dict[str, List[str]]:
    """
    Group a list of features by their categories.

    Args:
        feature_list: List of feature codes to categorize

    Returns:
        Dictionary mapping categories to lists of features
    """
    categorized = {category: [] for category in FEATURE_CATEGORIES.keys()}
    categorized['unknown'] = []

    for feature in feature_list:
        category = get_category_for_feature(feature)
        categorized[category].append(feature)

    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def validate_feature_coverage(feature_list: List[str]) -> Dict[str, any]:
    """
    Validate how well a feature list covers different categories.

    Args:
        feature_list: List of feature codes to validate

    Returns:
        Dictionary with coverage statistics
    """
    categorized = categorize_features(feature_list)

    total_registered = sum(len(features) for features in FEATURE_CATEGORIES.values())
    total_provided = len(feature_list)
    unknown_count = len(categorized.get('unknown', []))

    category_coverage = {}
    for category in FEATURE_CATEGORIES.keys():
        total_in_category = len(FEATURE_CATEGORIES[category])
        provided_in_category = len(categorized.get(category, []))
        coverage_pct = (provided_in_category / total_in_category * 100
                       if total_in_category > 0 else 0)
        category_coverage[category] = {
            'total': total_in_category,
            'provided': provided_in_category,
            'coverage_pct': coverage_pct
        }

    return {
        'total_registered_features': total_registered,
        'total_provided_features': total_provided,
        'unknown_features': categorized.get('unknown', []),
        'unknown_count': unknown_count,
        'category_coverage': category_coverage,
        'categorized_features': categorized
    }


# Convenience groupings for common use cases
CATEGORY_GROUPS = {
    'economic': ['gdp', 'economic_structure', 'trade'],
    'social': ['education', 'health', 'demographics'],
    'governance': ['government'],
    'development': ['infrastructure', 'education', 'health'],
    'all': list(FEATURE_CATEGORIES.keys())
}


def get_features_by_group(group: str) -> List[str]:
    """
    Get all features from a predefined category group.

    Args:
        group: Group name (e.g., 'economic', 'social', 'development')

    Returns:
        List of feature codes from all categories in the group
    """
    categories = CATEGORY_GROUPS.get(group, [])
    return get_features_by_categories(categories)
