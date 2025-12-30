"""
Feature Name Mapping Configuration
====================================
Centralized mapping of World Bank indicator codes and engineered features
to human-readable display names for tables, plots, and reports.

This ensures consistency across all outputs (tables, figures, reports).
"""

# =============================================================================
# FEATURE NAME MAPPING
# =============================================================================

FEATURE_NAME_MAPPING = {
    # Economic Indicators
    'NY.GDP.PCAP.CD': 'GDP per capita (current $)',
    'NY.GDP.MKTP.CD': 'GDP (current $)',
    'NY.GDP.PCAP.PP.CD': 'GDP per capita (PPP)',
    'NY.GDP.MKTP.KD.ZG': 'GDP growth rate',
    'NV.SRV.TOTL.ZS': 'Services (% GDP)',
    'NV.AGR.TOTL.ZS': 'Agriculture (% GDP)',
    'NV.IND.TOTL.ZS': 'Industry (% GDP)',

    # Population & Demographics
    'SP.POP.TOTL': 'Total population',
    'SP.POP.DPND': 'Dependency ratio',
    'SP.DYN.TFRT.IN': 'Fertility rate',
    'SP.URB.TOTL': 'Urban population',
    'SP.RUR.TOTL': 'Rural population',
    'SP.URB.GROW': 'Urban population growth',

    # Health Expenditure
    'SH.XPD.CHEX.GD.ZS': 'Health expenditure (% GDP)',
    'SH.XPD.GHED.GD.ZS': 'Gov health expenditure (% GDP)',
    'SH.XPD.CHEX.PC.CD': 'Health expenditure per capita',
    'SH.XPD.PVTD.CH.ZS': 'Private health expenditure (%)',

    # Education
    'SE.PRM.ENRL': 'Primary school enrollment',
    'SE.SEC.ENRL': 'Secondary school enrollment',
    'SE.XPD.TOTL.GD.ZS': 'Education expenditure (% GDP)',
    'SE.ENR.PRIM.FM.ZS': 'Primary gender parity index',

    # Labor Market - General
    'SL.TLF.CACT.ZS': 'Labor participation rate',
    'SL.UEM.TOTL.ZS': 'Unemployment rate',
    'SL.UEM.1524.ZS': 'Youth unemployment rate',

    # Labor Market - Gender
    'SL.TLF.CACT.MA.ZS': 'Male labor participation',
    'SL.TLF.CACT.FE.ZS': 'Female labor participation',
    'SL.UEM.TOTL.MA.ZS': 'Male unemployment rate',
    'SL.UEM.TOTL.FE.ZS': 'Female unemployment rate',

    # Labor Market - Sectoral Employment
    'SL.IND.EMPL.ZS': 'Industrial employment (%)',
    'SL.AGR.EMPL.ZS': 'Agricultural employment (%)',
    'SL.SRV.EMPL.ZS': 'Services employment (%)',

    # Trade & Finance
    'NE.EXP.GNFS.ZS': 'Exports (% GDP)',
    'NE.IMP.GNFS.ZS': 'Imports (% GDP)',
    'BX.KLT.DINV.WD.GD.ZS': 'Foreign direct investment',
    'FR.INR.DPST': 'Deposit interest rate',
    'BN.CAB.XOKA.GD.ZS': 'Current account balance',
    'PA.NUS.FCRF': 'Exchange rate',

    # Infrastructure & Energy
    'EG.ELC.ACCS.ZS': 'Total electricity access',
    'EG.ELC.ACCS.RU.ZS': 'Rural electricity access',
    'EG.ELC.ACCS.UR.ZS': 'Urban electricity access',
    'IT.NET.USER.ZS': 'Internet users (%)',
    'EG.USE.COMM.FO.ZS': 'Fossil fuel consumption',
    'EG.FEC.RNEW.ZS': 'Renewable energy',
    'EG.ELC.RNEW.ZS': 'Renewable electricity',

    # Agriculture & Land
    'AG.LND.AGRI.ZS': 'Agricultural land (%)',
    'AG.LND.FRST.ZS': 'Forest area (% land)',
    'AG.YLD.CREL.KG': 'Cereal yield (kg/hectare)',
    'AG.PRD.CROP.XD': 'Crop production index',

    # Water & Environment
    'ER.H2O.FWTL.K3': 'Total freshwater withdrawal',
    'ER.H2O.FWST.ZS': 'Freshwater withdrawal',
    'EN.ATM.PM25.MC.M3': 'PM2.5 air pollution',

    # Governance & Social
    'SG.GEN.PARL.ZS': 'Female parliamentary seats (%)',

    # Engineered Features
    'year_feature': 'Year',
    'urbanization_rate': 'Urbanization rate',
    'log_gdp_per_capita': 'Log GDP per capita',
    'trade_openness': 'Trade openness',
    'health_to_edu_ratio': 'Health to education ratio',
    'gender_labor_gap': 'Gender labor gap',
    'economic_diversity': 'Economic diversity',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_display_name(feature_code: str) -> str:
    """
    Get human-readable display name for a feature code.

    Parameters:
    -----------
    feature_code : str
        World Bank indicator code or engineered feature name

    Returns:
    --------
    str
        Human-readable display name

    Examples:
    ---------
    >>> get_display_name('EG.ELC.ACCS.RU.ZS')
    'Rural electricity access'

    >>> get_display_name('trade_openness')
    'Trade openness'

    >>> get_display_name('UNKNOWN_CODE')
    'Unknown Code'
    """
    # Try exact match first
    if feature_code in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[feature_code]

    # Try case-insensitive match
    feature_upper = feature_code.upper()
    for key, value in FEATURE_NAME_MAPPING.items():
        if key.upper() == feature_upper:
            return value

    # Fallback: clean up the code and return
    return feature_code.replace('_', ' ').title()


