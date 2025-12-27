"""
World Bank Indicators Configuration

This module contains the list of World Bank indicator codes to fetch
for GINI coefficient prediction. Each indicator represents a socioeconomic
metric that may influence income inequality.

Indicator categories:
- Economic indicators (GDP, trade, investment)
- Population & demographics
- Health expenditure
- Education
- Labor & employment
- Infrastructure & technology
- Environment & agriculture
- Gender equality
"""

# World Bank API indicator codes
# Format: 'INDICATOR.CODE' with descriptive comments
WORLD_BANK_INDICATORS = [
    # Target variable
    'SI.POV.GINI',         # GINI coefficient (TARGET VARIABLE - what we're predicting)

    # Economic indicators
    'NY.GDP.MKTP.CD',      # GDP (current US$)
    'NY.GDP.MKTP.KD.ZG',   # GDP growth (annual %)
    'NY.GDP.PCAP.CD',      # GDP per capita (current US$)
    'NY.GDP.PCAP.PP.CD',   # GDP per capita, PPP (current international $)
    'NV.AGR.TOTL.ZS',      # Agriculture, forestry, and fishing, value added (% of GDP)
    'NV.IND.TOTL.ZS',      # Industry (including construction), value added (% of GDP)
    'NV.SRV.TOTL.ZS',      # Services, value added (% of GDP)

    # Population & demographics
    'SP.POP.TOTL',         # Population, total
    'SP.URB.TOTL',         # Urban population
    'SP.URB.GROW',         # Urban population growth (annual %)
    'SP.RUR.TOTL',         # Rural population
    'SP.POP.DPND',         # Age dependency ratio (% of working-age population)
    'SP.DYN.TFRT.IN',      # Fertility rate, total (births per woman)

    # Health expenditure
    'SH.XPD.CHEX.GD.ZS',   # Current health expenditure (% of GDP)
    'SH.XPD.CHEX.PC.CD',   # Current health expenditure per capita (current US$)
    'SH.XPD.GHED.GD.ZS',   # Domestic general government health expenditure (% of GDP)
    'SH.XPD.PVTD.CH.ZS',   # Domestic private health expenditure (% of current health expenditure)

    # Education
    'SE.PRM.ENRL',         # School enrollment, primary
    'SE.SEC.ENRL',         # School enrollment, secondary
    'SE.XPD.TOTL.GD.ZS',   # Government expenditure on education, total (% of GDP)

    # Labor & employment
    'SL.TLF.CACT.ZS',      # Labor force participation rate, total (% of total population ages 15+)
    'SL.TLF.CACT.MA.ZS',   # Labor force participation rate, male (% of male population ages 15+)
    'SL.TLF.CACT.FE.ZS',   # Labor force participation rate, female (% of female population ages 15+)
    'SL.UEM.TOTL.ZS',      # Unemployment, total (% of total labor force)
    'SL.UEM.TOTL.MA.ZS',   # Unemployment, male (% of male labor force)
    'SL.UEM.TOTL.FE.ZS',   # Unemployment, female (% of female labor force)
    'SL.UEM.1524.ZS',      # Unemployment, youth total (% of total labor force ages 15-24)
    'SL.AGR.EMPL.ZS',      # Employment in agriculture (% of total employment)
    'SL.IND.EMPL.ZS',      # Employment in industry (% of total employment)
    'SL.SRV.EMPL.ZS',      # Employment in services (% of total employment)

    # Trade & finance
    'NE.EXP.GNFS.ZS',      # Exports of goods and services (% of GDP)
    'NE.IMP.GNFS.ZS',      # Imports of goods and services (% of GDP)
    'BX.KLT.DINV.WD.GD.ZS',# Foreign direct investment, net inflows (% of GDP)
    'BN.CAB.XOKA.GD.ZS',   # Current account balance (% of GDP)
    'FR.INR.DPST',         # Deposit interest rate (%)
    'PA.NUS.FCRF',         # Official exchange rate (LCU per US$, period average)

    # Infrastructure & technology
    'IT.NET.USER.ZS',      # Individuals using the Internet (% of population)
    'EG.ELC.ACCS.ZS',      # Access to electricity (% of population)
    'EG.ELC.ACCS.UR.ZS',   # Access to electricity, urban (% of urban population)
    'EG.ELC.ACCS.RU.ZS',   # Access to electricity, rural (% of rural population)

    # Environment
    'AG.LND.FRST.ZS',      # Forest area (% of land area)
    'AG.LND.AGRI.ZS',      # Agricultural land (% of land area)
    'ER.H2O.FWTL.K3',      # Annual freshwater withdrawals, total (billion cubic meters)
    'ER.H2O.FWST.ZS',      # Level of water stress
    'EG.USE.COMM.FO.ZS',   # Fossil fuel energy consumption (% of total)
    'EG.FEC.RNEW.ZS',      # Renewable energy consumption (% of total final energy consumption)
    'EG.ELC.RNEW.ZS',      # Renewable electricity output (% of total electricity output)
    'EN.ATM.PM25.MC.M3',   # PM2.5 air pollution

    # Agriculture
    'AG.PRD.CROP.XD',      # Crop production index
    'AG.YLD.CREL.KG',      # Cereal yield (kg per hectare)

    # Gender equality
    'SG.GEN.PARL.ZS',      # Proportion of seats held by women in national parliaments (%)
    'SE.ENR.PRIM.FM.ZS',   # Ratio of girls to boys in primary and secondary education (%)
]


def get_indicators():
    """
    Returns the list of World Bank indicator codes.

    Returns:
    --------
    list
        List of World Bank indicator codes
    """
    return WORLD_BANK_INDICATORS
