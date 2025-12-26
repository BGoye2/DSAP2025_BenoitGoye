"""
World Bank Data Collection Script
Fetches economic and social indicators for GINI coefficient prediction
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class WorldBankDataCollector:
    """Collects data from World Bank API"""
    
    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2"
        self.indicators = [
            'SI.POV.GINI',         # GINI coefficient (TARGET VARIABLE)
            'NY.GDP.MKTP.CD',      # GDP (current US$)
            'NY.GDP.MKTP.KD.ZG',   # GDP growth (annual %)
            'NY.GDP.PCAP.CD',      # GDP per capita (current US$)
            'NY.GDP.PCAP.PP.CD',   # GDP per capita, PPP (current international $)
            'NV.AGR.TOTL.ZS',      # Agriculture, forestry, and fishing, value added (% of GDP)
            'NV.IND.TOTL.ZS',      # Industry (including construction), value added (% of GDP)
            'NV.SRV.TOTL.ZS',      # Services, value added (% of GDP)
            'SP.POP.TOTL',         # Population, total
            'SP.URB.TOTL',         # Urban population
            'SP.URB.GROW',         # Urban population growth (annual %)
            'SP.RUR.TOTL',         # Rural population
            'SP.POP.DPND',         # Age dependency ratio (% of working-age population)
            'SP.DYN.TFRT.IN',      # Fertility rate, total (births per woman)
            'SH.XPD.CHEX.GD.ZS',   # Current health expenditure (% of GDP)
            'SH.XPD.CHEX.PC.CD',   # Current health expenditure per capita (current US$)
            'SH.XPD.GHED.GD.ZS',   # Domestic general government health expenditure (% of GDP)
            'SH.XPD.PVTD.CH.ZS',   # Domestic private health expenditure (% of current health expenditure)
            'SE.PRM.ENRL',         # School enrollment, primary
            'SE.SEC.ENRL',         # School enrollment, secondary
            'SE.XPD.TOTL.GD.ZS',   # Government expenditure on education, total (% of GDP)
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
            'NE.EXP.GNFS.ZS',      # Exports of goods and services (% of GDP)
            'NE.IMP.GNFS.ZS',      # Imports of goods and services (% of GDP)
            'BX.KLT.DINV.WD.GD.ZS',# Foreign direct investment, net inflows (% of GDP)
            'BN.CAB.XOKA.GD.ZS',   # Current account balance (% of GDP)
            'FR.INR.DPST',         # Deposit interest rate (%)
            'PA.NUS.FCRF',         # Official exchange rate (LCU per US$, period average)
            'IT.NET.USER.ZS',      # Individuals using the Internet (% of population)
            'EG.ELC.ACCS.ZS',      # Access to electricity (% of population)
            'EG.ELC.ACCS.UR.ZS',   # Access to electricity, urban (% of urban population)
            'EG.ELC.ACCS.RU.ZS',   # Access to electricity, rural (% of rural population)
            'AG.LND.FRST.ZS',      # Forest area (% of land area)
            'AG.LND.AGRI.ZS',      # Agricultural land (% of land area)
            'ER.H2O.FWTL.K3',      # Annual freshwater withdrawals, total (billion cubic meters)
            'ER.H2O.FWST.ZS',      # Level of water stress
            'EG.USE.COMM.FO.ZS',   # Fossil fuel energy consumption (% of total)
            'EG.FEC.RNEW.ZS',      # Renewable energy consumption (% of total final energy consumption)
            'EG.ELC.RNEW.ZS',      # Renewable electricity output (% of total electricity output)
            'EN.ATM.PM25.MC.M3',   # PM2.5 air pollution
            'AG.PRD.CROP.XD',      # Crop production index
            'AG.YLD.CREL.KG',      # Cereal yield (kg per hectare)
            'SG.GEN.PARL.ZS',      # Proportion of seats held by women in national parliaments (%)
            'SE.ENR.PRIM.FM.ZS',   # Ratio of girls to boys in primary and secondary education (%)
        ]
    
    def fetch_indicator_data(self, indicator: str, start_year: int = 2000, 
                            end_year: int = 2023) -> pd.DataFrame:
        """
        Fetch data for a single indicator from World Bank API
        
        Parameters:
        -----------
        indicator : str
            World Bank indicator code
        start_year : int
            Starting year for data collection
        end_year : int
            Ending year for data collection
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with country, year, and indicator value
        """
        url = f"{self.base_url}/country/all/indicator/{indicator}"
        params = {
            'date': f'{start_year}:{end_year}',
            'format': 'json',
            'per_page': 20000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) < 2 or not data[1]:
                print(f"No data available for {indicator}")
                return pd.DataFrame()
            
            records = []
            for entry in data[1]:
                if entry['value'] is not None:
                    records.append({
                        'country_code': entry['countryiso3code'],
                        'country_name': entry['country']['value'],
                        'year': int(entry['date']),
                        indicator: float(entry['value'])
                    })
            
            df = pd.DataFrame(records)
            print(f"Fetched {len(df)} records for {indicator}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {indicator}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error for {indicator}: {e}")
            return pd.DataFrame()
    
    def collect_all_data(self, start_year: int = 2000, end_year: int = 2023,
                        save_path: str = 'output/world_bank_data.csv') -> pd.DataFrame:
        """
        Collect all indicators and merge into a single dataset
        
        Parameters:
        -----------
        start_year : int
            Starting year for data collection
        end_year : int
            Ending year for data collection
        save_path : str
            Path to save the collected data
            
        Returns:
        --------
        pd.DataFrame
            Merged dataset with all indicators
        """
        print(f"Collecting World Bank data from {start_year} to {end_year}")
        print(f"Total indicators to fetch: {len(self.indicators)}\n")
        
        all_data = None
        
        for i, indicator in enumerate(self.indicators, 1):
            print(f"[{i}/{len(self.indicators)}] Fetching {indicator}...")
            
            df = self.fetch_indicator_data(indicator, start_year, end_year)
            
            if df.empty:
                continue
            
            if all_data is None:
                all_data = df
            else:
                all_data = pd.merge(
                    all_data, 
                    df, 
                    on=['country_code', 'country_name', 'year'],
                    how='outer'
                )
            
            # Rate limiting to be respectful to the API
            time.sleep(0.5)
        
        if all_data is not None:
            # Add year as a feature
            all_data['year_feature'] = all_data['year']
            
            print(f"\nData collection complete!")
            print(f"Total records: {len(all_data)}")
            print(f"Countries: {all_data['country_code'].nunique()}")
            print(f"Years: {sorted(all_data['year'].unique())}")
            
            # Save to CSV
            all_data.to_csv(save_path, index=False)
            print(f"\nData saved to: {save_path}")
            
            # Display summary statistics
            print("\nData Summary:")
            print(f"Shape: {all_data.shape}")
            print(f"\nMissing values per column:")
            missing = all_data.isnull().sum()
            missing_pct = (missing / len(all_data) * 100).round(2)
            missing_summary = pd.DataFrame({
                'Missing': missing,
                'Percentage': missing_pct
            }).sort_values('Percentage', ascending=False)
            print(missing_summary[missing_summary['Missing'] > 0])
            
            return all_data
        else:
            print("No data was collected!")
            return pd.DataFrame()


def main():
    """Main execution function"""
    collector = WorldBankDataCollector()
    
    # Collect data from 2000 to 2023
    data = collector.collect_all_data(
        start_year=2000,
        end_year=2023,
        save_path='output/world_bank_data.csv'
    )
    
    if not data.empty:
        print("\n" + "="*50)
        print("DATA COLLECTION SUCCESSFUL")
        print("="*50)
        print(f"\nFirst few rows:")
        print(data.head())
        print(f"\nData types:")
        print(data.dtypes)


if __name__ == "__main__":
    main()
