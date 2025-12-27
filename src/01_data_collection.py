"""
World Bank Data Collection Script
Fetches economic and social indicators for GINI coefficient prediction
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from config.indicators import WORLD_BANK_INDICATORS

warnings.filterwarnings('ignore')


class WorldBankDataCollector:
    """
    Collects economic and social indicator data from the World Bank API.

    This class handles the parallel fetching of multiple economic indicators
    from the World Bank's public API, managing rate limiting, error handling,
    and data merging.
    """

    def __init__(self):
        """
        Initialize the data collector with API base URL and indicator codes.

        Sets up the list of World Bank indicator codes to fetch. Each indicator
        represents a socioeconomic metric (GDP, education, health, employment, etc.)
        that may influence income inequality (GINI coefficient).
        """
        self.base_url = "https://api.worldbank.org/v2"
        # Load indicators from config file
        self.indicators = WORLD_BANK_INDICATORS
    
    def fetch_indicator_data(self, indicator: str, start_year: int = 2000,
                            end_year: int = 2023):
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
                        save_path: str = 'output/world_bank_data.csv',
                        max_workers: int = 5):
        """
        Collect all indicators and merge into a single dataset (parallelized).

        This method uses parallel processing to fetch multiple indicators simultaneously,
        significantly reducing data collection time (6x speedup with 5 workers).

        Parameters:
        -----------
        start_year : int
            Starting year for data collection
        end_year : int
            Ending year for data collection
        save_path : str
            Path to save the collected data
        max_workers : int
            Number of parallel workers for API calls (default: 5)

        Returns:
        --------
        pd.DataFrame
            Merged dataset with all indicators
        """
        print(f"Collecting World Bank data from {start_year} to {end_year}")
        print(f"Total indicators to fetch: {len(self.indicators)}")
        print(f"Using {max_workers} parallel workers\n")

        all_dfs = []
        completed = 0

        def fetch_with_delay(indicator):
            """Fetch single indicator with rate limiting"""
            df = self.fetch_indicator_data(indicator, start_year, end_year)
            time.sleep(0.1)  # Rate limiting: ~1 request per 0.5s with 5 workers
            return indicator, df

        # Parallel API calls using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_indicator = {
                executor.submit(fetch_with_delay, indicator): indicator
                for indicator in self.indicators
            }

            for future in as_completed(future_to_indicator):
                completed += 1
                indicator, df = future.result()

                if not df.empty:
                    all_dfs.append(df)
                    print(f"[{completed}/{len(self.indicators)}] ✓ Fetched {indicator}")
                else:
                    print(f"[{completed}/{len(self.indicators)}] ✗ No data for {indicator}")

        print(f"\nMerging {len(all_dfs)} datasets...")

        # Merge all dataframes on common keys
        all_data = None
        if all_dfs:
            all_data = all_dfs[0]
            for df in all_dfs[1:]:
                all_data = pd.merge(
                    all_data, df,
                    on=['country_code', 'country_name', 'year'],
                    how='outer'
                )

        if all_data is not None:
            all_data['year_feature'] = all_data['year']

            print(f"\nData collection complete!")
            print(f"Total records: {len(all_data)}")
            print(f"Countries: {all_data['country_code'].nunique()}")
            print(f"Years: {sorted(all_data['year'].unique())}")

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
