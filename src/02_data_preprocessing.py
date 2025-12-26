"""
Data Preprocessing Script
Cleans and prepares World Bank data for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses World Bank data for modeling"""
    
    def __init__(self, data_path: str = 'output/world_bank_data.csv'):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} records")
        print(f"Shape: {self.data.shape}")
        return self.data
    
    def filter_target_variable(self) -> pd.DataFrame:
        """
        Filter dataset to only include rows where GINI coefficient is available
        """
        print("\nFiltering for rows with GINI coefficient...")
        initial_count = len(self.data)
        
        # Keep only rows where GINI is not null
        self.data = self.data[self.data['SI.POV.GINI'].notna()].copy()
        
        final_count = len(self.data)
        print(f"Rows with GINI data: {final_count} (removed {initial_count - final_count})")
        
        return self.data
    
    def handle_missing_values(self, strategy: str = 'median', 
                             threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values in features
        
        Parameters:
        -----------
        strategy : str
            Imputation strategy ('median', 'mean', 'knn', or 'drop')
        threshold : float
            Drop columns with missing percentage above this threshold
            
        Returns:
        --------
        pd.DataFrame
            Data with handled missing values
        """
        print(f"\nHandling missing values (strategy: {strategy})...")
        
        # Identify feature columns (exclude identifiers and target)
        id_cols = ['country_code', 'country_name', 'year']
        target_col = 'SI.POV.GINI'
        feature_cols = [col for col in self.data.columns 
                       if col not in id_cols and col != target_col]
        
        # Calculate missing percentage for each feature
        missing_pct = self.data[feature_cols].isnull().mean()
        print(f"\nFeatures with >30% missing data:")
        high_missing = missing_pct[missing_pct > 0.3].sort_values(ascending=False)
        print(high_missing)
        
        # Drop columns with too much missing data
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            print(f"\nDropping {len(cols_to_drop)} columns with >{threshold*100}% missing data:")
            print(cols_to_drop)
            self.data = self.data.drop(columns=cols_to_drop)
            feature_cols = [col for col in feature_cols if col not in cols_to_drop]
        
        # Impute remaining missing values
        if strategy in ['median', 'mean']:
            imputer = SimpleImputer(strategy=strategy)
            self.data[feature_cols] = imputer.fit_transform(self.data[feature_cols])
            print(f"\nImputed missing values using {strategy}")
            
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.data[feature_cols] = imputer.fit_transform(self.data[feature_cols])
            print(f"\nImputed missing values using KNN")
            
        elif strategy == 'drop':
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=feature_cols)
            print(f"\nDropped {initial_rows - len(self.data)} rows with missing values")
        
        print(f"Remaining missing values: {self.data[feature_cols].isnull().sum().sum()}")
        
        return self.data
    
    def create_engineered_features(self) -> pd.DataFrame:
        """
        Create additional engineered features
        """
        print("\nCreating engineered features...")
        
        # Urbanization rate
        if 'SP.URB.TOTL' in self.data.columns and 'SP.POP.TOTL' in self.data.columns:
            self.data['urbanization_rate'] = (self.data['SP.URB.TOTL'] / 
                                              self.data['SP.POP.TOTL'] * 100)
        
        # GDP per capita log (to handle skewness)
        if 'NY.GDP.PCAP.CD' in self.data.columns:
            self.data['log_gdp_per_capita'] = np.log1p(self.data['NY.GDP.PCAP.CD'])
        
        # Trade openness (exports + imports as % of GDP)
        if 'NE.EXP.GNFS.ZS' in self.data.columns and 'NE.IMP.GNFS.ZS' in self.data.columns:
            self.data['trade_openness'] = (self.data['NE.EXP.GNFS.ZS'] + 
                                          self.data['NE.IMP.GNFS.ZS'])
        
        # Health to education spending ratio
        if 'SH.XPD.CHEX.GD.ZS' in self.data.columns and 'SE.XPD.TOTL.GD.ZS' in self.data.columns:
            self.data['health_to_edu_ratio'] = (self.data['SH.XPD.CHEX.GD.ZS'] / 
                                                (self.data['SE.XPD.TOTL.GD.ZS'] + 1e-5))
        
        # Gender labor gap
        if 'SL.TLF.CACT.MA.ZS' in self.data.columns and 'SL.TLF.CACT.FE.ZS' in self.data.columns:
            self.data['gender_labor_gap'] = (self.data['SL.TLF.CACT.MA.ZS'] - 
                                            self.data['SL.TLF.CACT.FE.ZS'])
        
        # Economic structure diversity (entropy-like measure)
        if all(col in self.data.columns for col in ['NV.AGR.TOTL.ZS', 'NV.IND.TOTL.ZS', 'NV.SRV.TOTL.ZS']):
            agr = self.data['NV.AGR.TOTL.ZS'] / 100
            ind = self.data['NV.IND.TOTL.ZS'] / 100
            srv = self.data['NV.SRV.TOTL.ZS'] / 100
            
            # Calculate Shannon entropy
            self.data['economic_diversity'] = -(
                agr * np.log(agr + 1e-5) + 
                ind * np.log(ind + 1e-5) + 
                srv * np.log(srv + 1e-5)
            )
        
        print(f"Total features after engineering: {len(self.data.columns)}")
        
        return self.data
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the dataset (vectorized for performance)

        Parameters:
        -----------
        method : str
            'iqr' for Interquartile Range or 'zscore' for Z-score
        threshold : float
            Threshold for outlier detection (IQR multiplier or Z-score)

        Returns:
        --------
        pd.DataFrame
            Data without outliers
        """
        print(f"\nRemoving outliers using {method} method (vectorized)...")
        initial_count = len(self.data)

        # Get numeric columns only
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        id_cols = ['year', 'year_feature']
        feature_cols = [col for col in numeric_cols if col not in id_cols]

        if method == 'iqr':
            # Vectorized computation for all columns at once
            Q1 = self.data[feature_cols].quantile(0.25)
            Q3 = self.data[feature_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds = Q1 - threshold * IQR
            upper_bounds = Q3 + threshold * IQR

            # Create boolean mask for all columns simultaneously
            mask = (
                (self.data[feature_cols] >= lower_bounds) &
                (self.data[feature_cols] <= upper_bounds)
            ).all(axis=1)

            self.data = self.data[mask]

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.data[feature_cols], nan_policy='omit'))
            self.data = self.data[(z_scores < threshold).all(axis=1)]

        final_count = len(self.data)
        print(f"Removed {initial_count - final_count} outlier rows")
        print(f"Remaining records: {final_count}")

        return self.data
    
    def prepare_for_modeling(self, save_path: str = 'output/processed_data.csv') -> tuple:
        """
        Final preparation for modeling
        
        Returns:
        --------
        tuple
            (X, y, feature_names, data)
        """
        print("\nPreparing data for modeling...")
        
        # Separate features and target
        id_cols = ['country_code', 'country_name', 'year']
        target_col = 'SI.POV.GINI'
        
        feature_cols = [col for col in self.data.columns 
                       if col not in id_cols and col != target_col]
        
        X = self.data[feature_cols].values
        y = self.data[target_col].values
        
        print(f"\nFinal dataset shape:")
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Save processed data
        self.data.to_csv(save_path, index=False)
        print(f"\nProcessed data saved to: {save_path}")
        
        # Save feature names
        feature_names_df = pd.DataFrame({'feature': feature_cols})
        feature_names_df.to_csv('output/feature_names.csv', index=False)
        print(f"Feature names saved to: feature_names.csv")
        
        return X, y, feature_cols, self.data
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of the processed data"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        summary = numeric_data.describe()
        return summary


def main():
    """Main execution function"""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('output/world_bank_data.csv')
    
    # Load data
    preprocessor.load_data()
    
    # Filter for rows with GINI data
    preprocessor.filter_target_variable()
    
    # Handle missing values
    preprocessor.handle_missing_values(strategy='median', threshold=0.6)
    
    # Create engineered features
    preprocessor.create_engineered_features()
    
    # Remove outliers (optional - can comment out if you want to keep all data)
    # preprocessor.remove_outliers(method='iqr', threshold=3.0)
    
    # Prepare for modeling
    X, y, feature_names, data = preprocessor.prepare_for_modeling()
    
    # Display summary
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"\nTarget variable (GINI) statistics:")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std: {y.std():.2f}")
    print(f"Min: {y.min():.2f}")
    print(f"Max: {y.max():.2f}")
    
    print(f"\nFeature list ({len(feature_names)} features):")
    for i, feat in enumerate(feature_names, 1):
        print(f"{i:2d}. {feat}")


if __name__ == "__main__":
    main()
