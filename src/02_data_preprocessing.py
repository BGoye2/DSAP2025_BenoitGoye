"""
Data Preprocessing Script
Cleans and prepares World Bank data for machine learning models
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

from config.constants import (
    FEATURE_NAMES_PATH,
    PROCESSED_DATA_PATH,
    TARGET_VARIABLE,
    WORLD_BANK_DATA_PATH,
)
from config.feature_engineering import create_all_engineered_features

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesses World Bank data for machine learning models.

    This class handles data cleaning, missing value imputation, feature engineering,
    and outlier removal to prepare the dataset for GINI coefficient prediction.

    Key responsibilities:
    - Filter rows with GINI coefficient data (target variable)
    - Handle missing values via imputation or column removal
    - Engineer domain-specific features (urbanization rate, trade openness, etc.)
    - Remove statistical outliers using IQR or Z-score methods
    - Scale and standardize features for modeling
    """

    def __init__(self, data_path: str = None):
        """
        Initialize preprocessor with data path.

        Parameters:
        -----------
        data_path : str, optional
            Path to the raw data CSV file from World Bank API
            (defaults to WORLD_BANK_DATA_PATH from constants)
        """
        self.data_path = data_path if data_path else WORLD_BANK_DATA_PATH
        self.data = None
        
    def load_data(self):
        """Load data from CSV"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} records")
        print(f"Shape: {self.data.shape}")
        return self.data
    
    def filter_target_variable(self):
        """
        Filter dataset to only include rows where GINI coefficient is available
        """
        print("\nFiltering for rows with GINI coefficient...")
        initial_count = len(self.data)
        
        # Keep only rows where GINI is not null
        self.data = self.data[self.data[TARGET_VARIABLE].notna()].copy()
        
        final_count = len(self.data)
        print(f"Rows with GINI data: {final_count} (removed {initial_count - final_count})")
        
        return self.data
    
    def handle_missing_values(self, strategy: str = 'knn',
                             threshold: float = 0.5,
                             n_neighbors: int = 5):
        """
        Handle missing values in features

        Parameters:
        -----------
        strategy : str
            Imputation strategy ('median', 'mean', 'knn', or 'drop')
        threshold : float
            Drop columns with missing percentage above this threshold
        n_neighbors : int
            Number of neighbors for KNN imputation (default: 5)

        Returns:
        --------
        pd.DataFrame
            Data with handled missing values
        """
        print(f"\nHandling missing values (strategy: {strategy})...")
        
        # Identify feature columns (exclude identifiers and target)
        id_cols = ['country_code', 'country_name', 'year']
        target_col = TARGET_VARIABLE
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
            imputer = KNNImputer(n_neighbors=n_neighbors)
            self.data[feature_cols] = imputer.fit_transform(self.data[feature_cols])
            print(f"\nImputed missing values using KNN (n_neighbors={n_neighbors})")
            
        elif strategy == 'drop':
            initial_rows = len(self.data)
            self.data = self.data.dropna(subset=feature_cols)
            print(f"\nDropped {initial_rows - len(self.data)} rows with missing values")
        
        print(f"Remaining missing values: {self.data[feature_cols].isnull().sum().sum()}")
        
        return self.data
    
    def create_engineered_features(self):
        """
        Create engineered features from World Bank indicators.

        Engineered features:
        - Urbanization rate: Urban/rural population balance
        - Log GDP per capita: Reduces skewness
        - Trade openness: Exports + imports as % of GDP
        - Health-to-education ratio: Government spending priorities
        - Gender labor gap: Male-female labor participation difference
        - Economic diversity: Shannon entropy of sector distribution

        Returns:
        --------
        pd.DataFrame
            Data with engineered features added
        """
        print("\nCreating engineered features...")

        # Use centralized feature engineering configuration
        self.data = create_all_engineered_features(self.data)

        print(f"Total features after engineering: {len(self.data.columns)}")

        return self.data
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 3.0):
        """
        Remove outliers from the dataset (vectorized).

        Parameters:
        -----------
        method : str
            'iqr' for Interquartile Range or 'zscore' for Z-score
        threshold : float
            IQR multiplier or Z-score threshold

        Returns:
        --------
        pd.DataFrame
            Data without outliers
        """
        print(f"\nRemoving outliers using {method} method...")
        initial_count = len(self.data)

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        id_cols = ['year', 'year_feature']
        feature_cols = [col for col in numeric_cols if col not in id_cols]

        if method == 'iqr':
            Q1 = self.data[feature_cols].quantile(0.25)
            Q3 = self.data[feature_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds = Q1 - threshold * IQR
            upper_bounds = Q3 + threshold * IQR

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
    
    def prepare_for_modeling(self, save_path: str = None):
        """
        Final preparation for modeling

        Parameters:
        -----------
        save_path : str, optional
            Path to save processed data (defaults to PROCESSED_DATA_PATH from constants)

        Returns:
        --------
        tuple
            (X, y, feature_names, data)
        """
        if save_path is None:
            save_path = PROCESSED_DATA_PATH

        print("\nPreparing data for modeling...")
        
        # Separate features and target
        id_cols = ['country_code', 'country_name', 'year']
        target_col = TARGET_VARIABLE
        
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
        feature_names_df.to_csv(FEATURE_NAMES_PATH, index=False)
        print(f"Feature names saved to: {FEATURE_NAMES_PATH}")

        return X, y, feature_cols, self.data


def main(imputation_strategy='knn', imputation_threshold=0.6,
         n_neighbors=5, outlier_method=None, outlier_threshold=3.0):
    """
    Main execution function

    Parameters:
    -----------
    imputation_strategy : str, default='knn'
        Strategy for handling missing values ('knn', 'median', 'mean', or 'drop')
    imputation_threshold : float, default=0.6
        Drop columns with missing percentage above this threshold
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation
    outlier_method : str or None, default=None
        Method for outlier removal ('iqr', 'zscore', or None to skip)
    outlier_threshold : float, default=3.0
        Threshold for outlier detection (IQR multiplier or Z-score)
    """

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data
    preprocessor.load_data()

    # Filter for rows with GINI data
    preprocessor.filter_target_variable()

    # Handle missing values
    preprocessor.handle_missing_values(strategy=imputation_strategy,
                                      threshold=imputation_threshold,
                                      n_neighbors=n_neighbors)

    # Create engineered features
    preprocessor.create_engineered_features()

    # Remove outliers (if specified)
    if outlier_method:
        preprocessor.remove_outliers(method=outlier_method, threshold=outlier_threshold)

    # Prepare for modeling
    X, y, feature_names, data = preprocessor.prepare_for_modeling()

    # Display summary
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    print(f"\nConfiguration:")
    print(f"  • Imputation strategy: {imputation_strategy}")
    if imputation_strategy == 'knn':
        print(f"  • KNN neighbors: {n_neighbors}")
    print(f"  • Imputation threshold: {imputation_threshold}")
    print(f"  • Outlier removal: {outlier_method if outlier_method else 'Disabled'}")
    if outlier_method:
        print(f"  • Outlier threshold: {outlier_threshold}")

    print(f"\nTarget variable (GINI) statistics:")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std: {y.std():.2f}")
    print(f"Min: {y.min():.2f}")
    print(f"Max: {y.max():.2f}")

    print(f"\nFeature list ({len(feature_names)} features):")
    for i, feat in enumerate(feature_names, 1):
        print(f"{i:2d}. {feat}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Data Preprocessing for GINI Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/02_data_preprocessing.py                          # KNN imputation (default)
  python src/02_data_preprocessing.py --strategy median        # Median imputation
  python src/02_data_preprocessing.py --strategy knn --knn-neighbors 10
  python src/02_data_preprocessing.py --outliers iqr           # With outlier removal
  python src/02_data_preprocessing.py --strategy mean --outliers zscore --outlier-threshold 2.5
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default='knn',
        choices=['knn', 'median', 'mean', 'drop'],
        help='Imputation strategy for missing values (default: knn)'
    )

    parser.add_argument(
        '--knn-neighbors',
        type=int,
        default=5,
        help='Number of neighbors for KNN imputation (default: 5)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Drop columns with missing percentage above this threshold (default: 0.6)'
    )

    parser.add_argument(
        '--outliers',
        type=str,
        default=None,
        choices=['iqr', 'zscore', None],
        help='Outlier removal method (default: None - no removal)'
    )

    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=3.0,
        help='Threshold for outlier detection (default: 3.0)'
    )

    args = parser.parse_args()

    main(
        imputation_strategy=args.strategy,
        imputation_threshold=args.threshold,
        n_neighbors=args.knn_neighbors,
        outlier_method=args.outliers,
        outlier_threshold=args.outlier_threshold
    )
